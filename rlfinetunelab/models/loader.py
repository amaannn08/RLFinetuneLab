"""
Model and tokenizer loading engine supporting QLoRA 4-bit, 8-bit, and full 16-bit precision.
"""

import os
from typing import Tuple, Optional, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from peft import prepare_model_for_kbit_training
from rlfinetunelab.config.schema import ModelConfig, QuantizationConfig, LoraConfigSchema
from rlfinetunelab.models.lora import build_peft_config, get_trainable_parameters_summary
from rlfinetunelab.utils.logging import get_logger
from rlfinetunelab.utils.hardware import validate_training_hardware

logger = get_logger(__name__)


def _resolve_torch_dtype(dtype_str: str) -> Any:
    """Map string representation to torch dtype."""
    import torch
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }
    return mapping.get(dtype_str, torch.bfloat16)


def get_quantization_config(quant_cfg: QuantizationConfig) -> Optional[Any]:
    """Constructs BitsAndBytesConfig for 4-bit or 8-bit quantization."""
    if not (quant_cfg.load_in_4bit or quant_cfg.load_in_8bit):
        return None

    try:
        from transformers import BitsAndBytesConfig
        import torch

        compute_dtype = _resolve_torch_dtype(quant_cfg.bnb_4bit_compute_dtype)
        if quant_cfg.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
            )
        elif quant_cfg.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
    except ImportError:
        logger.warning("bitsandbytes not installed or not supported on this platform. Running in unquantized mode.")
        return None
    return None


def load_tokenizer(
    model_config: ModelConfig,
    padding_side: str = "right"
) -> PreTrainedTokenizer:
    """Loads and configures tokenizer with proper pad/eos tokens and padding side."""
    tok_path = model_config.tokenizer_name_or_path or model_config.name_or_path
    logger.info("Loading tokenizer from: %s (padding_side=%s)", tok_path, padding_side)

    tokenizer = AutoTokenizer.from_pretrained(
        tok_path,
        use_fast=model_config.use_fast_tokenizer,
        trust_remote_code=model_config.trust_remote_code,
        padding_side=padding_side,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


def load_model_and_tokenizer(
    model_config: ModelConfig,
    quant_config: QuantizationConfig,
    lora_config: Optional[LoraConfigSchema] = None,
    is_trainable: bool = True,
    padding_side: str = "right",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Unified entrypoint to load model, apply 4-bit/8-bit QLoRA configs,
    configure gradient checkpointing, and return model + tokenizer.
    """
    validate_training_hardware(is_4bit=quant_config.load_in_4bit)

    tokenizer = load_tokenizer(model_config, padding_side=padding_side)
    bnb_config = get_quantization_config(quant_config)
    torch_dtype = _resolve_torch_dtype(model_config.torch_dtype)

    model_kwargs: dict = {
        "trust_remote_code": model_config.trust_remote_code,
        "torch_dtype": torch_dtype,
    }

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = model_config.device_map or "auto"
    else:
        if model_config.device_map is not None:
            import torch
            if torch.cuda.is_available():
                model_kwargs["device_map"] = model_config.device_map

    # Attention implementation
    if model_config.attn_implementation != "eager":
        model_kwargs["attn_implementation"] = model_config.attn_implementation

    logger.info("Loading CausalLM model from: %s", model_config.name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_config.name_or_path,
        **model_kwargs
    )

    # Resize token embeddings if special tokens were added
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    if is_trainable:
        if quant_config.load_in_4bit or quant_config.load_in_8bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        if lora_config is not None:
            from peft import get_peft_model
            peft_cfg = build_peft_config(lora_config)
            model = get_peft_model(model, peft_cfg)
            get_trainable_parameters_summary(model)

    return model, tokenizer
