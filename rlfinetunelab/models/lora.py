"""
PEFT LoRA configuration construction and parameter inspection utilities.
"""

from typing import Dict, Any, Union, List
from peft import LoraConfig, TaskType
from rlfinetunelab.config.schema import LoraConfigSchema
from rlfinetunelab.utils.logging import get_logger

logger = get_logger(__name__)


def build_peft_config(schema: LoraConfigSchema) -> LoraConfig:
    """Builds a PEFT LoraConfig object from our validated Pydantic schema."""
    target_modules: Union[List[str], str]
    if isinstance(schema.target_modules, list):
        target_modules = list(schema.target_modules)
    else:
        target_modules = schema.target_modules

    peft_config = LoraConfig(
        r=schema.r,
        lora_alpha=schema.lora_alpha,
        lora_dropout=schema.lora_dropout,
        target_modules=target_modules,
        bias=schema.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    logger.info("Initialized LoRA config (rank=%d, alpha=%d, dropout=%.2f, targets=%s)",
                schema.r, schema.lora_alpha, schema.lora_dropout, schema.target_modules)
    return peft_config


def get_trainable_parameters_summary(model: Any) -> Dict[str, Any]:
    """Calculates total, trainable, and frozen parameters of a PyTorch/PEFT model."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    percentage = 100.0 * trainable_params / max(all_params, 1)
    summary = {
        "trainable_params": trainable_params,
        "all_params": all_params,
        "trainable_percentage": round(percentage, 4),
    }
    logger.info(
        "Model Parameters: %s trainable / %s total (%.2f%% trainable)",
        f"{trainable_params:,}", f"{all_params:,}", percentage
    )
    return summary
