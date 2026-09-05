"""
LoRA adapter merging utility.
Fuses trained low-rank adapter weights into base model parameters for zero-overhead inference.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rlfinetunelab.utils.logging import get_logger

logger = get_logger(__name__)


def merge_lora_to_base(
    base_model_name_or_path: str,
    adapter_path: str,
    output_dir: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True
) -> Path:
    """
    Loads base model in 16-bit precision, mounts PEFT adapter, merges weights,
    and saves consolidated model + tokenizer to output_dir.
    """
    import torch
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model: %s in %s", base_model_name_or_path, torch_dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map if torch.cuda.is_available() else "cpu",
        trust_remote_code=trust_remote_code,
    )

    logger.info("Loading LoRA adapter from: %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging adapter into base model weights...")
    merged_model = model.merge_and_unload()

    logger.info("Saving merged model to: %s", out_path)
    merged_model.save_pretrained(str(out_path), safe_serialization=True)

    # Save tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=trust_remote_code)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(str(out_path))

    logger.info("Merge complete! Consolidated model saved at: %s", out_path)
    return out_path
