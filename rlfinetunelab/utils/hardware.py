"""
Hardware diagnostic and VRAM budget calculation utilities.
Provides guardrails for resource-constrained training environments (e.g. free T4 GPU vs local CPU).
"""

import sys
from typing import Dict, Any, Optional
from rlfinetunelab.utils.logging import get_logger

logger = get_logger(__name__)


def get_hardware_summary() -> Dict[str, Any]:
    """Inspects available compute devices, CUDA capability, and memory budget."""
    info: Dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "total_vram_gb": 0.0,
        "recommended_precision": "fp32",
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_count"] = torch.cuda.device_count()
            total_vram = 0.0
            for i in range(info["device_count"]):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024 ** 3)
                total_vram += vram_gb
                info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(vram_gb, 2),
                    "major": props.major,
                    "minor": props.minor,
                })
            info["total_vram_gb"] = round(total_vram, 2)
            # Recommend bfloat16 for Ampere+ (major >= 8), float16 for Turing/Pascal (T4 is 7.5)
            first_major = info["devices"][0]["major"]
            if first_major >= 8:
                info["recommended_precision"] = "bfloat16"
            else:
                info["recommended_precision"] = "float16"
    except ImportError:
        pass

    return info


def validate_training_hardware(is_4bit: bool = True, model_size_b: float = 0.5) -> None:
    """Logs warnings or guidance based on detected hardware profile."""
    hw = get_hardware_summary()
    if not hw["cuda_available"]:
        logger.warning(
            "NO GPU DETECTED! Running on CPU. Training Large/Small Language Models on CPU is extremely slow. "
            "For full training runs, please execute on cloud GPU (e.g., Google Colab T4/A100 or Lambda Labs)."
        )
        return

    vram = hw["total_vram_gb"]
    dev_name = hw["devices"][0]["name"] if hw["devices"] else "Unknown GPU"
    logger.info("Hardware verified: %d GPU(s) detected. Primary: %s (Total VRAM: %.1f GB)", hw["device_count"], dev_name, vram)

    if vram < 15.0 and not is_4bit and model_size_b >= 1.5:
        logger.warning(
            "Caution: GPU has %.1f GB VRAM. Training a %.1fB model in 16-bit without 4-bit quantization "
            "may trigger CUDA Out-Of-Memory (OOM). 4-bit QLoRA is strongly recommended.",
            vram, model_size_b
        )
