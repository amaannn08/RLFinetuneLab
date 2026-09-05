"""
Unified command-line interface for RLFinetuneLab.
Provides training, evaluation, export, serving, and verification subcommands.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from rlfinetunelab.config.loader import load_config
from rlfinetunelab.utils.logging import get_logger
from rlfinetunelab.utils.hardware import get_hardware_summary

logger = get_logger("rlfinetune")


def cmd_train(args: argparse.Namespace) -> None:
    """Dispatches training stage (SFT, DPO, or GRPO) from YAML config."""
    config = load_config(args.config, overrides=args.set)
    stage = args.stage or config.stage

    logger.info("Executing pipeline stage: '%s' for project: '%s'", stage, config.project_name)

    if stage == "sft":
        from rlfinetunelab.trainers.sft_trainer import run_sft_training
        run_sft_training(config)
    elif stage == "dpo":
        from rlfinetunelab.trainers.dpo_trainer import run_dpo_training
        run_dpo_training(config)
    elif stage == "grpo":
        from rlfinetunelab.trainers.grpo_trainer import run_grpo_training
        run_grpo_training(config)
    else:
        raise ValueError(f"Unknown training stage: '{stage}'. Must be sft, dpo, or grpo.")


def cmd_eval(args: argparse.Namespace) -> None:
    """Executes evaluation harness."""
    config = load_config(args.config, overrides=args.set)
    from rlfinetunelab.evaluation.runner import run_evaluation_pipeline
    run_evaluation_pipeline(config, output_dir=args.output_dir)


def cmd_export(args: argparse.Namespace) -> None:
    """Merges LoRA adapter weights into the base model."""
    from rlfinetunelab.export.merge_lora import merge_lora_to_base
    merge_lora_to_base(
        base_model_name_or_path=args.base_model,
        adapter_path=args.adapter,
        output_dir=args.output_dir,
        torch_dtype=args.torch_dtype,
    )


def cmd_export_gguf(args: argparse.Namespace) -> None:
    """Converts a merged HuggingFace model to quantized GGUF format for llama.cpp."""
    from rlfinetunelab.export.convert_gguf import convert_to_gguf
    convert_to_gguf(
        model_dir=args.model_dir,
        output_gguf_path=args.output_file,
        quantization_type=args.quant_type,
    )


def cmd_validate_config(args: argparse.Namespace) -> None:
    """Validates configuration file syntax and schema."""
    config = load_config(args.config, overrides=args.set)
    logger.info("Configuration is 100%% VALID!")
    logger.info("Stage: %s | Model: %s | LoRA Rank: %d",
                config.stage, config.model.name_or_path, config.lora.r)


def cmd_hardware(args: argparse.Namespace) -> None:
    """Displays hardware profile and VRAM budget diagnostics."""
    hw = get_hardware_summary()
    logger.info("=== RLFinetuneLab Hardware Diagnostic ===")
    logger.info("CUDA Available: %s", hw["cuda_available"])
    logger.info("GPU Devices Detected: %d", hw["device_count"])
    for dev in hw["devices"]:
        logger.info("  [%d] %s (%.1f GB VRAM, Compute Capability %d.%d)",
                    dev["index"], dev["name"], dev["vram_gb"], dev["major"], dev["minor"])
    logger.info("Recommended Precision: %s", hw["recommended_precision"])


def cmd_smoke(args: argparse.Namespace) -> None:
    """Runs fast local CPU smoke test using in-memory mock models."""
    from rlfinetunelab.mock.dummy_models import run_local_smoke_test
    run_local_smoke_test()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rlfinetune",
        description="RLFinetuneLab: Config-driven RL fine-tuning pipeline for Small Language Models (SFT, DPO, GRPO)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    p_train = subparsers.add_parser("train", help="Run training stage defined in config (SFT, DPO, GRPO)")
    p_train.add_argument("--config", "-c", required=True, help="Path to pipeline YAML configuration file")
    p_train.add_argument("--stage", choices=["sft", "dpo", "grpo"], default=None, help="Override stage in config")
    p_train.add_argument("--set", action="append", help="Override config fields (e.g. --set sft.learning_rate=1e-4)")
    p_train.set_defaults(func=cmd_train)

    # Eval
    p_eval = subparsers.add_parser("eval", help="Run evaluation harness on trained checkpoint")
    p_eval.add_argument("--config", "-c", required=True, help="Path to evaluation YAML configuration")
    p_eval.add_argument("--output-dir", default="outputs/eval", help="Directory to save evaluation results")
    p_eval.add_argument("--set", action="append", help="Override config fields")
    p_eval.set_defaults(func=cmd_eval)

    # Export
    p_export = subparsers.add_parser("export", help="Merge LoRA adapter into base model weights")
    p_export.add_argument("--base-model", required=True, help="Base HuggingFace repo ID or path")
    p_export.add_argument("--adapter", required=True, help="Path to saved LoRA adapter checkpoint")
    p_export.add_argument("--output-dir", default="outputs/merged_model", help="Directory for merged model")
    p_export.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p_export.set_defaults(func=cmd_export)

    # Export GGUF
    p_gguf = subparsers.add_parser("export-gguf", help="Convert merged model to GGUF format")
    p_gguf.add_argument("--model-dir", default="outputs/merged_model", help="Directory containing merged HF model")
    p_gguf.add_argument("--output-file", default="outputs/model_q4_k_m.gguf", help="Output GGUF file path")
    p_gguf.add_argument("--quant-type", default="Q4_K_M", help="llama.cpp quantization type (Q4_K_M, Q8_0, etc.)")
    p_gguf.set_defaults(func=cmd_export_gguf)

    # Validate Config
    p_val = subparsers.add_parser("validate-config", help="Validate YAML configuration schema")
    p_val.add_argument("--config", "-c", required=True, help="Path to configuration file")
    p_val.add_argument("--set", action="append", help="Override config fields")
    p_val.set_defaults(func=cmd_validate_config)

    # Hardware
    p_hw = subparsers.add_parser("hardware", help="Display local hardware profile and memory recommendations")
    p_hw.set_defaults(func=cmd_hardware)

    # Smoke Test
    p_smoke = subparsers.add_parser("smoke", help="Run local lightweight CPU smoke test (zero download)")
    p_smoke.set_defaults(func=cmd_smoke)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
