"""
Supervised Fine-Tuning (SFT) pipeline using TRL SFTTrainer and PEFT LoRA/QLoRA.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import SFTTrainer, SFTConfig
from rlfinetunelab.config.schema import PipelineConfig
from rlfinetunelab.models.loader import load_model_and_tokenizer
from rlfinetunelab.models.lora import build_peft_config
from rlfinetunelab.data.loader import load_pipeline_dataset
from rlfinetunelab.trainers.callbacks import MetricsLoggingCallback
from rlfinetunelab.utils.logging import get_logger
from rlfinetunelab.utils.seed import set_seed

logger = get_logger(__name__)


def run_sft_training(
    config: PipelineConfig,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    train_dataset: Optional[Any] = None,
    eval_dataset: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Executes the SFT training loop with LoRA/QLoRA, gradient checkpointing,
    cosine learning rate schedule, and structured metric logging.
    """
    sft_cfg = config.sft
    if sft_cfg is None:
        raise ValueError("PipelineConfig.sft must not be None when running SFT.")

    set_seed(sft_cfg.seed)
    output_dir = Path(sft_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model & tokenizer if not injected
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(
            model_config=config.model,
            quant_config=config.quantization,
            lora_config=config.lora,
            is_trainable=True,
            padding_side="right",
        )

    # 2. Load dataset if not injected
    if train_dataset is None:
        ds = load_pipeline_dataset(config.dataset, stage="sft")
        if isinstance(ds, dict) or hasattr(ds, "keys"):
            train_dataset = ds["train"]
            eval_dataset = ds.get("test", None)
        else:
            train_dataset = ds

    # 3. Setup SFTConfig / TrainingArguments
    peft_config = build_peft_config(config.lora)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=sft_cfg.num_train_epochs,
        max_steps=sft_cfg.max_steps,
        per_device_train_batch_size=sft_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=sft_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
        learning_rate=sft_cfg.learning_rate,
        lr_scheduler_type=sft_cfg.lr_scheduler_type,
        warmup_ratio=sft_cfg.warmup_ratio,
        weight_decay=sft_cfg.weight_decay,
        logging_steps=sft_cfg.logging_steps,
        save_steps=sft_cfg.save_steps,
        eval_steps=sft_cfg.eval_steps,
        save_total_limit=sft_cfg.save_total_limit,
        bf16=sft_cfg.bf16,
        fp16=sft_cfg.fp16,
        gradient_checkpointing=sft_cfg.gradient_checkpointing,
        report_to=sft_cfg.report_to,
        max_seq_length=sft_cfg.max_seq_length,
        packing=sft_cfg.packing,
        dataset_text_field=config.dataset.prompt_field if not config.dataset.response_field else None,
    )

    # 4. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[MetricsLoggingCallback()],
    )

    logger.info("Starting SFT training: %d samples, %s epochs, target lr=%.2e",
                len(train_dataset), sft_cfg.num_train_epochs, sft_cfg.learning_rate)

    train_result = trainer.train(resume_from_checkpoint=sft_cfg.resume_from_checkpoint)

    # 5. Save adapter and tokenizer
    final_save_path = output_dir / "final_adapter"
    trainer.save_model(str(final_save_path))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(final_save_path))

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("SFT training completed! Adapter artifacts saved to: %s", final_save_path)
    return {
        "global_step": train_result.global_step,
        "training_loss": train_result.training_loss,
        "adapter_dir": str(final_save_path),
        "metrics": metrics,
    }
