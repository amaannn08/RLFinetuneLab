"""
Direct Preference Optimization (DPO) alignment pipeline.
Leverages TRL DPOTrainer with parameter-efficient reference model freezing (adapter disabling).
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import DPOTrainer, DPOConfig
from rlfinetunelab.config.schema import PipelineConfig
from rlfinetunelab.models.loader import load_model_and_tokenizer
from rlfinetunelab.models.lora import build_peft_config
from rlfinetunelab.data.loader import load_pipeline_dataset
from rlfinetunelab.trainers.callbacks import MetricsLoggingCallback
from rlfinetunelab.utils.logging import get_logger
from rlfinetunelab.utils.seed import set_seed

logger = get_logger(__name__)


def run_dpo_training(
    config: PipelineConfig,
    model: Optional[PreTrainedModel] = None,
    ref_model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    train_dataset: Optional[Any] = None,
    eval_dataset: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Executes the Direct Preference Optimization (DPO) alignment loop.
    When PEFT/LoRA is active, DPOTrainer handles reference logprobs by disabling the adapter,
    saving up to 50% GPU VRAM by eliminating the need for a separate reference model in memory.
    """
    dpo_cfg = config.dpo
    if dpo_cfg is None:
        raise ValueError("PipelineConfig.dpo must not be None when running DPO.")

    set_seed(dpo_cfg.seed)
    output_dir = Path(dpo_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model & tokenizer if not injected
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(
            model_config=config.model,
            quant_config=config.quantization,
            lora_config=config.lora,
            is_trainable=True,
            padding_side="left", # Left padding is standard for generation/DPO
        )

    # 2. Load dataset if not injected
    if train_dataset is None:
        ds = load_pipeline_dataset(config.dataset, stage="dpo")
        if isinstance(ds, dict) or hasattr(ds, "keys"):
            train_dataset = ds["train"]
            eval_dataset = ds.get("test", None)
        else:
            train_dataset = ds

    peft_config = build_peft_config(config.lora) if config.lora else None

    # 3. Setup DPOConfig
    training_args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=dpo_cfg.num_train_epochs,
        max_steps=dpo_cfg.max_steps,
        per_device_train_batch_size=dpo_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=dpo_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=dpo_cfg.gradient_accumulation_steps,
        learning_rate=dpo_cfg.learning_rate,
        lr_scheduler_type=dpo_cfg.lr_scheduler_type,
        warmup_ratio=dpo_cfg.warmup_ratio,
        weight_decay=dpo_cfg.weight_decay,
        logging_steps=dpo_cfg.logging_steps,
        save_steps=dpo_cfg.save_steps,
        eval_steps=dpo_cfg.eval_steps,
        save_total_limit=dpo_cfg.save_total_limit,
        bf16=dpo_cfg.bf16,
        fp16=dpo_cfg.fp16,
        gradient_checkpointing=dpo_cfg.gradient_checkpointing,
        report_to=dpo_cfg.report_to,
        beta=dpo_cfg.beta,
        loss_type=dpo_cfg.loss_type,
        label_smoothing=dpo_cfg.label_smoothing,
        max_prompt_length=dpo_cfg.max_prompt_length,
        max_length=dpo_cfg.max_seq_length,
    )

    # 4. Initialize DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model, # None enables PEFT adapter-disabling ref mode
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[MetricsLoggingCallback()],
    )

    logger.info("Starting DPO training: %d preference pairs, beta=%.2f, loss_type=%s",
                len(train_dataset), dpo_cfg.beta, dpo_cfg.loss_type)

    train_result = trainer.train(resume_from_checkpoint=dpo_cfg.resume_from_checkpoint)

    # 5. Save adapter & metrics
    final_save_path = output_dir / "final_adapter"
    trainer.save_model(str(final_save_path))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(final_save_path))

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("DPO alignment completed! Final adapter saved to: %s", final_save_path)
    return {
        "global_step": train_result.global_step,
        "training_loss": train_result.training_loss,
        "adapter_dir": str(final_save_path),
        "metrics": metrics,
    }
