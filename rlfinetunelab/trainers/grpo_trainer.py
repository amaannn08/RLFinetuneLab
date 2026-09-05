"""
Group Relative Policy Optimization (GRPO) training pipeline.
Implements DeepSeek-R1 / DeepSeekMath style reasoning reinforcement learning with verifiable reward scoring.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import GRPOTrainer, GRPOConfig
from rlfinetunelab.config.schema import PipelineConfig
from rlfinetunelab.models.loader import load_model_and_tokenizer
from rlfinetunelab.models.lora import build_peft_config
from rlfinetunelab.data.loader import load_pipeline_dataset
from rlfinetunelab.rewards.registry import get_reward_function
from rlfinetunelab.trainers.callbacks import MetricsLoggingCallback
from rlfinetunelab.utils.logging import get_logger
from rlfinetunelab.utils.seed import set_seed

logger = get_logger(__name__)


def _create_trl_reward_wrapper(reward_fn) -> Callable:
    """Adapts BaseRewardFunction to TRL GRPOTrainer reward function interface."""
    def wrapper(completions, target=None, **kwargs):
        targets = target if target is not None else kwargs.get("targets", kwargs.get("answer", [""] * len(completions)))
        prompts = kwargs.get("prompts", [""] * len(completions))
        return reward_fn.compute_rewards(prompts=prompts, completions=completions, targets=targets)

    wrapper.__name__ = reward_fn.name
    return wrapper


def run_grpo_training(
    config: PipelineConfig,
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    train_dataset: Optional[Any] = None,
    eval_dataset: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Executes the GRPO reasoning training loop.
    Eliminates the separate Critic model used in PPO by normalizing advantages
    across a sampled group of G candidate completions per prompt.
    """
    grpo_cfg = config.grpo
    if grpo_cfg is None:
        raise ValueError("PipelineConfig.grpo must not be None when running GRPO.")

    set_seed(grpo_cfg.seed)
    output_dir = Path(grpo_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model & tokenizer if not injected
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer(
            model_config=config.model,
            quant_config=config.quantization,
            lora_config=config.lora,
            is_trainable=True,
            padding_side="left",
        )

    # 2. Load dataset if not injected
    if train_dataset is None:
        ds = load_pipeline_dataset(config.dataset, stage="grpo")
        if isinstance(ds, dict) or hasattr(ds, "keys"):
            train_dataset = ds["train"]
            eval_dataset = ds.get("test", None)
        else:
            train_dataset = ds

    peft_config = build_peft_config(config.lora) if config.lora else None
    warmup_steps = max(1, int(grpo_cfg.warmup_ratio * (grpo_cfg.max_steps if grpo_cfg.max_steps > 0 else 100)))

    # 3. Build reward function callables
    reward_weights = grpo_cfg.reward_weights or [1.0] * len(grpo_cfg.reward_funcs)
    trl_reward_callables: List[Callable] = []
    for r_name, r_w in zip(grpo_cfg.reward_funcs, reward_weights):
        base_fn = get_reward_function(r_name, weight=r_w)
        trl_reward_callables.append(_create_trl_reward_wrapper(base_fn))

    # 4. Setup GRPOConfig
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=grpo_cfg.num_train_epochs,
        max_steps=grpo_cfg.max_steps,
        per_device_train_batch_size=grpo_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=grpo_cfg.gradient_accumulation_steps,
        learning_rate=grpo_cfg.learning_rate,
        lr_scheduler_type=grpo_cfg.lr_scheduler_type,
        warmup_steps=warmup_steps,
        weight_decay=grpo_cfg.weight_decay,
        num_generations=grpo_cfg.num_generations,
        max_prompt_length=grpo_cfg.max_prompt_length,
        max_completion_length=grpo_cfg.max_completion_length,
        temperature=grpo_cfg.temperature,
        top_p=grpo_cfg.top_p,
        beta=grpo_cfg.beta,
        epsilon=grpo_cfg.epsilon,
        reward_weights=reward_weights,
        logging_steps=grpo_cfg.logging_steps,
        save_steps=grpo_cfg.save_steps,
        save_total_limit=grpo_cfg.save_total_limit,
        bf16=grpo_cfg.bf16,
        fp16=grpo_cfg.fp16,
        gradient_checkpointing=grpo_cfg.gradient_checkpointing,
        report_to=grpo_cfg.report_to,
        log_completions=True,
    )

    # 5. Initialize GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=trl_reward_callables,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[MetricsLoggingCallback()],
    )

    logger.info("Starting GRPO training: %d prompts, group_size=%d, beta=%.3f, rewards=%s",
                len(train_dataset), grpo_cfg.num_generations, grpo_cfg.beta, grpo_cfg.reward_funcs)

    train_result = trainer.train(resume_from_checkpoint=grpo_cfg.resume_from_checkpoint)

    # 6. Save final adapter & metrics
    final_save_path = output_dir / "final_adapter"
    trainer.save_model(str(final_save_path))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(final_save_path))

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("GRPO training completed! Final adapter saved to: %s", final_save_path)
    return {
        "global_step": train_result.global_step,
        "training_loss": train_result.training_loss,
        "adapter_dir": str(final_save_path),
        "metrics": metrics,
    }
