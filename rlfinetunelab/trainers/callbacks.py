"""
HuggingFace Trainer callbacks for structured metrics reporting, wandb tracking,
and qualitative sample generation during training.
"""

from typing import List, Optional
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from rlfinetunelab.utils.logging import get_logger

logger = get_logger(__name__)


class MetricsLoggingCallback(TrainerCallback):
    """Logs concise training throughput, step loss, and learning rate."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs
    ):
        if logs is None:
            return

        step = state.global_step
        loss = logs.get("loss", logs.get("eval_loss", None))
        lr = logs.get("learning_rate", None)
        epoch = logs.get("epoch", None)

        msg_parts = [f"Step {step}"]
        if epoch is not None:
            msg_parts.append(f"Epoch {epoch:.2f}")
        if loss is not None:
            msg_parts.append(f"Loss: {loss:.4f}")
        if lr is not None:
            msg_parts.append(f"LR: {lr:.2e}")

        # Additional TRL metrics (reward, margin, kl)
        for k in ["rewards/chosen", "rewards/rejected", "rewards/margins", "reward", "kl"]:
            if k in logs:
                msg_parts.append(f"{k}: {logs[k]:.3f}")

        logger.info(" | ".join(msg_parts))


class SampleGenerationCallback(TrainerCallback):
    """Periodically generates and logs sample outputs from test prompts to inspect training progress."""

    def __init__(
        self,
        tokenizer,
        prompts: List[str],
        eval_steps: int = 100,
        max_new_tokens: int = 128
    ):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.eval_steps = eval_steps
        self.max_new_tokens = max_new_tokens

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        if model is None or self.tokenizer is None or not self.prompts:
            return

        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            logger.info("--- Qualitative Generation Probe (Step %d) ---", state.global_step)
            model.eval()
            import torch
            device = next(model.parameters()).device
            for p in self.prompts[:2]:
                inputs = self.tokenizer(p, return_tensors="pt").to(device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                gen_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                logger.info("Prompt: %s", p[:80])
                logger.info("Output: %s", gen_text[len(p):].strip()[:150])
            model.train()
