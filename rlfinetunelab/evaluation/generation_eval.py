"""
Generation quality and verifiable reasoning benchmark evaluation.
"""

from typing import List, Dict, Any, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from rlfinetunelab.rewards.reasoning_rewards import (
    extract_answer_content,
    normalize_answer_string,
    FormatReward,
)
from rlfinetunelab.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    targets: Optional[List[str]] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 4,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates completions for a test set of prompts and computes reasoning metrics.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    format_fn = FormatReward()
    completions: List[str] = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            }
            if temperature > 0.0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
            else:
                gen_kwargs["do_sample"] = False

            output_ids = model.generate(**inputs, **gen_kwargs)

        input_lengths = [len(x) for x in inputs["input_ids"]]
        for j, out_ids in enumerate(output_ids):
            # Only take newly generated tokens
            gen_ids = out_ids[input_lengths[j] :]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            completions.append(text.strip())

    # Calculate metrics
    format_scores = format_fn.compute_rewards(prompts, completions, targets or [""] * len(completions))
    avg_format_score = sum(format_scores) / max(len(format_scores), 1)

    results: Dict[str, Any] = {
        "num_samples": len(prompts),
        "avg_format_score": round(avg_format_score, 4),
        "completions": completions,
    }

    if targets is not None and len(targets) == len(completions):
        correct = 0
        for comp, tgt in zip(completions, targets):
            ext = extract_answer_content(comp)
            if ext and normalize_answer_string(ext) == normalize_answer_string(tgt):
                correct += 1
        acc = correct / max(len(targets), 1)
        results["accuracy"] = round(acc, 4)
        results["correct_count"] = correct
        logger.info("Generation Evaluation: Accuracy = %.2f%% (%d/%d), Format Score = %.2f",
                    acc * 100.0, correct, len(targets), avg_format_score)

    return results
