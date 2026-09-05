"""
GRPO (Group Relative Policy Optimization) dataset formatting and reasoning prompt templates.
Standardizes prompts for verifiable tasks (Math, Logic, Coding) requiring chain-of-thought XML tags.
"""

from typing import Dict, Any, List, Optional
from rlfinetunelab.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_REASONING_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it.\n"
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think>\n<answer> final answer here </answer>."
)


def format_reasoning_prompt(
    question: str,
    system_prompt: Optional[str] = None
) -> str:
    """Wraps a question into the standard DeepSeek-R1 / GRPO reasoning prompt template."""
    sys_prompt = system_prompt if system_prompt is not None else DEFAULT_REASONING_SYSTEM_PROMPT
    return f"{sys_prompt}\n\nUser: {question.strip()}\nAssistant: <think>"


def prepare_grpo_dataset(
    raw_records: List[Dict[str, Any]],
    prompt_key: str = "prompt",
    target_key: str = "target",
    system_prompt: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Prepares a dataset of prompts and verifiable targets for GRPO sampling.
    """
    formatted: List[Dict[str, Any]] = []
    for row in raw_records:
        prompt_text = str(row.get(prompt_key, row.get("question", ""))).strip()
        target_text = str(row.get(target_key, row.get("answer", ""))).strip()

        if not prompt_text:
            continue

        item = {
            "prompt": format_reasoning_prompt(prompt_text, system_prompt=system_prompt),
            "raw_prompt": prompt_text,
            "target": target_text,
        }
        if "metadata" in row:
            item["metadata"] = row["metadata"]

        formatted.append(item)

    logger.info("Prepared %d GRPO reasoning prompt-target items", len(formatted))
    return formatted
