"""
Deterministic, verifiable reward functions for math reasoning and structural format compliance.
Inspired by DeepSeek-R1, DeepSeekMath, and GSM8K verifiable benchmarks.
"""

import re
from typing import List, Dict, Any, Optional
from rlfinetunelab.rewards.base import BaseRewardFunction
from rlfinetunelab.utils.logging import get_logger

logger = get_logger(__name__)


def extract_answer_content(text: str) -> Optional[str]:
    """
    Extracts answer string from either:
    1. <answer>...</answer> tags
    2. LaTeX \\boxed{...}
    3. 'The answer is X' or '#### X'
    4. Fallback: last numerical token
    """
    # 1. XML tag
    answer_tag = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if answer_tag:
        return answer_tag.group(1).strip()

    # 2. Boxed notation
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed[-1].strip()

    # 3. GSM8K "The answer is X" or "#### X"
    gsm_match = re.search(r"(?:####|the answer is:?)\s*([-+]?\d+(?:\.\d+)?|[^\s\.\,]+)", text, re.IGNORECASE)
    if gsm_match:
        return gsm_match.group(1).strip()

    # 4. Fallback: last numerical token
    num_matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if num_matches:
        return num_matches[-1]

    return None


def normalize_answer_string(ans_str: str) -> str:
    """Normalizes an answer string by stripping spaces, commas, and dollar signs."""
    cleaned = ans_str.strip().replace("$", "").replace(",", "").replace("%", "")
    try:
        val = float(cleaned)
        if val.is_integer():
            return str(int(val))
        return f"{val:.4f}".rstrip("0").rstrip(".")
    except ValueError:
        return cleaned.lower()


class AccuracyReward(BaseRewardFunction):
    """
    Verifiable mathematical accuracy reward.
    Gives +1.0 for an exact or numerical equivalence match, 0.0 otherwise.
    """

    def __init__(self, name: str = "accuracy", weight: float = 1.0):
        super().__init__(name=name, weight=weight)

    def compute_rewards(
        self,
        prompts: List[str],
        completions: List[str],
        targets: List[str],
        **kwargs: Any
    ) -> List[float]:
        rewards = []
        for completion, target in zip(completions, targets):
            extracted = extract_answer_content(completion)
            if extracted is None:
                rewards.append(0.0)
                continue

            norm_extracted = normalize_answer_string(extracted)
            norm_target = normalize_answer_string(str(target))

            if norm_extracted == norm_target:
                rewards.append(1.0)
            else:
                try:
                    val_pred = float(norm_extracted)
                    val_true = float(norm_target)
                    if abs(val_pred - val_true) < 1e-4:
                        rewards.append(1.0)
                    else:
                        rewards.append(0.0)
                except ValueError:
                    rewards.append(0.0)

        return rewards


class FormatReward(BaseRewardFunction):
    """
    Format compliance reward.
    Validates structural chain-of-thought XML tags:
    - <think>...</think> (+0.5)
    - <answer>...</answer> (+0.5)
    """

    def __init__(self, name: str = "format", weight: float = 1.0):
        super().__init__(name=name, weight=weight)

    def compute_rewards(
        self,
        prompts: List[str],
        completions: List[str],
        targets: List[str],
        **kwargs: Any
    ) -> List[float]:
        rewards = []
        for completion in completions:
            score = 0.0
            has_think_open = "<think>" in completion
            has_think_close = "</think>" in completion
            if has_think_open and has_think_close:
                if completion.find("<think>") < completion.find("</think>"):
                    score += 0.5

            has_ans_open = "<answer>" in completion
            has_ans_close = "</answer>" in completion
            if has_ans_open and has_ans_close:
                if completion.find("<answer>") < completion.find("</answer>"):
                    score += 0.5

            rewards.append(score)
        return rewards


class ThinkingLengthReward(BaseRewardFunction):
    """
    Encourages thorough reasoning while penalizing runaway repetition.
    Optimal token length range [min_chars, max_chars].
    """

    def __init__(
        self,
        name: str = "thinking_length",
        weight: float = 0.5,
        min_chars: int = 50,
        max_chars: int = 1500
    ):
        super().__init__(name=name, weight=weight)
        self.min_chars = min_chars
        self.max_chars = max_chars

    def compute_rewards(
        self,
        prompts: List[str],
        completions: List[str],
        targets: List[str],
        **kwargs: Any
    ) -> List[float]:
        rewards = []
        for completion in completions:
            think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
            if not think_match:
                rewards.append(0.0)
                continue

            think_len = len(think_match.group(1).strip())
            if think_len < self.min_chars:
                rewards.append(think_len / self.min_chars * 0.5)
            elif think_len > self.max_chars:
                penalty = max(0.0, 1.0 - (think_len - self.max_chars) / 500.0)
                rewards.append(penalty)
            else:
                rewards.append(1.0)

        return rewards
