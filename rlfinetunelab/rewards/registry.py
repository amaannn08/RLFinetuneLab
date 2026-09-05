"""
Reward function registry for dynamic reward configuration in GRPO.
"""

from typing import Dict, Type, List, Optional, Callable
from rlfinetunelab.rewards.base import BaseRewardFunction
from rlfinetunelab.rewards.reasoning_rewards import (
    AccuracyReward,
    FormatReward,
    ThinkingLengthReward,
)
from rlfinetunelab.utils.logging import get_logger

logger = get_logger(__name__)

_REWARD_REGISTRY: Dict[str, Type[BaseRewardFunction]] = {
    "accuracy": AccuracyReward,
    "format": FormatReward,
    "thinking_length": ThinkingLengthReward,
}


def register_reward(name: str):
    """Decorator to register a new reward function subclass."""
    def decorator(cls: Type[BaseRewardFunction]):
        _REWARD_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_reward_function(name: str, weight: float = 1.0, **kwargs) -> BaseRewardFunction:
    """Instantiate a registered reward function by name."""
    clean_name = name.lower()
    if clean_name not in _REWARD_REGISTRY:
        available = list(_REWARD_REGISTRY.keys())
        raise KeyError(f"Reward function '{name}' not found. Available: {available}")
    return _REWARD_REGISTRY[clean_name](name=clean_name, weight=weight, **kwargs)


class CompositeRewardFunction:
    """Combines multiple reward functions and computes weighted total rewards."""

    def __init__(self, reward_funcs: List[BaseRewardFunction]):
        self.reward_funcs = reward_funcs

    def compute_all(
        self,
        prompts: List[str],
        completions: List[str],
        targets: List[str]
    ) -> Dict[str, List[float]]:
        """Computes individual rewards for each registered metric, plus total."""
        results: Dict[str, List[float]] = {}
        total = [0.0] * len(completions)

        for fn in self.reward_funcs:
            sub_rewards = fn(prompts, completions, targets)
            results[fn.name] = sub_rewards
            for i, r in enumerate(sub_rewards):
                total[i] += r

        results["total"] = total
        return results


def build_reward_pipeline(
    reward_names: List[str],
    weights: Optional[List[float]] = None
) -> CompositeRewardFunction:
    """Builds a composite reward pipeline from string names and optional weights."""
    if weights is None:
        weights = [1.0] * len(reward_names)

    funcs: List[BaseRewardFunction] = []
    for name, w in zip(reward_names, weights):
        funcs.append(get_reward_function(name, weight=w))

    logger.info("Initialized reward pipeline with: %s (weights: %s)", reward_names, weights)
    return CompositeRewardFunction(funcs)
