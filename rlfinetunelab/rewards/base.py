"""
Base abstraction for verifiable and heuristic reward functions in RL pipelines.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseRewardFunction(ABC):
    """Abstract class defining the contract for reward computation."""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[str],
        completions: List[str],
        targets: List[str],
        **kwargs: Any
    ) -> List[float]:
        """Compute scalar rewards for a batch of candidate completions against targets."""
        pass

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        targets: List[str],
        **kwargs: Any
    ) -> List[float]:
        raw_rewards = self.compute_rewards(prompts, completions, targets, **kwargs)
        return [r * self.weight for r in raw_rewards]
