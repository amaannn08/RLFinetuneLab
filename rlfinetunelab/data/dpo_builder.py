"""
DPO (Direct Preference Optimization) pair construction and quality filtering utilities.
"""

from typing import Dict, Any, List, Optional
from rlfinetunelab.utils.logging import get_logger

logger = get_logger(__name__)


def build_dpo_pair(
    prompt: str,
    chosen: str,
    rejected: str,
    system_prompt: Optional[str] = None
) -> Dict[str, str]:
    """Constructs a clean DPO dictionary with prompt, chosen, and rejected strings."""
    formatted_prompt = prompt.strip()
    if system_prompt:
        formatted_prompt = f"System: {system_prompt.strip()}\n\nUser: {formatted_prompt}"

    return {
        "prompt": formatted_prompt,
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
    }


def is_valid_dpo_pair(
    pair: Dict[str, str],
    min_length: int = 5,
    max_length: int = 4096,
    min_diff_chars: int = 5
) -> bool:
    """
    Validates a DPO preference pair:
    1. Neither chosen nor rejected are empty or below minimum character length.
    2. Prompt is not empty.
    3. Chosen and rejected are sufficiently distinct (not identical).
    4. Text lengths do not exceed maximum character boundary.
    """
    prompt = pair.get("prompt", "").strip()
    chosen = pair.get("chosen", "").strip()
    rejected = pair.get("rejected", "").strip()

    if len(prompt) < min_length or len(chosen) < min_length or len(rejected) < min_length:
        return False

    if len(prompt) > max_length or len(chosen) > max_length or len(rejected) > max_length:
        return False

    # Check that chosen and rejected are not identical or trivially similar
    if chosen == rejected:
        return False

    if abs(len(chosen) - len(rejected)) == 0 and chosen.lower() == rejected.lower():
        return False

    diff_chars = sum(1 for a, b in zip(chosen, rejected) if a != b) + abs(len(chosen) - len(rejected))
    if diff_chars < min_diff_chars:
        return False

    return True


def filter_dpo_pairs(
    dataset_records: List[Dict[str, str]],
    min_length: int = 5,
    max_length: int = 4096,
    min_diff_chars: int = 5
) -> List[Dict[str, str]]:
    """Filters a list of DPO pairs according to quality heuristics."""
    initial_count = len(dataset_records)
    filtered = [
        p for p in dataset_records
        if is_valid_dpo_pair(p, min_length=min_length, max_length=max_length, min_diff_chars=min_diff_chars)
    ]
    logger.info("Filtered DPO pairs: retained %d of %d (%.1f%%)", len(filtered), initial_count, (len(filtered)/max(initial_count, 1))*100)
    return filtered


def prepare_dpo_dataset(
    raw_records: List[Dict[str, Any]],
    prompt_key: str = "prompt",
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
    system_prompt: Optional[str] = None,
    min_length: int = 5,
    max_length: int = 4096
) -> List[Dict[str, str]]:
    """Transforms raw dictionary records into validated DPO pair format."""
    pairs: List[Dict[str, str]] = []
    for row in raw_records:
        p = str(row.get(prompt_key, ""))
        c = str(row.get(chosen_key, ""))
        r = str(row.get(rejected_key, ""))
        pair = build_dpo_pair(prompt=p, chosen=c, rejected=r, system_prompt=system_prompt)
        pairs.append(pair)

    return filter_dpo_pairs(pairs, min_length=min_length, max_length=max_length)
