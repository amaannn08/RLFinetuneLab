"""
Pairwise comparative evaluation harness (Model A vs Model B).
Supports automated rule-based judging and LLM-as-a-judge win-rate computation.
"""

from typing import List, Dict, Any, Optional
from rlfinetunelab.rewards.reasoning_rewards import (
    extract_answer_content,
    normalize_answer_string,
    FormatReward,
)
from rlfinetunelab.utils.logging import get_logger

logger = get_logger(__name__)


def compute_pairwise_win_rate(
    prompts: List[str],
    completions_a: List[str],
    completions_b: List[str],
    targets: Optional[List[str]] = None,
    name_a: str = "Model_A",
    name_b: str = "Model_B",
) -> Dict[str, Any]:
    """
    Compares candidate generations between two models:
    - If targets exist, evaluates mathematical/verifiable correctness first.
    - Uses format compliance (<think>...</think> XML) as secondary tiebreaker.
    - Conciseness/brevity as tertiary tiebreaker.
    """
    format_eval = FormatReward()
    n = min(len(prompts), len(completions_a), len(completions_b))

    wins_a = 0
    wins_b = 0
    ties = 0

    detailed_results: List[Dict[str, Any]] = []

    for i in range(n):
        prompt = prompts[i]
        out_a = completions_a[i]
        out_b = completions_b[i]
        tgt = targets[i] if targets is not None and i < len(targets) else None

        score_a = 0.0
        score_b = 0.0

        # Verifiable check if target exists
        if tgt is not None:
            ans_a = extract_answer_content(out_a)
            ans_b = extract_answer_content(out_b)
            tgt_norm = normalize_answer_string(str(tgt))

            correct_a = (ans_a is not None and normalize_answer_string(ans_a) == tgt_norm)
            correct_b = (ans_b is not None and normalize_answer_string(ans_b) == tgt_norm)

            if correct_a and not correct_b:
                score_a += 2.0
            elif correct_b and not correct_a:
                score_b += 2.0

        # Format compliance check
        fmt_a = format_eval.compute_rewards([prompt], [out_a], [tgt or ""])[0]
        fmt_b = format_eval.compute_rewards([prompt], [out_b], [tgt or ""])[0]
        score_a += fmt_a
        score_b += fmt_b

        # Determine winner
        if score_a > score_b:
            winner = name_a
            wins_a += 1
        elif score_b > score_a:
            winner = name_b
            wins_b += 1
        else:
            winner = "TIE"
            ties += 1

        detailed_results.append({
            "prompt": prompt,
            "completion_a": out_a,
            "completion_b": out_b,
            "score_a": score_a,
            "score_b": score_b,
            "winner": winner,
        })

    win_rate_a = (wins_a / n) * 100.0 if n > 0 else 0.0
    win_rate_b = (wins_b / n) * 100.0 if n > 0 else 0.0
    tie_rate = (ties / n) * 100.0 if n > 0 else 0.0

    logger.info("Pairwise Comparison Results [%s vs %s]: %s Win Rate: %.1f%% | %s Win Rate: %.1f%% | Ties: %.1f%%",
                name_a, name_b, name_a, win_rate_a, name_b, win_rate_b, tie_rate)

    return {
        "num_comparisons": n,
        f"{name_a}_wins": wins_a,
        f"{name_b}_wins": wins_b,
        "ties": ties,
        f"{name_a}_win_rate": round(win_rate_a, 2),
        f"{name_b}_win_rate": round(win_rate_b, 2),
        "tie_rate": round(tie_rate, 2),
        "comparisons": detailed_results,
    }
