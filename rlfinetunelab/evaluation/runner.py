"""
Orchestrator running full evaluation suite according to EvalConfig.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from rlfinetunelab.config.schema import PipelineConfig
from rlfinetunelab.evaluation.perplexity import compute_perplexity
from rlfinetunelab.evaluation.generation_eval import evaluate_generation
from rlfinetunelab.utils.logging import get_logger

logger = get_logger(__name__)


def run_evaluation_pipeline(
    config: PipelineConfig,
    model=None,
    tokenizer=None,
    test_prompts=None,
    test_targets=None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Runs perplexity and generation evaluation benchmarks and exports report."""
    eval_cfg = config.eval
    metrics_to_run = eval_cfg.metrics if eval_cfg else ["perplexity", "generation"]

    summary: Dict[str, Any] = {}

    if "perplexity" in metrics_to_run and test_prompts:
        summary["perplexity"] = compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=test_prompts,
            batch_size=eval_cfg.eval_batch_size if eval_cfg else 4
        )

    if "generation" in metrics_to_run and test_prompts:
        summary["generation"] = evaluate_generation(
            model=model,
            tokenizer=tokenizer,
            prompts=test_prompts,
            targets=test_targets,
            max_new_tokens=eval_cfg.max_new_tokens if eval_cfg else 512,
            temperature=eval_cfg.temperature if eval_cfg else 0.7,
            top_p=eval_cfg.top_p if eval_cfg else 0.9,
            batch_size=eval_cfg.eval_batch_size if eval_cfg else 4,
        )

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        report_file = out_p / "eval_results.json"
        with open(report_file, "w", encoding="utf-8") as f:
            # Strip raw completions from json summary to keep compact
            serializable = dict(summary)
            if "generation" in serializable and "completions" in serializable["generation"]:
                serializable["generation"] = {k: v for k, v in serializable["generation"].items() if k != "completions"}
            json.dump(serializable, f, indent=2)
        logger.info("Evaluation results saved to: %s", report_file)

    return summary
