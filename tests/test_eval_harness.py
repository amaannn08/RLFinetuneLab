from rlfinetunelab.evaluation.perplexity import compute_perplexity
from rlfinetunelab.evaluation.generation_eval import evaluate_generation
from rlfinetunelab.evaluation.pairwise_judge import compute_pairwise_win_rate


def test_perplexity_computation(tiny_model, tiny_tokenizer):
    texts = ["Hello world, this is a test text for perplexity."]
    res = compute_perplexity(tiny_model, tiny_tokenizer, texts, batch_size=1)
    assert res["perplexity"] > 0.0
    assert res["total_tokens"] > 0
    assert res["num_samples"] == 1


def test_generation_eval(tiny_model, tiny_tokenizer):
    prompts = ["What is 2+2?"]
    targets = ["4"]
    res = evaluate_generation(tiny_model, tiny_tokenizer, prompts, targets, max_new_tokens=4, batch_size=1)
    assert res["num_samples"] == 1
    assert "avg_format_score" in res
    assert len(res["completions"]) == 1


def test_pairwise_judge():
    prompts = ["Question 1", "Question 2"]
    out_a = ["<think>r</think><answer>10</answer>", "<think>r</think><answer>20</answer>"]
    out_b = ["Wrong", "Wrong"]
    targets = ["10", "20"]

    res = compute_pairwise_win_rate(prompts, out_a, out_b, targets, name_a="ModelA", name_b="ModelB")
    assert res["ModelA_win_rate"] == 100.0
    assert res["ModelB_win_rate"] == 0.0
    assert res["num_comparisons"] == 2
