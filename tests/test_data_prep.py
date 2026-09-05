from rlfinetunelab.data.formatters import (
    format_alpaca_to_chatml,
    format_sharegpt_to_chatml,
    apply_chatml_template,
)
from rlfinetunelab.data.dpo_builder import build_dpo_pair, filter_dpo_pairs, is_valid_dpo_pair
from rlfinetunelab.data.grpo_dataset import format_reasoning_prompt, prepare_grpo_dataset


def test_alpaca_formatting():
    sample = {"instruction": "Calculate square of 9", "input": "", "output": "81"}
    chatml = format_alpaca_to_chatml(sample, system_prompt="Test Sys")
    assert len(chatml) == 3
    assert chatml[0]["role"] == "system"
    assert chatml[1]["role"] == "user"
    assert "square of 9" in chatml[1]["content"]
    assert chatml[2]["role"] == "assistant"
    assert chatml[2]["content"] == "81"


def test_sharegpt_formatting():
    sample = {
        "conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello there"},
        ]
    }
    chatml = format_sharegpt_to_chatml(sample)
    assert len(chatml) == 2
    assert chatml[0]["role"] == "user"
    assert chatml[1]["role"] == "assistant"


def test_chatml_render():
    messages = [
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI is artificial intelligence."},
    ]
    rendered = apply_chatml_template(messages, add_generation_prompt=True)
    assert "<|im_start|>user\nWhat is AI?<|im_end|>\n" in rendered
    assert rendered.endswith("<|im_start|>assistant\n")


def test_dpo_filtering():
    valid = {
        "prompt": "Solve 2+2",
        "chosen": "The answer to 2+2 is 4.",
        "rejected": "I do not know the answer.",
    }
    assert is_valid_dpo_pair(valid, min_length=1)

    identical = {
        "prompt": "Solve 2+2",
        "chosen": "The answer is 4",
        "rejected": "The answer is 4",
    }
    assert not is_valid_dpo_pair(identical)

    empty = {"prompt": "", "chosen": "ok", "rejected": "no"}
    assert not is_valid_dpo_pair(empty)


def test_grpo_prompt_formatting():
    p = format_reasoning_prompt("What is 10 divided by 2?")
    assert "<think>" in p
    assert "User: What is 10 divided by 2?" in p
