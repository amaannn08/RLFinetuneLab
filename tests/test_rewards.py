from rlfinetunelab.rewards.reasoning_rewards import (
    extract_answer_content,
    normalize_answer_string,
    AccuracyReward,
    FormatReward,
    ThinkingLengthReward,
)
from rlfinetunelab.rewards.registry import (
    get_reward_function,
    build_reward_pipeline,
    register_reward,
    BaseRewardFunction,
)


def test_answer_extraction():
    assert extract_answer_content("<answer>42</answer>") == "42"
    assert extract_answer_content("Solution gives \\boxed{128} finally.") == "128"
    assert extract_answer_content("#### 99") == "99"
    assert extract_answer_content("The answer is: 3.14") == "3.14"


def test_answer_normalization():
    assert normalize_answer_string("$1,000.00") == "1000"
    assert normalize_answer_string("  42.0  ") == "42"
    assert normalize_answer_string("hello") == "hello"


def test_accuracy_reward():
    r = AccuracyReward()
    scores = r.compute_rewards(
        prompts=["p1", "p2", "p3"],
        completions=[
            "<answer>100</answer>",
            "The answer is 50",
            "<answer>99</answer>"
        ],
        targets=["100", "50", "100"]
    )
    assert scores == [1.0, 1.0, 0.0]


def test_format_reward():
    r = FormatReward()
    scores = r.compute_rewards(
        prompts=["p1", "p2", "p3"],
        completions=[
            "<think>step 1</think>\n<answer>42</answer>",
            "<think>only think</think>",
            "no tags at all"
        ],
        targets=["", "", ""]
    )
    assert scores[0] == 1.0
    assert scores[1] == 0.5
    assert scores[2] == 0.0


def test_composite_reward_pipeline():
    pipe = build_reward_pipeline(["accuracy", "format"], weights=[2.0, 1.0])
    res = pipe.compute_all(
        prompts=["What is 2+2?"],
        completions=["<think>2+2 is 4</think><answer>4</answer>"],
        targets=["4"]
    )
    # accuracy = 2.0 * 1.0 = 2.0, format = 1.0 * 1.0 = 1.0, total = 3.0
    assert res["accuracy"][0] == 2.0
    assert res["format"][0] == 1.0
    assert res["total"][0] == 3.0
