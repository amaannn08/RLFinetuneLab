"""
Global pytest fixtures with in-memory dummy models and zero network access.
"""

import pytest
import tempfile
import shutil
from rlfinetunelab.mock.dummy_models import (
    create_tiny_tokenizer,
    create_tiny_causal_lm,
    create_synthetic_sft_dataset,
    create_synthetic_dpo_dataset,
    create_synthetic_grpo_dataset,
)
from rlfinetunelab.config.schema import (
    PipelineConfig,
    ModelConfig,
    QuantizationConfig,
    LoraConfigSchema,
    SFTTrainingConfig,
    DPOTrainingConfig,
    GRPOTrainingConfig,
    DatasetConfig,
    EvalConfig,
)


@pytest.fixture
def tiny_tokenizer():
    return create_tiny_tokenizer()


@pytest.fixture
def tiny_model(tiny_tokenizer):
    return create_tiny_causal_lm(vocab_size=len(tiny_tokenizer.get_vocab()))


@pytest.fixture
def sft_dataset():
    return create_synthetic_sft_dataset()


@pytest.fixture
def dpo_dataset():
    return create_synthetic_dpo_dataset()


@pytest.fixture
def grpo_dataset():
    return create_synthetic_grpo_dataset()


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="rlfinetune_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def base_test_config(temp_dir):
    return PipelineConfig(
        stage="sft",
        project_name="PyTestRun",
        model=ModelConfig(name_or_path="mock-tiny", torch_dtype="float32", attn_implementation="eager"),
        quantization=QuantizationConfig(load_in_4bit=False),
        lora=LoraConfigSchema(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]),
        sft=SFTTrainingConfig(
            output_dir=f"{temp_dir}/sft_test",
            num_train_epochs=1.0,
            max_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            max_seq_length=64,
            gradient_checkpointing=False,
            bf16=False,
            fp16=False,
            report_to=[],
        ),
        eval=EvalConfig(metrics=["perplexity", "generation"], max_new_tokens=16, eval_batch_size=2),
    )
