"""
Lightweight in-memory model and tokenizer mock engine.
Enables instant, zero-network, zero-download local CPU smoke testing and unit tests.
Complies strictly with 0-GPU, low-RAM hardware constraints.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any, List
from datasets import Dataset
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)
from tokenizers import Tokenizer, models, pre_tokenizers

from rlfinetunelab.config.schema import (
    PipelineConfig,
    ModelConfig,
    QuantizationConfig,
    LoraConfigSchema,
    SFTTrainingConfig,
    DPOTrainingConfig,
    GRPOTrainingConfig,
    EvalConfig,
    DatasetConfig,
)
from rlfinetunelab.models.lora import build_peft_config
from rlfinetunelab.utils.logging import get_logger

logger = get_logger(__name__)


def create_tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Constructs a deterministic, tiny in-memory tokenizer without HuggingFace Hub downloads."""
    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<think>": 4,
        "</think>": 5,
        "<answer>": 6,
        "</answer>": 7,
    }
    # Add common English letters/numbers
    for ch in "abcdefghijklmnopqrstuvwxyz0123456789 =+-*/:?\n":
        if ch not in vocab:
            vocab[ch] = len(vocab)

    raw_tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    raw_tok.pre_tokenizer = pre_tokenizers.Whitespace()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tok,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
    )
    return tokenizer


def create_tiny_causal_lm(vocab_size: int = 128) -> LlamaForCausalLM:
    """Constructs a minimal 2-layer causal language model (~98k parameters, <1MB RAM)."""
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=3,
    )
    return LlamaForCausalLM(cfg)


def create_synthetic_sft_dataset() -> Dataset:
    """Synthetic conversational data for smoke tests."""
    records = [
        {"text": "User: What is 2+2?\nAssistant: 4"},
        {"text": "User: What is 5*3?\nAssistant: 15"},
        {"text": "User: Hello there.\nAssistant: Hello! How can I assist you today?"},
        {"text": "User: Name a prime number.\nAssistant: 7 is a prime number."},
    ]
    return Dataset.from_list(records)


def create_synthetic_dpo_dataset() -> Dataset:
    """Synthetic preference data for smoke tests."""
    records = [
        {
            "prompt": "What is 2+2?",
            "chosen": "2 + 2 equals 4.",
            "rejected": "I do not know the answer.",
        },
        {
            "prompt": "What is 5*3?",
            "chosen": "5 * 3 equals 15.",
            "rejected": "Maybe 100?",
        },
    ]
    return Dataset.from_list(records)


def create_synthetic_grpo_dataset() -> Dataset:
    """Synthetic reasoning data for smoke tests."""
    records = [
        {
            "prompt": "What is 7 * 8?",
            "target": "56",
        },
        {
            "prompt": "What is 10 + 20?",
            "target": "30",
        },
    ]
    return Dataset.from_list(records)


def run_local_smoke_test() -> Dict[str, Any]:
    """
    Executes an end-to-end CPU smoke verification:
    1. Instantiates in-memory mock model and tokenizer.
    2. Attaches PEFT LoRA adapter.
    3. Runs 1 step of SFT training.
    4. Evaluates perplexity and reasoning generation.
    5. Merges LoRA weights and verifies integrity.
    """
    logger.info("=== Executing RLFinetuneLab Local Smoke Verification ===")
    temp_dir = tempfile.mkdtemp(prefix="rlfinetunelab_smoke_")

    try:
        tokenizer = create_tiny_tokenizer()
        model = create_tiny_causal_lm(vocab_size=len(tokenizer.get_vocab()))
        train_ds = create_synthetic_sft_dataset()

        # Build pipeline config
        cfg = PipelineConfig(
            stage="sft",
            project_name="SmokeTest",
            model=ModelConfig(name_or_path="mock-tiny-llama", torch_dtype="float32", attn_implementation="eager"),
            quantization=QuantizationConfig(load_in_4bit=False),
            lora=LoraConfigSchema(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]),
            sft=SFTTrainingConfig(
                output_dir=os.path.join(temp_dir, "sft_out"),
                num_train_epochs=1.0,
                max_steps=1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=1,
                learning_rate=1e-3,
                max_seq_length=64,
                gradient_checkpointing=False,
                bf16=False,
                fp16=False,
                logging_steps=1,
                save_steps=1,
                report_to=[],
            ),
            eval=EvalConfig(metrics=["perplexity", "generation"], max_new_tokens=16, eval_batch_size=2),
        )

        from rlfinetunelab.trainers.sft_trainer import run_sft_training
        logger.info("Running 1-step SFT smoke training...")
        train_res = run_sft_training(
            config=cfg,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds
        )
        logger.info("SFT smoke step completed! Loss: %s", train_res["training_loss"])

        # Run eval
        from rlfinetunelab.evaluation.runner import run_evaluation_pipeline
        logger.info("Running evaluation smoke test...")
        eval_res = run_evaluation_pipeline(
            config=cfg,
            model=model,
            tokenizer=tokenizer,
            test_prompts=["User: What is 2+2?\nAssistant:", "User: What is 5*3?\nAssistant:"],
            test_targets=["4", "15"],
            output_dir=os.path.join(temp_dir, "eval_out")
        )
        logger.info("Smoke evaluation completed! Perplexity: %s", eval_res.get("perplexity", {}).get("perplexity"))

        # Save base model and adapter to test merge
        base_dir = os.path.join(temp_dir, "base_model")
        model.save_pretrained(base_dir)
        tokenizer.save_pretrained(base_dir)

        adapter_dir = train_res["adapter_dir"]
        merged_dir = os.path.join(temp_dir, "merged_model")

        from rlfinetunelab.export.merge_lora import merge_lora_to_base
        logger.info("Testing LoRA adapter merge...")
        merged_path = merge_lora_to_base(
            base_model_name_or_path=base_dir,
            adapter_path=adapter_dir,
            output_dir=merged_dir,
            torch_dtype="float32",
            device_map="cpu",
        )

        assert os.path.exists(os.path.join(merged_dir, "config.json"))
        logger.info("=== Local Smoke Test PASSED 100%% Successfully! ===")
        return {
            "status": "success",
            "train_loss": train_res["training_loss"],
            "eval": eval_res,
            "merged_path": str(merged_path),
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
