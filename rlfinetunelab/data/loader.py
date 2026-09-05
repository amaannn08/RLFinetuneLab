"""
Unified dataset ingestion engine handling local files (JSON, JSONL, Parquet) and HF Hub datasets.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datasets import Dataset, DatasetDict, load_dataset
from rlfinetunelab.config.schema import DatasetConfig
from rlfinetunelab.data.formatters import format_alpaca_to_chatml, format_sharegpt_to_chatml
from rlfinetunelab.data.dpo_builder import prepare_dpo_dataset
from rlfinetunelab.data.grpo_dataset import prepare_grpo_dataset
from rlfinetunelab.utils.logging import get_logger

logger = get_logger(__name__)


def _load_local_data(filepath: str) -> Dataset:
    """Loads local JSON, JSONL, or CSV file into a HF Dataset."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Local dataset file not found: {path}")

    ext = path.suffix.lower()
    if ext in (".jsonl", ".ndjson"):
        return load_dataset("json", data_files=str(path), split="train")
    elif ext == ".json":
        # Check if list or dict
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return Dataset.from_list(data)
        elif isinstance(data, dict):
            return Dataset.from_dict(data)
        return load_dataset("json", data_files=str(path), split="train")
    elif ext == ".csv":
        return load_dataset("csv", data_files=str(path), split="train")
    elif ext == ".parquet":
        return load_dataset("parquet", data_files=str(path), split="train")
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def load_pipeline_dataset(
    config: DatasetConfig,
    stage: str = "sft"
) -> Union[Dataset, DatasetDict]:
    """
    Loads and standardizes dataset for SFT, DPO, or GRPO according to the config.
    """
    logger.info("Loading dataset for stage '%s' (Format: %s)...", stage, config.format_type)

    raw_train: Optional[Dataset] = None
    raw_eval: Optional[Dataset] = None

    if config.train_file:
        raw_train = _load_local_data(config.train_file)
        if config.eval_file:
            raw_eval = _load_local_data(config.eval_file)
    elif config.dataset_name:
        hf_dataset = load_dataset(
            config.dataset_name,
            config.dataset_config_name,
            split=config.dataset_split
        )
        if isinstance(hf_dataset, DatasetDict):
            raw_train = hf_dataset.get("train", hf_dataset[list(hf_dataset.keys())[0]])
            raw_eval = hf_dataset.get("test", hf_dataset.get("validation", None))
        else:
            raw_train = hf_dataset
    else:
        raise ValueError("Neither 'train_file' nor 'dataset_name' specified in DatasetConfig.")

    # Apply sample limits if requested
    if config.max_train_samples and raw_train is not None and len(raw_train) > config.max_train_samples:
        raw_train = raw_train.select(range(config.max_train_samples))
    if config.max_eval_samples and raw_eval is not None and len(raw_eval) > config.max_eval_samples:
        raw_eval = raw_eval.select(range(config.max_eval_samples))

    # Perform stage-specific transformations
    if stage == "dpo":
        train_records = raw_train.to_list() if raw_train is not None else []
        processed_train = prepare_dpo_dataset(
            train_records,
            prompt_key=config.prompt_field,
            chosen_key=config.chosen_field,
            rejected_key=config.rejected_field,
            system_prompt=config.system_prompt
        )
        raw_train = Dataset.from_list(processed_train)

        if raw_eval is not None:
            eval_records = raw_eval.to_list()
            processed_eval = prepare_dpo_dataset(
                eval_records,
                prompt_key=config.prompt_field,
                chosen_key=config.chosen_field,
                rejected_key=config.rejected_field,
                system_prompt=config.system_prompt
            )
            raw_eval = Dataset.from_list(processed_eval)

    elif stage == "grpo":
        train_records = raw_train.to_list() if raw_train is not None else []
        processed_train = prepare_grpo_dataset(
            train_records,
            prompt_key=config.prompt_field,
            target_key=config.response_field,
            system_prompt=config.system_prompt
        )
        raw_train = Dataset.from_list(processed_train)

    if raw_eval is not None:
        return DatasetDict({"train": raw_train, "test": raw_eval})
    return raw_train
