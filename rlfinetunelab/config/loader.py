"""
YAML configuration loader with dot-notation CLI overrides and Pydantic validation.
"""

import os
import re
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from rlfinetunelab.config.schema import PipelineConfig
from rlfinetunelab.utils.logging import get_logger

logger = get_logger(__name__)


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand environment variables formatted as ${VAR_NAME} or ${VAR:default}."""
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    elif isinstance(data, str):
        pattern = re.compile(r"\$\{([A-Za-z0-9_]+)(?::([^}]*))?\}")
        def replace(match: re.Match) -> str:
            var_name = match.group(1)
            default_val = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default_val)
        return pattern.sub(replace, data)
    return data


def _apply_override(cfg_dict: Dict[str, Any], key_path: str, value_str: str) -> None:
    """Apply a single key-path override (e.g. 'sft.learning_rate' = '1e-4') into the dictionary."""
    keys = key_path.strip().split(".")
    target = cfg_dict
    for k in keys[:-1]:
        if k not in target or not isinstance(target[k], dict):
            target[k] = {}
        target = target[k]

    final_key = keys[-1]
    # Attempt primitive type casting
    val: Any = value_str
    if value_str.lower() in ("true", "yes"):
        val = True
    elif value_str.lower() in ("false", "no"):
        val = False
    elif value_str.lower() in ("none", "null"):
        val = None
    else:
        try:
            if "." in value_str or "e" in value_str.lower():
                val = float(value_str)
            else:
                val = int(value_str)
        except ValueError:
            # Check for comma-separated list
            if "," in value_str:
                val = [item.strip() for item in value_str.split(",")]
            else:
                val = value_str

    target[final_key] = val
    logger.debug("Applied config override: %s = %s", key_path, val)


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[List[str]] = None
) -> PipelineConfig:
    """
    Load YAML configuration, apply environment variables and CLI overrides,
    and validate against the Pydantic schema.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    raw_dict = _expand_env_vars(raw_dict)

    if overrides:
        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Invalid override format '{override}'. Expected 'key.path=value'")
            k, v = override.split("=", 1)
            _apply_override(raw_dict, k, v)

    config = PipelineConfig.model_validate(raw_dict)
    logger.info("Successfully loaded and validated config from: %s (Stage: %s)", path, config.stage)
    return config


def save_config(config: PipelineConfig, output_path: Union[str, Path]) -> None:
    """Serialize a validated PipelineConfig back into a YAML file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config.model_dump(exclude_none=True), f, default_flow_style=False, sort_keys=False)
    logger.info("Configuration saved to: %s", path)
