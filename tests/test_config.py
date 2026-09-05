import pytest
from pydantic import ValidationError
from rlfinetunelab.config import load_config, save_config
from rlfinetunelab.config.schema import PipelineConfig, QuantizationConfig, ModelConfig


def test_load_base_config():
    cfg = load_config("configs/base_config.yaml")
    assert cfg.project_name == "RLFinetuneLab"
    assert cfg.stage == "sft"
    assert cfg.model.name_or_path == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.lora.r == 16


def test_config_overrides():
    cfg = load_config(
        "configs/base_config.yaml",
        overrides=["sft.learning_rate=5e-5", "lora.r=32", "quantization.load_in_4bit=false"]
    )
    assert cfg.sft.learning_rate == 5e-5
    assert cfg.lora.r == 32
    assert cfg.quantization.load_in_4bit is False


def test_quantization_mutual_exclusion():
    with pytest.raises(ValidationError):
        QuantizationConfig(load_in_4bit=True, load_in_8bit=True)


def test_save_and_reload_config(temp_dir):
    cfg = load_config("configs/base_config.yaml")
    save_path = f"{temp_dir}/saved_config.yaml"
    save_config(cfg, save_path)
    reloaded = load_config(save_path)
    assert reloaded.project_name == cfg.project_name
    assert reloaded.stage == cfg.stage
