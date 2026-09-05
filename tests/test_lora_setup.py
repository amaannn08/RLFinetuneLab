from rlfinetunelab.config.schema import LoraConfigSchema
from rlfinetunelab.models.lora import build_peft_config, get_trainable_parameters_summary
from rlfinetunelab.utils.hardware import get_hardware_summary


def test_peft_config_builder():
    schema = LoraConfigSchema(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])
    peft_cfg = build_peft_config(schema)
    assert peft_cfg.r == 8
    assert peft_cfg.lora_alpha == 16
    assert peft_cfg.target_modules == {"q_proj", "v_proj"}


def test_trainable_params_summary(tiny_model):
    from peft import get_peft_model
    schema = LoraConfigSchema(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
    peft_cfg = build_peft_config(schema)
    peft_model = get_peft_model(tiny_model, peft_cfg)

    summary = get_trainable_parameters_summary(peft_model)
    assert summary["trainable_params"] > 0
    assert summary["trainable_params"] < summary["all_params"]
    assert 0.0 < summary["trainable_percentage"] < 100.0


def test_hardware_summary_structure():
    hw = get_hardware_summary()
    assert "cuda_available" in hw
    assert "device_count" in hw
    assert "recommended_precision" in hw
