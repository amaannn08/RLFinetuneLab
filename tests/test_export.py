import os
from rlfinetunelab.export.convert_gguf import generate_gguf_conversion_command


def test_gguf_command_generation():
    cmds = generate_gguf_conversion_command(
        model_dir="outputs/test_model",
        output_gguf_path="outputs/model.gguf",
        quantization_type="Q4_K_M"
    )
    assert any("convert_hf_to_gguf.py" in c for c in cmds)
    assert any("llama-quantize" in c for c in cmds)
    assert any("Q4_K_M" in c for c in cmds)
