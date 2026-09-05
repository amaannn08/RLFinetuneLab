"""
llama.cpp GGUF conversion and quantization automation wrapper.
Enables local high-speed CPU/Metal inference (e.g. via Ollama or llama.cpp).
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, List
from rlfinetunelab.utils.logging import get_logger

logger = get_logger(__name__)


def generate_gguf_conversion_command(
    model_dir: str,
    output_gguf_path: str,
    quantization_type: str = "Q4_K_M",
    llama_cpp_dir: Optional[str] = None
) -> List[str]:
    """Generates the bash commands needed to convert a merged model directory to GGUF."""
    model_p = Path(model_dir).resolve()
    out_p = Path(output_gguf_path).resolve()
    llama_dir = Path(llama_cpp_dir).resolve() if llama_cpp_dir else Path("./llama.cpp")

    commands = [
        f"# 1. Convert HuggingFace to FP16 GGUF",
        f"python3 {llama_dir}/convert_hf_to_gguf.py {model_p} --outfile {out_p.with_suffix('.fp16.gguf')}",
        f"# 2. Quantize to {quantization_type}",
        f"{llama_dir}/build/bin/llama-quantize {out_p.with_suffix('.fp16.gguf')} {out_p} {quantization_type}",
    ]
    return commands


def convert_to_gguf(
    model_dir: str,
    output_gguf_path: str,
    quantization_type: str = "Q4_K_M",
    llama_cpp_dir: Optional[str] = None
) -> None:
    """Executes GGUF conversion if llama.cpp toolchain is locally available."""
    cmds = generate_gguf_conversion_command(model_dir, output_gguf_path, quantization_type, llama_cpp_dir)
    logger.info("GGUF Conversion Plan:\n" + "\n".join(cmds))

    # Check if llama.cpp exists
    llama_p = Path(llama_cpp_dir or "./llama.cpp")
    convert_script = llama_p / "convert_hf_to_gguf.py"

    if not convert_script.is_file():
        logger.warning(
            "llama.cpp convert script not found at '%s'. "
            "Please clone llama.cpp or run 'bash scripts/export_gguf.sh %s %s %s'",
            convert_script, model_dir, output_gguf_path, quantization_type
        )
        return

    logger.info("Executing GGUF conversion...")
    intermediate_fp16 = Path(output_gguf_path).with_suffix(".fp16.gguf")
    subprocess.run(
        [sys.executable, str(convert_script), model_dir, "--outfile", str(intermediate_fp16)],
        check=True
    )
    quantize_bin = llama_p / "build/bin/llama-quantize"
    if quantize_bin.is_file():
        subprocess.run(
            [str(quantize_bin), str(intermediate_fp16), output_gguf_path, quantization_type],
            check=True
        )
        intermediate_fp16.unlink(missing_ok=True)
        logger.info("Successfully created quantized GGUF: %s", output_gguf_path)
