#!/usr/bin/env bash
# =====================================================================
# RLFinetuneLab GGUF Export Script
# Automates llama.cpp setup and quantized GGUF export for local inference
# =====================================================================

set -e

MODEL_DIR=${1:-"outputs/merged_model"}
OUTPUT_FILE=${2:-"outputs/model_q4_k_m.gguf"}
QUANT_TYPE=${3:-"Q4_K_M"}

echo "=== RLFinetuneLab: GGUF Export Pipeline ==="
echo "Source Model:      ${MODEL_DIR}"
echo "Output GGUF:       ${OUTPUT_FILE}"
echo "Quantization Type: ${QUANT_TYPE}"

if [ ! -d "${MODEL_DIR}" ]; then
  echo "Error: Model directory '${MODEL_DIR}' does not exist."
  echo "Run 'rlfinetune export --adapter <path>' first."
  exit 1
fi

# Clone llama.cpp if not present
if [ ! -d "llama.cpp" ]; then
  echo "Cloning llama.cpp repository..."
  git clone https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp
echo "Installing llama.cpp requirements..."
pip install -r requirements.txt -q

if [ ! -f "build/bin/llama-quantize" ]; then
  echo "Building llama.cpp binaries with cmake..."
  cmake -B build
  cmake --build build --config Release -j $(nproc)
fi
cd ..

FP16_GGUF="${OUTPUT_FILE%.*}.fp16.gguf"

echo "Step 1: Converting HuggingFace model to FP16 GGUF..."
python3 llama.cpp/convert_hf_to_gguf.py "${MODEL_DIR}" --outfile "${FP16_GGUF}"

echo "Step 2: Quantizing to ${QUANT_TYPE}..."
llama.cpp/build/bin/llama-quantize "${FP16_GGUF}" "${OUTPUT_FILE}" "${QUANT_TYPE}"

rm -f "${FP16_GGUF}"
echo "Export complete: ${OUTPUT_FILE}"
