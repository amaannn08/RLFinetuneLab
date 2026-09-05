#!/usr/bin/env bash
# =====================================================================
# RLFinetuneLab Cloud GPU Environment Setup Script
# Configures CUDA dependencies, PyTorch, FlashAttention, and vLLM
# =====================================================================

set -e

echo "=== Initializing RLFinetuneLab Cloud Environment ==="

# Upgrade core pip/build tools
python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1+ support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core fine-tuning packages
pip install -r requirements.txt

# Install optional acceleration dependencies
echo "Installing bitsandbytes for 4-bit QLoRA..."
pip install bitsandbytes>=0.43.0

echo "Installing packaging and ninja for FlashAttention compilation..."
pip install ninja packaging
pip install flash-attn --no-build-isolation || echo "FlashAttention compilation skipped (SDPA fallback active)"

# Editable install
pip install -e .

echo "Setup complete! Verifying installation:"
rlfinetune hardware
