#!/bin/bash

# Environment Setup Script for VLM Quantization Research
# Run this script first to set up the complete environment

set -e  # Exit on error

echo "======================================================================"
echo "VLM Quantization Research - Environment Setup"
echo "======================================================================"

# Check CUDA availability
echo -e "\n[1/6] Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo "✓ CUDA is available"
else
    echo "⚠ Warning: nvidia-smi not found. CUDA may not be available."
fi

# Upgrade pip
echo -e "\n[2/6] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo -e "\n[3/6] Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo -e "\n[4/6] Installing core dependencies..."
pip install transformers accelerate sentencepiece protobuf

# Install quantization libraries
echo -e "\n[5/6] Installing quantization libraries..."

# GPTQModel
echo "  - Installing GPTQModel..."
pip install gptqmodel>=1.4.1

# AutoAWQ
echo "  - Installing AutoAWQ..."
pip install autoawq>=0.2.5

# BitsAndBytes
echo "  - Installing bitsandbytes..."
pip install bitsandbytes>=0.44.0

# TorchAO (from source for latest features)
echo "  - Installing TorchAO from source..."
pip install git+https://github.com/pytorch/ao.git

# Install VLM-specific libraries
echo "  - Installing Qwen VL utilities..."
pip install qwen-vl-utils

# VLMEvalKit
echo "  - Installing VLMEvalKit..."
pip install git+https://github.com/open-compass/VLMEvalKit.git
pip install vlmeval

# Install utility libraries
echo -e "\n[6/6] Installing utility libraries..."
pip install \
    wandb \
    tensorboard \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    Pillow \
    tqdm \
    scikit-learn \
    datasets \
    jupyter \
    ipywidgets \
    pyyaml

echo -e "\n======================================================================"
echo "Environment setup complete!"
echo "======================================================================"

# Verify installations
echo -e "\nVerifying installations..."

python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'✓ Transformers: {transformers.__version__}')"
python -c "import gptqmodel; print(f'✓ GPTQModel installed')"
python -c "import awq; print(f'✓ AutoAWQ installed')"
python -c "import bitsandbytes; print(f'✓ BitsAndBytes installed')"
python -c "import torchao; print(f'✓ TorchAO installed')"

echo -e "\n======================================================================"
echo "Ready to run experiments!"
echo "======================================================================"
