#!/bin/bash
# WSL2 environment setup script for LLM fine-tuning
set -e

echo "=========================================="
echo "WSL2 Setup for LLM Fine-Tuning Pipeline"
echo "=========================================="

# Check if running in WSL
if ! grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null ; then
    echo "Warning: This script is designed for WSL2. Continuing anyway..."
fi

# Check NVIDIA driver
echo ""
echo "Checking NVIDIA driver..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers for WSL2."
    echo "See: https://docs.nvidia.com/cuda/wsl-user-guide/index.html"
    exit 1
fi

nvidia-smi
echo "✓ NVIDIA driver detected"

# Check Python
echo ""
echo "Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "Installing Python 3..."
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip python3-venv
fi

PYTHON_VERSION=$(python3 --version)
echo "✓ $PYTHON_VERSION"

# Install uv if not present
echo ""
echo "Checking uv package manager..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    if ! command -v uv &> /dev/null; then
        echo "ERROR: Failed to install uv. Trying alternative method..."
        pip3 install uv
    fi
fi

if command -v uv &> /dev/null; then
    echo "✓ uv installed: $(uv --version)"
else
    echo "WARNING: uv installation failed. Falling back to pip."
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping..."
else
    if command -v uv &> /dev/null; then
        uv venv
        source venv/bin/activate
        uv pip install --upgrade pip
    else
        python3 -m venv venv
        source venv/bin/activate
        pip install --upgrade pip
    fi
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing Python dependencies..."
echo "This may take several minutes..."

# Base dependencies
PACKAGES="torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
PACKAGES="$PACKAGES transformers datasets accelerate peft trl"
PACKAGES="$PACKAGES sentencepiece evaluate safetensors"
PACKAGES="$PACKAGES pyyaml pydantic"

if command -v uv &> /dev/null; then
    uv pip install $PACKAGES
    uv pip install bitsandbytes
else
    pip install $PACKAGES
    pip install bitsandbytes
fi

echo "✓ Dependencies installed"

# Install additional utilities
echo ""
echo "Installing additional utilities..."
if command -v uv &> /dev/null; then
    uv pip install rouge-score 2>/dev/null || echo "Warning: rouge-score installation failed (optional)"
else
    pip install rouge-score 2>/dev/null || echo "Warning: rouge-score installation failed (optional)"
fi

# CUDA sanity check
echo ""
echo "Running CUDA sanity check..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    
    # Test tensor operation
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = torch.matmul(x, y)
    print('✓ CUDA tensor operations working')
else:
    print('ERROR: CUDA not available. Check your NVIDIA driver installation.')
    exit(1)
"

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run accelerate setup: bash scripts/setup_accelerate.sh"
echo "3. Set up your Hugging Face token in .env file"
echo "4. Generate synthetic data: python scripts/generate_synthetic_data.py --count 100"
echo "5. Prepare dataset: python scripts/prepare_dataset.py --input data/raw --output data/splits"
echo ""

