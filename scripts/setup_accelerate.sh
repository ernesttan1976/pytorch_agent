#!/bin/bash
# Setup accelerate configuration for single GPU
set -e

echo "=========================================="
echo "Setting up Accelerate Configuration"
echo "=========================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if accelerate is installed
if ! python -c "import accelerate" 2>/dev/null; then
    echo "ERROR: accelerate not installed. Run setup_wsl.sh first."
    exit 1
fi

echo ""
echo "Configuring accelerate for single GPU..."
echo "This will create a default configuration for single GPU usage."
echo ""

# Create accelerate config directory
ACCELERATE_CONFIG_DIR="$HOME/.cache/huggingface/accelerate"
mkdir -p "$ACCELERATE_CONFIG_DIR"

# Generate default config for single GPU
cat > "$ACCELERATE_CONFIG_DIR/default_config.yaml" <<EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: NO
downcast_bf16: 'no'
gpu_ids: '0'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "✓ Accelerate configuration saved to: $ACCELERATE_CONFIG_DIR/default_config.yaml"
echo ""
echo "Configuration summary:"
echo "  - Single GPU (device 0)"
echo "  - Mixed precision: bf16"
echo "  - Distributed: NO"
echo ""
echo "You can view/edit the config with:"
echo "  accelerate env"
echo ""
echo "Or reconfigure interactively with:"
echo "  accelerate config"
echo ""

