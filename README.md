# Small LLM Fine-Tuning Pipeline

A production-ready, single-GPU fine-tuning pipeline for instruction-tuned LLMs using LoRA and QLoRA. Optimized for RTX 4090 (24GB) on Windows via WSL2.

## Features

- **LoRA & QLoRA Support**: Train with parameter-efficient fine-tuning (4-bit quantized or full precision)
- **Multi-Format Dataset Support**: Auto-detect and convert Alpaca, ShareGPT, CSV, and JSONL formats
- **Synthetic Data Generation**: Test the pipeline immediately with generated datasets
- **Comprehensive Evaluation**: Loss, perplexity, and prompt-based evaluation
- **Model Export**: Export adapters and merge with base models
- **WSL2 Optimized**: Reliable setup for Windows + WSL2 + CUDA

## Quick Start

### Prerequisites

- Windows 10/11 with WSL2 (Ubuntu 22.04/24.04 recommended)
- NVIDIA RTX 4090 or compatible GPU (24GB VRAM recommended)
- NVIDIA drivers installed for WSL2
- ~50GB free disk space (for models and datasets)

### Golden Path Commands

From repo root in WSL2:

```bash
# 1. Setup environment
bash scripts/setup_wsl.sh
source venv/bin/activate
bash scripts/setup_accelerate.sh

# 2. Configure environment variables
cp .env.example .env
# Edit .env and add your HF_TOKEN for gated models

# 3. Generate synthetic data (for testing)
python scripts/generate_synthetic_data.py --count 100

# 4. Prepare dataset
python scripts/prepare_dataset.py --input data/raw --output data/splits --seed 42

# 5. Sanity check dataset
python scripts/sanity_check_dataset.py data/splits/train.jsonl

# 6. Smoke test (quick validation)
python scripts/train_sft.py --config configs/sft_qlora.yaml --smoke

# 7. Full training
python scripts/train_sft.py --config configs/sft_qlora.yaml

# 8. Evaluate
python scripts/evaluate.py --run runs/qlora_sft_<timestamp> --config configs/eval_lm.yaml

# 9. Export/merge model
python scripts/export_merge_adapter.py --run runs/qlora_sft_<timestamp> --merge
```

## Repository Structure

```
small-llm-ft/
├── README.md                    # This file
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore patterns
├── .cursorrules                 # Cursor agent rules
├── agent/
│   ├── CURSOR_AGENT.md          # Agent instructions
│   └── CHECKLIST.md             # Validation checklist
├── configs/
│   ├── sft_qlora.yaml           # QLoRA config (4090-optimized)
│   ├── sft_lora.yaml            # LoRA config (faster)
│   └── eval_lm.yaml             # Evaluation config
├── data/
│   ├── raw/                     # Your raw data files
│   ├── interim/                 # Normalized intermediate files
│   ├── processed/               # Tokenized/cached data
│   └── splits/                  # Train/val/test splits (JSONL)
├── scripts/
│   ├── setup_wsl.sh             # WSL2 environment setup
│   ├── setup_accelerate.sh      # Accelerate config
│   ├── download_model.py        # Model downloader
│   ├── prepare_dataset.py       # Dataset converter
│   ├── generate_synthetic_data.py # Synthetic data generator
│   ├── sanity_check_dataset.py  # Dataset validator
│   ├── train_sft.py             # Main training script
│   ├── evaluate.py              # Evaluation script
│   └── export_merge_adapter.py  # Model export/merge
├── src/
│   ├── dataset_schema.py        # JSONL schema definitions
│   ├── data_loaders.py          # Multi-format loaders
│   ├── formatting.py            # Message formatting
│   ├── metrics.py               # Evaluation metrics
│   └── utils.py                 # Common utilities
└── runs/                        # Training outputs (gitignored)
```

## Detailed Usage

### 1. Environment Setup

#### WSL2 Setup

```bash
bash scripts/setup_wsl.sh
```

This script:
- Checks NVIDIA driver (nvidia-smi)
- Installs uv package manager
- Creates Python virtual environment
- Installs all dependencies (PyTorch, transformers, peft, trl, bitsandbytes, etc.)
- Runs CUDA sanity check

#### Accelerate Configuration

```bash
bash scripts/setup_accelerate.sh
```

Configures accelerate for single GPU training.

### 2. Dataset Preparation

#### Supported Formats

The pipeline auto-detects and supports:

- **Alpaca JSON**: `{"instruction": "...", "input": "...", "output": "..."}`
- **ShareGPT JSONL**: `{"conversations": [{"from": "human", "value": "..."}]}`
- **CSV**: Columns for instruction and response
- **JSONL Messages**: Target format with `{"messages": [{"role": "...", "content": "..."}]}`

#### Prepare Your Data

```bash
# From raw files
python scripts/prepare_dataset.py \
    --input data/raw \
    --output data/splits \
    --seed 42 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

#### Validate Dataset

```bash
python scripts/sanity_check_dataset.py data/splits/train.jsonl
```

#### Generate Synthetic Data

For quick testing:

```bash
python scripts/generate_synthetic_data.py --count 100
python scripts/prepare_dataset.py --input data/raw/synthetic --output data/splits
```

### 3. Training

#### QLoRA Training (Recommended for 4090)

```bash
python scripts/train_sft.py --config configs/sft_qlora.yaml
```

Features:
- 4-bit quantization (NF4)
- Batch size 1, gradient accumulation 16
- Max sequence length 4096
- Paged AdamW 8-bit optimizer

#### LoRA Training (Faster, Less VRAM Efficient)

```bash
python scripts/train_sft.py --config configs/sft_lora.yaml
```

Features:
- Full precision (BF16)
- Batch size 1, gradient accumulation 8
- Max sequence length 2048

#### Smoke Test

Quick validation run (50 steps, small subset):

```bash
python scripts/train_sft.py --config configs/sft_qlora.yaml --smoke
```

#### Training Output

Training saves to `runs/<run_name>_<timestamp>/`:
- `checkpoint-*/` - Training checkpoints
- `adapter_model/` - LoRA adapter weights
- `config.json` - Training configuration
- `training.log` - Training logs

### 4. Evaluation

```bash
python scripts/evaluate.py \
    --run runs/qlora_sft_20240101_120000 \
    --config configs/eval_lm.yaml
```

Generates:
- `eval_report.json` - Loss, perplexity, metrics
- `samples.md` - Sample generations (if prompts enabled)

### 5. Model Export

#### Export Adapter Only

```bash
python scripts/export_merge_adapter.py --run runs/qlora_sft_<timestamp>
```

#### Merge Adapter with Base Model

```bash
python scripts/export_merge_adapter.py \
    --run runs/qlora_sft_<timestamp> \
    --merge
```

Merged model saved to `runs/<run_id>/merged_model/`

## Configuration

### Training Configs

Edit YAML files in `configs/`:

- **sft_qlora.yaml**: QLoRA settings optimized for 24GB VRAM
- **sft_lora.yaml**: LoRA settings for faster iteration
- **eval_lm.yaml**: Evaluation parameters

Key parameters:
- `model.base_model`: Hugging Face model name
- `data.max_seq_length`: Maximum sequence length
- `train.per_device_train_batch_size`: Batch size per device
- `train.gradient_accumulation_steps`: Gradient accumulation
- `lora.r`: LoRA rank
- `lora.alpha`: LoRA alpha

### Environment Variables

Create `.env` from `.env.example`:

```bash
HF_TOKEN=your_huggingface_token_here
CUDA_VISIBLE_DEVICES=0
```

## Troubleshooting

### Out of Memory (OOM)

If you get OOM errors:

1. **Reduce batch size**: Set `per_device_train_batch_size: 1` and increase `gradient_accumulation_steps`
2. **Reduce sequence length**: Lower `max_seq_length` (e.g., 2048 instead of 4096)
3. **Use QLoRA**: Switch from LoRA to QLoRA config
4. **Enable gradient checkpointing**: Set `gradient_checkpointing: true`
5. **Close other GPU processes**: Check with `nvidia-smi`

### CUDA Issues in WSL2

```bash
# Verify NVIDIA driver
nvidia-smi

# Check CUDA in Python
python -c "import torch; print(torch.cuda.is_available())"

# If not working, reinstall CUDA drivers for WSL2
# See: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
```

### Model Download Issues

For gated models (e.g., Llama):

1. Get token from https://huggingface.co/settings/tokens
2. Add to `.env`: `HF_TOKEN=your_token`
3. Run: `python scripts/download_model.py --model meta-llama/Meta-Llama-3-8B-Instruct`

### bitsandbytes Installation Issues

If bitsandbytes fails on WSL2:

```bash
# Try installing from source
pip uninstall bitsandbytes
pip install bitsandbytes --no-cache-dir

# Or use LoRA instead (no bitsandbytes needed)
python scripts/train_sft.py --config configs/sft_lora.yaml
```

### Dataset Format Errors

If dataset preparation fails:

1. Check format with `sanity_check_dataset.py`
2. Ensure JSONL is valid (one JSON object per line)
3. Verify required fields: `id`, `messages`, `meta`
4. Check that last message is from assistant

## Dataset Format

Target format (JSONL):

```json
{
  "id": "unique_id",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."}
  ],
  "meta": {
    "source": "alpaca",
    "tags": ["education"],
    "split": "train"
  }
}
```

Rules:
- `id`: Unique identifier (string)
- `messages`: Array of message objects with `role` and `content`
- Roles: `system`, `user`, `assistant`
- Last message must be from `assistant`
- `system` message can only be first message
- Content cannot be empty

## Expected Outputs

### Training Run Directory

```
runs/qlora_sft_20240101_120000/
├── checkpoint-200/
│   └── ...
├── adapter_model/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── config.json
├── training.log
├── training_args.json
├── eval_report.json
└── samples.md
```

### Evaluation Report

```json
{
  "eval_loss": 0.123,
  "perplexity": 1.131,
  "num_samples": 1000,
  "total_tokens": 50000,
  "prompt_results": [...]
}
```

## Performance Tips

### For RTX 4090 (24GB)

- **QLoRA**: Use for longer contexts (4096 tokens), larger batches, or when close to VRAM limit
- **LoRA**: Use for faster iteration, shorter contexts (2048 tokens)
- **Gradient Checkpointing**: Always enable for memory efficiency
- **BF16**: Prefer over FP16 for better stability

### Batch Size Tuning

Start with config defaults, then adjust:
- Increase `gradient_accumulation_steps` to simulate larger batch
- Reduce `per_device_train_batch_size` if OOM
- Monitor VRAM usage: `watch -n 1 nvidia-smi`

## License

This pipeline is provided as-is for educational and research purposes.

## Contributing

See `agent/CURSOR_AGENT.md` for development guidelines and agent instructions.

## References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [WSL2 CUDA Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

