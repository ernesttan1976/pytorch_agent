# Quick Start Guide

This is a condensed version of the full verification guide. Use this for a fast setup.

## Prerequisites

- WSL2 with Ubuntu
- NVIDIA GPU with drivers installed
- `nvidia-smi` works

## Setup (5 minutes)

```bash
# 1. Setup environment
bash scripts/setup_wsl.sh
source venv/bin/activate

# 2. Configure accelerate
bash scripts/setup_accelerate.sh

# 3. Create .env file
cp .env.example .env
# Edit .env and add your HF_TOKEN (if using gated models)
```

## Test Pipeline (10 minutes)

```bash
# 1. Generate test data
python scripts/generate_synthetic_data.py --count 100

# 2. Prepare dataset
python scripts/prepare_dataset.py --input data/raw --output data/splits --seed 42

# 3. Sanity check
python scripts/sanity_check_dataset.py data/splits/train.jsonl

# 4. Smoke test training
python scripts/train_sft.py --config configs/sft_qlora.yaml --smoke

# 5. Evaluate
LATEST_RUN=$(ls -td runs/qlora_sft_* | head -1)
python scripts/evaluate.py --run $LATEST_RUN --config configs/eval_lm.yaml

# 6. Check results
cat $LATEST_RUN/eval_report.json | python -m json.tool
```

## If Smoke Test Fails

**Try LoRA instead:**
```bash
python scripts/train_sft.py --config configs/sft_lora.yaml --smoke
```

**Use smaller model:**
Edit `configs/sft_qlora.yaml` and change:
```yaml
model:
  base_model: "microsoft/phi-2"  # Smaller, faster for testing
```

## Full Training

Once smoke test passes:
```bash
python scripts/train_sft.py --config configs/sft_qlora.yaml
```

## Export Model

```bash
LATEST_RUN=$(ls -td runs/qlora_sft_* | head -1)
python scripts/export_merge_adapter.py --run $LATEST_RUN --merge
```

## Troubleshooting

- **OOM Error**: Use LoRA config or reduce `max_seq_length` in config
- **Model Download Fails**: Check `.env` has `HF_TOKEN` set
- **bitsandbytes Error**: Use LoRA config instead

For detailed troubleshooting, see `VERIFICATION_GUIDE.md`.

