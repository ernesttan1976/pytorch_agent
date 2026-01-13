# Step-by-Step Verification Guide

This guide will help you verify that your PyTorch fine-tuning pipeline is working correctly from start to finish.

## Prerequisites Check

Before starting, verify your environment:

```bash
# 1. Check you're in WSL2 (if on Windows)
uname -a
# Should show "Microsoft" or "WSL" in the output

# 2. Check NVIDIA driver
nvidia-smi
# Should show your GPU (e.g., RTX 4090) with driver version

# 3. Check Python version
python3 --version
# Should be Python 3.8 or higher

# 4. Navigate to project root
cd /mnt/d/pytorch_agent  # or your actual path
pwd
# Should show your project directory
```

---

## Step 1: Environment Setup

### 1.1 Run WSL Setup Script

```bash
bash scripts/setup_wsl.sh
```

**What to expect:**
- ✓ NVIDIA driver detected
- ✓ Python version displayed
- ✓ uv package manager installed (or pip fallback)
- ✓ Virtual environment created in `venv/`
- ✓ Dependencies installed (PyTorch, transformers, peft, trl, bitsandbytes, etc.)
- ✓ CUDA sanity check passes with tensor operations

**If errors occur:**
- **nvidia-smi not found**: Install NVIDIA drivers for WSL2
- **CUDA not available**: Check driver installation, restart WSL if needed
- **bitsandbytes fails**: This is okay for now - you can use LoRA instead of QLoRA

### 1.2 Activate Virtual Environment

```bash
source venv/bin/activate
```

**Verify activation:**
```bash
which python
# Should show: .../pytorch_agent/venv/bin/python

python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
# Should show PyTorch version and CUDA: True
```

### 1.3 Setup Accelerate Configuration

```bash
bash scripts/setup_accelerate.sh
```

**What to expect:**
- ✓ Accelerate configuration created
- ✓ Single GPU configuration saved

**Verify:**
```bash
accelerate env
# Should show your single GPU configuration
```

### 1.4 Create Environment File

```bash
# Create .env file from template
cat > .env <<EOF
HF_TOKEN=your_huggingface_token_here
CUDA_VISIBLE_DEVICES=0
EOF
```

**Important:** Replace `your_huggingface_token_here` with your actual Hugging Face token:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is enough)
3. Copy and paste it into `.env`

**For testing without gated models:** You can leave `HF_TOKEN=` empty if using public models.

---

## Step 2: Dataset Preparation

### 2.1 Generate Synthetic Test Data

```bash
python scripts/generate_synthetic_data.py --count 100
```

**What to expect:**
- Creates `data/raw/synthetic/train.jsonl` with 100 examples
- Each example has `id`, `messages`, and `meta` fields

**Verify:**
```bash
ls -lh data/raw/synthetic/
# Should show train.jsonl

head -n 1 data/raw/synthetic/train.jsonl | python -m json.tool
# Should show a valid JSON object with id, messages, meta
```

### 2.2 Prepare Dataset Splits

```bash
python scripts/prepare_dataset.py \
    --input data/raw \
    --output data/splits \
    --seed 42 \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

**What to expect:**
- Creates `data/splits/train.jsonl` (~80 examples)
- Creates `data/splits/val.jsonl` (~10 examples)
- Creates `data/splits/test.jsonl` (~10 examples)
- Creates `data/interim/stats.json` with statistics

**Verify:**
```bash
ls -lh data/splits/
# Should show train.jsonl, val.jsonl, test.jsonl

wc -l data/splits/*.jsonl
# Should show line counts for each file

cat data/interim/stats.json | python -m json.tool
# Should show statistics about the dataset
```

### 2.3 Sanity Check Dataset

```bash
python scripts/sanity_check_dataset.py data/splits/train.jsonl
```

**What to expect:**
- ✓ All examples have required fields (id, messages, meta)
- ✓ Messages are properly formatted
- ✓ Last message is from assistant
- ✓ No validation errors

**If errors occur:**
- Check the error messages - they'll tell you which examples are invalid
- Fix the dataset format and re-run preparation

---

## Step 3: Configuration Check

### 3.1 Verify Training Config

```bash
cat configs/sft_qlora.yaml
```

**Key settings to verify:**
- `model.base_model`: Should be a valid Hugging Face model
- `data.train_path`: Should point to `data/splits/train.jsonl`
- `data.eval_path`: Should point to `data/splits/val.jsonl`
- `train.method`: Should be `"qlora"` for QLoRA or `"lora"` for LoRA

**For testing, you might want to use a smaller model:**
- Change `base_model` to `"microsoft/phi-2"` (smaller, faster for testing)
- Or keep `"meta-llama/Meta-Llama-3-8B-Instruct"` if you have HF token

### 3.2 Test Model Download (Optional)

If using a gated model, test download:

```bash
python scripts/download_model.py --model meta-llama/Meta-Llama-3-8B-Instruct
```

**What to expect:**
- Model downloads to Hugging Face cache
- No errors about authentication

---

## Step 4: Smoke Test Training

**This is a critical step - it validates everything works with minimal resources.**

### 4.1 Run Smoke Test

```bash
python scripts/train_sft.py --config configs/sft_qlora.yaml --smoke
```

**What to expect:**
- Training starts with 50 examples
- Runs for 50 steps (should complete in 2-5 minutes)
- Creates a run directory in `runs/qlora_sft_<timestamp>/`
- Saves checkpoint and adapter model
- Training loss decreases over steps

**Monitor progress:**
```bash
# In another terminal, watch GPU usage
watch -n 1 nvidia-smi

# Or watch the log file
tail -f runs/qlora_sft_*/training.log
```

**If errors occur:**

**Out of Memory (OOM):**
```bash
# Try LoRA instead (uses more VRAM but sometimes more stable)
python scripts/train_sft.py --config configs/sft_lora.yaml --smoke
```

**Model download fails:**
- Check `.env` has correct `HF_TOKEN`
- For gated models, accept terms on Hugging Face website

**bitsandbytes errors:**
- Use LoRA config instead: `--config configs/sft_lora.yaml`

### 4.2 Verify Smoke Test Output

```bash
# Check run directory exists
ls -lh runs/qlora_sft_*/

# Should contain:
# - checkpoint-*/ (training checkpoint)
# - adapter_model/ (LoRA adapter weights)
# - config.json (training config)
# - training.log (training logs)
# - training_args.json (training arguments)

# Check training log
tail -20 runs/qlora_sft_*/training.log
# Should show training completed successfully

# Check checkpoint
ls -lh runs/qlora_sft_*/checkpoint-*/
# Should show model files
```

---

## Step 5: Full Training (Optional - Only if smoke test passes)

**Only proceed if smoke test completed successfully!**

### 5.1 Run Full Training

```bash
python scripts/train_sft.py --config configs/sft_qlora.yaml
```

**What to expect:**
- Training on full dataset
- Runs for configured number of epochs/steps
- Saves checkpoints periodically
- Evaluation runs on validation set
- Final model saved

**Monitor:**
- Watch `nvidia-smi` for GPU memory usage
- Check `training.log` for progress
- Training loss should decrease over time

**Expected duration:**
- Small dataset (100 examples): 10-30 minutes
- Medium dataset (1000 examples): 1-3 hours
- Large dataset: Several hours

### 5.2 Verify Training Output

```bash
# Get the latest run directory
LATEST_RUN=$(ls -td runs/qlora_sft_* | head -1)
echo "Latest run: $LATEST_RUN"

# Check contents
ls -lh $LATEST_RUN/

# Check final checkpoint
ls -lh $LATEST_RUN/checkpoint-*/

# Check adapter model
ls -lh $LATEST_RUN/adapter_model/
# Should contain adapter_config.json and adapter_model.safetensors
```

---

## Step 6: Evaluation

### 6.1 Run Evaluation

```bash
# Use the run directory from training
LATEST_RUN=$(ls -td runs/qlora_sft_* | head -1)
python scripts/evaluate.py --run $LATEST_RUN --config configs/eval_lm.yaml
```

**What to expect:**
- Loads trained adapter model
- Evaluates on validation/test set
- Calculates loss and perplexity
- Generates sample outputs (if prompts configured)
- Creates `eval_report.json` and `samples.md`

**Verify:**
```bash
# Check evaluation report
cat $LATEST_RUN/eval_report.json | python -m json.tool

# Should show:
# - eval_loss: A float value (lower is better)
# - perplexity: A float value (lower is better, typically < 100)
# - num_samples: Number of examples evaluated
# - total_tokens: Total tokens processed

# Check samples (if generated)
cat $LATEST_RUN/samples.md
# Should show sample model outputs
```

---

## Step 7: Model Export

### 7.1 Export Adapter Only

```bash
LATEST_RUN=$(ls -td runs/qlora_sft_* | head -1)
python scripts/export_merge_adapter.py --run $LATEST_RUN
```

**What to expect:**
- Adapter weights already saved (this just verifies)
- Confirmation message

### 7.2 Merge Adapter with Base Model (Optional)

```bash
python scripts/export_merge_adapter.py --run $LATEST_RUN --merge
```

**What to expect:**
- Loads base model and adapter
- Merges weights
- Saves merged model to `$LATEST_RUN/merged_model/`
- Takes several minutes (model size dependent)

**Verify:**
```bash
ls -lh $LATEST_RUN/merged_model/
# Should contain full model files (config.json, model files, tokenizer files)

# Test loading merged model
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('$LATEST_RUN/merged_model')
model = AutoModelForCausalLM.from_pretrained('$LATEST_RUN/merged_model')
print('✓ Merged model loads successfully')
"
```

---

## Step 8: End-to-End Test

Run the complete pipeline from scratch:

```bash
# Clean previous runs (optional)
# rm -rf runs/* data/splits/* data/interim/*

# 1. Generate data
python scripts/generate_synthetic_data.py --count 100

# 2. Prepare dataset
python scripts/prepare_dataset.py --input data/raw --output data/splits --seed 42

# 3. Sanity check
python scripts/sanity_check_dataset.py data/splits/train.jsonl

# 4. Smoke test
python scripts/train_sft.py --config configs/sft_qlora.yaml --smoke

# 5. Evaluate
LATEST_RUN=$(ls -td runs/qlora_sft_* | head -1)
python scripts/evaluate.py --run $LATEST_RUN --config configs/eval_lm.yaml

# 6. Export
python scripts/export_merge_adapter.py --run $LATEST_RUN
```

**If all steps complete without errors, your pipeline is working! 🎉**

---

## Troubleshooting Common Issues

### Issue: CUDA Out of Memory

**Solutions:**
1. Use QLoRA instead of LoRA (already using it)
2. Reduce `max_seq_length` in config (e.g., 2048 instead of 4096)
3. Reduce batch size: `per_device_train_batch_size: 1` (already set)
4. Increase gradient accumulation: `gradient_accumulation_steps: 32`
5. Enable gradient checkpointing: `gradient_checkpointing: true` (already set)

### Issue: Model Download Fails

**Solutions:**
1. Check `.env` has `HF_TOKEN` set
2. For gated models, accept terms on Hugging Face
3. Test token: `python -c "from huggingface_hub import login; login()"`
4. Use a public model for testing: `microsoft/phi-2`

### Issue: bitsandbytes Installation Fails

**Solutions:**
1. Use LoRA instead: `--config configs/sft_lora.yaml`
2. Reinstall: `pip uninstall bitsandbytes && pip install bitsandbytes --no-cache-dir`
3. Check CUDA compatibility

### Issue: Dataset Format Errors

**Solutions:**
1. Run sanity check: `python scripts/sanity_check_dataset.py data/splits/train.jsonl`
2. Check JSONL is valid: `python -m json.tool < data/splits/train.jsonl | head`
3. Ensure last message is from assistant
4. Check all required fields present

### Issue: Training Loss Not Decreasing

**Possible causes:**
1. Learning rate too high/low - adjust in config
2. Dataset too small - generate more synthetic data
3. Model too large for dataset - use smaller model or more data
4. Check training log for warnings

---

## Quick Verification Checklist

Use this checklist to quickly verify everything works:

- [ ] `nvidia-smi` shows GPU
- [ ] `bash scripts/setup_wsl.sh` completes successfully
- [ ] Virtual environment activates: `source venv/bin/activate`
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` prints `True`
- [ ] `bash scripts/setup_accelerate.sh` completes successfully
- [ ] `.env` file exists with `HF_TOKEN` (if needed)
- [ ] `python scripts/generate_synthetic_data.py --count 10` creates data
- [ ] `python scripts/prepare_dataset.py --input data/raw --output data/splits` creates splits
- [ ] `python scripts/sanity_check_dataset.py data/splits/train.jsonl` shows no errors
- [ ] `python scripts/train_sft.py --config configs/sft_qlora.yaml --smoke` completes in < 5 minutes
- [ ] Run directory created in `runs/` with checkpoint and adapter
- [ ] `python scripts/evaluate.py --run <run_dir> --config configs/eval_lm.yaml` generates report
- [ ] `python scripts/export_merge_adapter.py --run <run_dir>` exports successfully

---

## Next Steps

Once everything is verified:

1. **Use your own data**: Replace synthetic data with your dataset
2. **Tune hyperparameters**: Adjust learning rate, batch size, LoRA rank in configs
3. **Scale up**: Increase dataset size and training steps
4. **Experiment**: Try different models, LoRA configurations, training strategies

---

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review the relevant script's help: `python scripts/<script>.py --help`
3. Check logs in `runs/<run_id>/training.log`
4. Verify all prerequisites are met
5. Try the troubleshooting section above

Good luck with your fine-tuning! 🚀

