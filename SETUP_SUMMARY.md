# Setup Summary - PyTorch Fine-Tuning Pipeline

This document provides a high-level overview of what you need to do to get your PyTorch fine-tuning pipeline working.

## 📋 What You Have

A complete fine-tuning pipeline with:
- ✅ Environment setup scripts (WSL2)
- ✅ Dataset preparation and validation
- ✅ Training scripts (LoRA & QLoRA)
- ✅ Evaluation scripts
- ✅ Model export/merge scripts
- ✅ Configuration files
- ✅ Synthetic data generation for testing

## 🚀 Quick Start (3 Steps)

### Step 1: Initial Setup (One-time)
```bash
# Setup environment
bash scripts/setup_wsl.sh
source venv/bin/activate
bash scripts/setup_accelerate.sh

# Create .env file
cat > .env <<EOF
HF_TOKEN=your_token_here
CUDA_VISIBLE_DEVICES=0
EOF
```

### Step 2: Verify Setup
```bash
# Run verification script
python scripts/verify_setup.py
```

This checks:
- Python version
- PyTorch & CUDA availability
- Required dependencies
- Project files
- Environment variables

### Step 3: Test Pipeline
```bash
# Generate test data
python scripts/generate_synthetic_data.py --count 100

# Prepare dataset
python scripts/prepare_dataset.py --input data/raw --output data/splits --seed 42

# Sanity check
python scripts/sanity_check_dataset.py data/splits/train.jsonl

# Smoke test (quick validation)
python scripts/train_sft.py --config configs/sft_qlora.yaml --smoke
```

If smoke test passes → **You're ready!** 🎉

## 📚 Documentation

- **`QUICK_START.md`**: Condensed setup guide (5-10 minutes)
- **`VERIFICATION_GUIDE.md`**: Detailed step-by-step guide with troubleshooting
- **`README.md`**: Full documentation with all features
- **`agent/CHECKLIST.md`**: Validation checklist

## 🔍 Verification Checklist

Use this to verify everything works:

- [ ] `nvidia-smi` shows your GPU
- [ ] `bash scripts/setup_wsl.sh` completes successfully
- [ ] `python scripts/verify_setup.py` shows all checks passing
- [ ] `.env` file exists (with HF_TOKEN if using gated models)
- [ ] Synthetic data generation works
- [ ] Dataset preparation works
- [ ] Sanity check passes
- [ ] Smoke test training completes (< 5 minutes)
- [ ] Evaluation script runs
- [ ] Model export works

## 🎯 What Each Script Does

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `setup_wsl.sh` | Install dependencies, create venv | First time setup |
| `setup_accelerate.sh` | Configure accelerate for single GPU | After setup |
| `verify_setup.py` | Check if everything is configured | Anytime to verify |
| `generate_synthetic_data.py` | Create test dataset | Testing pipeline |
| `prepare_dataset.py` | Convert raw data to JSONL splits | With your own data |
| `sanity_check_dataset.py` | Validate dataset format | Before training |
| `train_sft.py` | Train model with LoRA/QLoRA | Main training |
| `evaluate.py` | Evaluate trained model | After training |
| `export_merge_adapter.py` | Export/merge adapter | After training |

## ⚙️ Configuration Files

- **`configs/sft_qlora.yaml`**: QLoRA training (4-bit, memory efficient)
- **`configs/sft_lora.yaml`**: LoRA training (full precision, faster)
- **`configs/eval_lm.yaml`**: Evaluation settings

## 🐛 Common Issues & Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| CUDA not available | Check `nvidia-smi`, restart WSL |
| Out of memory | Use QLoRA config, reduce `max_seq_length` |
| Model download fails | Check `.env` has `HF_TOKEN` |
| bitsandbytes error | Use LoRA config instead |
| Dataset format error | Run `sanity_check_dataset.py` |

## 📊 Expected Workflow

```
Setup → Generate Data → Prepare Dataset → Train → Evaluate → Export
  ↓         ↓              ↓              ↓         ↓         ↓
One-time  Testing      Validation    Training  Metrics   Deployment
```

## 🎓 Learning Path

1. **Start**: Run `verify_setup.py` to check your environment
2. **Test**: Use synthetic data to test the pipeline end-to-end
3. **Train**: Use your own data with the same pipeline
4. **Tune**: Adjust hyperparameters in config files
5. **Scale**: Increase dataset size and training steps

## 💡 Pro Tips

- Always run smoke test before full training
- Use QLoRA for memory efficiency (24GB GPU)
- Use LoRA for faster iteration
- Monitor GPU with `watch -n 1 nvidia-smi`
- Check logs in `runs/<run_id>/training.log`

## 📞 Need Help?

1. Check `VERIFICATION_GUIDE.md` for detailed troubleshooting
2. Review error messages in `runs/<run_id>/training.log`
3. Run `python scripts/<script>.py --help` for script options
4. Verify setup with `python scripts/verify_setup.py`

---

**Ready to start?** Run `python scripts/verify_setup.py` to check your setup!

