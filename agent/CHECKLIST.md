# Validation Checklist

## Setup Stage
- [ ] `bash scripts/setup_wsl.sh` completes without errors
- [ ] `nvidia-smi` works in WSL2
- [ ] Python environment activated and dependencies installed
- [ ] `bash scripts/setup_accelerate.sh` generates config successfully
- [ ] CUDA sanity check passes (small tensor operation)

## Dataset Preparation
- [ ] `python scripts/prepare_dataset.py --input data/raw --output data/splits --seed 42` runs successfully
- [ ] Output files exist: `data/splits/{train,val,test}.jsonl`
- [ ] `python scripts/sanity_check_dataset.py data/splits/train.jsonl` reports no errors
- [ ] `data/interim/stats.json` generated with valid statistics

## Synthetic Data Testing
- [ ] `python scripts/generate_synthetic_data.py --count 100` creates test data
- [ ] Synthetic data can be processed through full pipeline

## Training (Smoke Test)
- [ ] `python scripts/train_sft.py --config configs/sft_qlora.yaml --smoke` completes in < 5 minutes
- [ ] Output directory created in `runs/`
- [ ] Checkpoint saved
- [ ] Training logs show progress

## Training (Full Run)
- [ ] `python scripts/train_sft.py --config configs/sft_qlora.yaml` runs without OOM
- [ ] Training loss decreases over time
- [ ] Evaluation metrics logged during training
- [ ] Final checkpoint saved

## Evaluation
- [ ] `python scripts/evaluate.py --run runs/<run_id>` generates report
- [ ] `runs/<run_id>/eval_report.json` exists with valid metrics
- [ ] Perplexity value is reasonable (< 100 for most models)
- [ ] `runs/<run_id>/samples.md` generated (if prompts enabled)

## Export
- [ ] `python scripts/export_merge_adapter.py --run runs/<run_id>` exports adapter
- [ ] `python scripts/export_merge_adapter.py --run runs/<run_id> --merge` merges model successfully
- [ ] Merged model loads and generates text

## Expected Outputs
Each training run should create:
- `runs/<run_name>_<timestamp>/checkpoint-*/` - Training checkpoints
- `runs/<run_name>_<timestamp>/adapter_model/` - LoRA adapter weights
- `runs/<run_name>_<timestamp>/training_args.json` - Training configuration
- `runs/<run_name>_<timestamp>/eval_report.json` - Evaluation results
- `runs/<run_name>_<timestamp>/samples.md` - Sample generations (if enabled)

