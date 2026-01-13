# Cursor Agent: Single-GPU Small LLM Fine-Tuning (RTX 4090, Windows via WSL2)

## Mission
Set up a reproducible, single-GPU fine-tuning pipeline (LoRA + QLoRA) for an instruction-tuned LLM:
- Environment setup (WSL2 Ubuntu recommended)
- Dataset preparation into a strict JSONL schema
- Train (SFT) with LoRA/QLoRA
- Evaluate (held-out loss/perplexity + task metrics where applicable)
- Export adapter + optional merged model
- Document everything so it's repeatable

## Non-goals
- No "mystery meat" steps. Everything must be scripted.
- Avoid manual clicking. Prefer scripts, config files, and Make-like commands.
- Don't chase SOTA. Optimize for reliability + iteration speed.

## Operating principles
- Make it work, then make it right, then make it fast.
- Provide a single "golden path" command for each stage.
- Keep configs in YAML, runs in runs/, and data immutable in data/raw.

## Acceptance criteria
1) `scripts/setup_wsl.sh` completes without errors on WSL2.
2) `scripts/prepare_dataset.py` converts raw data -> `data/splits/{train,val,test}.jsonl`.
3) `scripts/train_sft.py --config configs/sft_qlora.yaml` launches training on one GPU.
4) `scripts/evaluate.py` reports eval loss/perplexity and saves a short report in `runs/<run_id>/eval_report.json`.
5) `scripts/export_merge_adapter.py` exports adapter and optionally merges into base model.
6) README includes the exact commands to reproduce.

## Dataset contract (JSONL)
Each line is one example:
```json
{
  "id": "string",
  "messages": [
    {"role":"system","content":"...optional..."},
    {"role":"user","content":"..."},
    {"role":"assistant","content":"..."}
  ],
  "meta": {"source":"...", "tags":["..."], "split":"train|val|test"}
}
```

## Implementation notes
- Use Hugging Face transformers + datasets + accelerate.
- Use PEFT for LoRA.
- For QLoRA: use bitsandbytes 4-bit quantization.
- Use TRL SFTTrainer (or transformers Trainer if simpler) with a formatting function.
- Ensure deterministic splits (seeded).
- Add a small smoke test run config (tiny subset) to validate end-to-end quickly.

## Deliverables
- Repo scaffold (folders, scripts, configs).
- Setup scripts for WSL2.
- Training + eval scripts with configs.
- Clear README.

