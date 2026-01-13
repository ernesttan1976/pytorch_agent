#!/usr/bin/env python3
"""Evaluate fine-tuned model."""
import argparse
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from dotenv import load_dotenv

from src.utils import setup_logging, load_config, save_json, load_json, setup_temp_dirs
from src.metrics import calculate_perplexity
from src.formatting import format_messages

load_dotenv()

# Configure temp directories to use D drive for swap files
setup_temp_dirs()


def evaluate_loss(model, tokenizer, eval_dataset, config, logger):
    """Evaluate model loss on evaluation dataset."""
    logger.info("Evaluating model loss...")
    
    eval_config = config["eval"]
    max_samples = eval_config.get("max_samples")
    batch_size = eval_config.get("batch_size", 4)
    max_length = eval_config.get("max_length", 4096)
    
    if max_samples:
        eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    device = next(model.parameters()).device
    
    for i in range(0, len(eval_dataset), batch_size):
        batch = eval_dataset[i:i+batch_size]
        
        texts = []
        for example in batch:
            if isinstance(example, str):
                # Already formatted text
                texts.append(example)
            elif isinstance(example, dict):
                messages = example.get("messages", [])
                if isinstance(messages, str):
                    # Already formatted
                    texts.append(messages)
                else:
                    text = format_messages(messages, tokenizer, add_generation_prompt=False)
                    texts.append(text)
            else:
                # Fallback
                texts.append(str(example))
        
        # Tokenize
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        
        total_loss += loss.item() * inputs["input_ids"].numel()
        total_tokens += inputs["input_ids"].numel()
        num_batches += 1
        
        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"  Processed {i + len(batch)}/{len(eval_dataset)} examples")
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = calculate_perplexity(avg_loss)
    
    logger.info(f"Evaluation loss: {avg_loss:.4f}")
    logger.info(f"Perplexity: {perplexity:.4f}")
    
    return {
        "eval_loss": avg_loss,
        "perplexity": perplexity,
        "num_samples": len(eval_dataset),
        "total_tokens": total_tokens,
    }


def evaluate_prompts(model, tokenizer, prompts_path, config, logger):
    """Evaluate model on prompt set."""
    if not Path(prompts_path).exists():
        logger.warning(f"Prompt file not found: {prompts_path}")
        return []
    
    logger.info(f"Evaluating on prompts from {prompts_path}")
    
    prompts_config = config.get("prompts", {})
    num_generations = prompts_config.get("num_generations", 5)
    max_new_tokens = prompts_config.get("max_new_tokens", 512)
    temperature = prompts_config.get("temperature", 0.7)
    top_p = prompts_config.get("top_p", 0.9)
    
    # Load prompts
    prompts = []
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    import json
                    prompt_data = json.loads(line)
                    prompts.append(prompt_data)
                except:
                    # Treat as plain text
                    prompts.append({"prompt": line})
    
    if num_generations:
        prompts = prompts[:num_generations]
    
    results = []
    device = next(model.parameters()).device
    model.eval()
    
    for i, prompt_data in enumerate(prompts):
        prompt = prompt_data.get("prompt", prompt_data.get("messages", ""))
        
        # Format if messages
        if isinstance(prompt, list):
            text = format_messages(prompt, tokenizer, add_generation_prompt=True)
        else:
            text = prompt
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only new tokens
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        result = {
            "prompt": prompt if isinstance(prompt, str) else prompt_data,
            "response": response,
            "full_text": generated_text,
        }
        results.append(result)
        
        logger.info(f"  Prompt {i+1}/{len(prompts)} completed")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a training run
  python scripts/evaluate.py --run runs/qlora_sft_20240101_120000 --config configs/eval_lm.yaml
  
  # Evaluate without prompts
  python scripts/evaluate.py --run runs/qlora_sft_20240101_120000
        """
    )
    parser.add_argument("--run", type=str, required=True, help="Path to training run directory")
    parser.add_argument("--config", type=str, default="configs/eval_lm.yaml", help="Evaluation config file")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1
    
    # Setup logging
    logger = setup_logging(log_file=str(run_dir / "evaluation.log"))
    logger.info(f"Evaluating model from: {run_dir}")
    
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.warning(f"Could not load config {args.config}, using defaults: {e}")
        config = {"eval": {}, "prompts": {"enabled": False}}
    
    # Load training config to get base model
    train_config_path = run_dir / "config.json"
    if train_config_path.exists():
        train_config = load_json(str(train_config_path))
        base_model = train_config["model"]["base_model"]
    else:
        logger.error("Could not find config.json in run directory")
        return 1
    
    # Load model
    logger.info(f"Loading model from {run_dir}")
    hf_token = os.getenv("HF_TOKEN")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Load adapter - check both adapter_model directory and root directory
    adapter_path = run_dir / "adapter_model"
    if not adapter_path.exists() and (run_dir / "adapter_config.json").exists():
        # Adapter files are in the root directory
        adapter_path = run_dir
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model_obj, str(adapter_path))
        logger.info("Loaded adapter model")
    else:
        model = base_model_obj
        logger.warning("No adapter found, using base model")
    
    # Load eval dataset
    eval_path = train_config["data"].get("eval_path")
    eval_dataset = None
    if eval_path and Path(eval_path).exists():
        logger.info(f"Loading eval dataset from {eval_path}")
        eval_dataset = load_dataset("json", data_files=eval_path, split="train")
    
    # Evaluate loss
    eval_results = {}
    if eval_dataset:
        loss_results = evaluate_loss(model, tokenizer, eval_dataset, config, logger)
        eval_results.update(loss_results)
    else:
        logger.warning("No eval dataset available, skipping loss evaluation")
    
    # Evaluate prompts
    prompts_results = []
    if config.get("prompts", {}).get("enabled", False):
        prompts_path = config["prompts"].get("path")
        if prompts_path:
            prompts_results = evaluate_prompts(model, tokenizer, prompts_path, config, logger)
            eval_results["prompt_results"] = prompts_results
    
    # Save results
    report_path = run_dir / config.get("output", {}).get("report_path", "eval_report.json")
    save_json(eval_results, str(report_path))
    logger.info(f"Evaluation report saved to: {report_path}")
    
    # Generate samples markdown
    if prompts_results:
        samples_path = run_dir / config.get("output", {}).get("samples_path", "samples.md")
        with open(samples_path, "w", encoding="utf-8") as f:
            f.write("# Evaluation Samples\n\n")
            for i, result in enumerate(prompts_results, 1):
                f.write(f"## Sample {i}\n\n")
                f.write("### Prompt\n\n")
                if isinstance(result["prompt"], str):
                    f.write(result["prompt"] + "\n\n")
                else:
                    f.write("```json\n" + str(result["prompt"]) + "\n```\n\n")
                f.write("### Response\n\n")
                f.write(result["response"] + "\n\n")
                f.write("---\n\n")
        logger.info(f"Sample outputs saved to: {samples_path}")
    
    logger.info("Evaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

