#!/usr/bin/env python3
"""Train SFT model with LoRA or QLoRA."""
import argparse
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
from dotenv import load_dotenv

from src.utils import setup_logging, load_config, set_seed, get_run_dir, save_json
from src.formatting import format_messages, get_model_max_length

# Load environment variables
load_dotenv()


def load_model_and_tokenizer(config, logger):
    """Load model and tokenizer with QLoRA or LoRA configuration."""
    model_config = config["model"]
    train_config = config["train"]
    quant_config = config.get("quant", {})
    
    base_model = model_config["base_model"]
    logger.info(f"Loading model: {base_model}")
    
    # Get HF token
    hf_token = os.getenv("HF_TOKEN")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        token=hf_token,
        trust_remote_code=model_config.get("trust_remote_code", False),
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    if train_config["method"] == "qlora":
        logger.info("Using QLoRA (4-bit quantization)")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.get("load_in_4bit", True),
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=getattr(torch, quant_config.get("bnb_4bit_compute_dtype", "bfloat16")),
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
            trust_remote_code=model_config.get("trust_remote_code", False),
            torch_dtype=torch.bfloat16 if train_config.get("bf16", False) else torch.float16,
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        logger.info("Using LoRA (full precision)")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            token=hf_token,
            trust_remote_code=model_config.get("trust_remote_code", False),
            torch_dtype=torch.bfloat16 if train_config.get("bf16", False) else torch.float16,
        )
    
    # Configure LoRA
    lora_config_dict = config["lora"]
    lora_config = LoraConfig(
        r=lora_config_dict["r"],
        lora_alpha=lora_config_dict["alpha"],
        target_modules=lora_config_dict["target_modules"],
        lora_dropout=lora_config_dict["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def load_dataset_from_jsonl(train_path, eval_path, logger):
    """Load dataset from JSONL files."""
    logger.info(f"Loading training data from {train_path}")
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    
    eval_dataset = None
    if eval_path and Path(eval_path).exists():
        logger.info(f"Loading eval data from {eval_path}")
        eval_dataset = load_dataset("json", data_files=eval_path, split="train")
    
    return train_dataset, eval_dataset


def formatting_func(examples):
    """Format function for SFTTrainer."""
    # This will be called with tokenizer in the trainer
    # For now, we'll use a simpler approach
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Train SFT model with LoRA/QLoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with QLoRA config
  python scripts/train_sft.py --config configs/sft_qlora.yaml
  
  # Smoke test (50 steps)
  python scripts/train_sft.py --config configs/sft_qlora.yaml --smoke
  
  # Train with LoRA config
  python scripts/train_sft.py --config configs/sft_lora.yaml
        """
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (50 steps, small subset)")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get run directory
    run_dir = get_run_dir(
        config["run"]["output_dir"],
        config["run"]["name"],
        create=True
    )
    
    # Setup logging
    logger = setup_logging(log_file=str(run_dir / "training.log"))
    logger.info(f"Starting training run: {run_dir.name}")
    logger.info(f"Config: {args.config}")
    
    # Set seed
    set_seed(config["run"]["seed"])
    
    # Save config to run directory
    save_json(config, str(run_dir / "config.json"))
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(config, logger)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Make sure:")
        logger.error("  1. HF_TOKEN is set in .env file (for gated models)")
        logger.error("  2. Model name is correct")
        logger.error("  3. You have sufficient VRAM")
        if "qlora" in str(args.config).lower():
            logger.error("  4. QLoRA failed - try LoRA instead: python scripts/train_sft.py --config configs/sft_lora.yaml")
        return 1
    
    # Load dataset
    data_config = config["data"]
    train_dataset, eval_dataset = load_dataset_from_jsonl(
        data_config["train_path"],
        data_config.get("eval_path"),
        logger
    )
    
    # Smoke test: use small subset
    if args.smoke:
        logger.info("Running smoke test (using first 50 examples)")
        train_dataset = train_dataset.select(range(min(50, len(train_dataset))))
        if eval_dataset:
            eval_dataset = eval_dataset.select(range(min(10, len(eval_dataset))))
    
    # Training arguments
    train_config = config["train"]
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        num_train_epochs=train_config["num_train_epochs"] if not args.smoke else 1,
        max_steps=50 if args.smoke else None,
        warmup_ratio=train_config["warmup_ratio"],
        logging_steps=train_config.get("logging_steps", 10),
        eval_steps=train_config.get("eval_steps") if eval_dataset else None,
        save_steps=train_config.get("save_steps", 500),
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        bf16=train_config.get("bf16", False),
        fp16=not train_config.get("bf16", False) and not train_config["method"] == "qlora",
        gradient_checkpointing=train_config.get("gradient_checkpointing", False),
        optim=train_config.get("optim", "adamw_torch"),
        report_to="none",  # Disable wandb/tensorboard by default
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False if eval_dataset else None,
        seed=config["run"]["seed"],
    )
    
    # Formatting function for messages
    def format_prompts(examples):
        """Format examples with chat template."""
        texts = []
        messages_col = examples.get("messages", [])
        
        # messages_col is a list where each element is the messages from one example
        for messages in messages_col:
            if isinstance(messages, str):
                # Already formatted text
                texts.append(messages)
            elif isinstance(messages, list):
                # List of message dicts
                text = format_messages(messages, tokenizer, add_generation_prompt=False)
                texts.append(text)
            else:
                # Fallback: convert to string
                texts.append(str(messages))
        
        return {"text": texts}
    
    # Apply formatting
    columns_to_remove = [col for col in train_dataset.column_names if col != "messages"]
    train_dataset = train_dataset.map(
        format_prompts,
        batched=True,
        remove_columns=columns_to_remove,
    )
    
    if eval_dataset:
        columns_to_remove = [col for col in eval_dataset.column_names if col != "messages"]
        eval_dataset = eval_dataset.map(
            format_prompts,
            batched=True,
            remove_columns=columns_to_remove,
        )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=config["data"]["max_seq_length"],
        packing=config["data"].get("packing", False),
        dataset_text_field="text",
    )
    
    # Train
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("Out of memory error!")
            logger.error("Try:")
            logger.error("  1. Reduce batch size in config")
            logger.error("  2. Reduce max_seq_length in config")
            logger.error("  3. Use QLoRA instead of LoRA (if not already)")
            logger.error("  4. Use gradient_checkpointing: true")
        else:
            logger.error(f"Training error: {e}")
        return 1
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(str(run_dir))
    
    logger.info(f"Training complete! Model saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

