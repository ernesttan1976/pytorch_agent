#!/usr/bin/env python3
"""Export adapter and optionally merge with base model."""
import argparse
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from dotenv import load_dotenv

from src.utils import setup_logging, load_json, setup_temp_dirs

load_dotenv()

# Configure temp directories to use D drive for swap files
setup_temp_dirs()


def main():
    parser = argparse.ArgumentParser(
        description="Export adapter and optionally merge with base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export adapter only
  python scripts/export_merge_adapter.py --run runs/qlora_sft_20240101_120000
  
  # Export and merge
  python scripts/export_merge_adapter.py --run runs/qlora_sft_20240101_120000 --merge
  
  # Export to custom directory
  python scripts/export_merge_adapter.py --run runs/qlora_sft_20240101_120000 --output merged_model
        """
    )
    parser.add_argument("--run", type=str, required=True, help="Path to training run directory")
    parser.add_argument("--merge", action="store_true", help="Merge adapter into base model")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: merged_model in run dir)")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return 1
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Processing model from: {run_dir}")
    
    # Load training config
    config_path = run_dir / "config.json"
    if not config_path.exists():
        logger.error("Could not find config.json in run directory")
        return 1
    
    train_config = load_json(str(config_path))
    base_model_name = train_config["model"]["base_model"]
    
    # Check if adapter exists - try both adapter_model directory and root directory
    adapter_path = run_dir / "adapter_model"
    if not adapter_path.exists() and (run_dir / "adapter_config.json").exists():
        # Adapter files are in the root directory
        adapter_path = run_dir
    if not adapter_path.exists() or not (adapter_path / "adapter_config.json").exists():
        logger.error(f"Adapter not found in {run_dir / 'adapter_model'} or {run_dir}")
        logger.error("Make sure training completed successfully")
        return 1
    
    logger.info(f"Found adapter at: {adapter_path}")
    logger.info(f"Base model: {base_model_name}")
    
    # Load base model and adapter
    hf_token = os.getenv("HF_TOKEN")
    
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=train_config["model"].get("trust_remote_code", False),
    )
    
    logger.info("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    
    if args.merge:
        logger.info("Merging adapter into base model...")
        model = model.merge_and_unload()
        
        # Determine output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = run_dir / "merged_model"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving merged model to {output_dir}...")
        model.save_pretrained(str(output_dir), safe_serialization=True)
        
        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
        tokenizer.save_pretrained(str(output_dir))
        
        logger.info(f"✓ Merged model saved to: {output_dir}")
        logger.info("You can now use this model with AutoModelForCausalLM.from_pretrained()")
    else:
        logger.info("Adapter is already exported at: {adapter_path}")
        logger.info("To merge, run with --merge flag")
        logger.info(f"  python scripts/export_merge_adapter.py --run {run_dir} --merge")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

