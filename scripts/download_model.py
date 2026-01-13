#!/usr/bin/env python3
"""Download and cache a model from Hugging Face."""
import argparse
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(
        description="Download and cache a model from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download model
  python scripts/download_model.py --model meta-llama/Meta-Llama-3-8B-Instruct
  
  # Download with custom cache directory
  python scripts/download_model.py --model meta-llama/Meta-Llama-3-8B-Instruct --cache-dir ~/models
        """
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--cache-dir", type=str, default=None, help="Custom cache directory")
    parser.add_argument("--token-only", action="store_true", help="Only download tokenizer")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set in environment. Some models may require authentication.")
    
    logger.info(f"Downloading model: {args.model}")
    
    try:
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            token=hf_token,
            cache_dir=args.cache_dir,
            trust_remote_code=True
        )
        logger.info("✓ Tokenizer downloaded")
        
        if not args.token_only:
            # Download model (just to cache it, we don't load it fully)
            logger.info("Downloading model (this may take a while)...")
            # Use from_pretrained with torch_dtype to avoid loading into memory
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                token=hf_token,
                cache_dir=args.cache_dir,
                trust_remote_code=True,
                torch_dtype="float16",  # Use float16 to reduce memory during download
                low_cpu_mem_usage=True
            )
            logger.info("✓ Model downloaded and cached")
            del model  # Free memory
        
        logger.info("✓ Download complete")
        logger.info(f"Model cached in: {args.cache_dir or 'default Hugging Face cache'}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        logger.error("Make sure you have:")
        logger.error("  1. HF_TOKEN set in .env file (for gated models)")
        logger.error("  2. Sufficient disk space")
        logger.error("  3. Internet connection")
        return 1


if __name__ == "__main__":
    sys.exit(main())

