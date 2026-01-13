#!/usr/bin/env python3
"""Sanity check dataset format and report statistics."""
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_schema import load_jsonl, validate_example
from src.utils import setup_logging, estimate_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Sanity check dataset format and report statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check train split
  python scripts/sanity_check_dataset.py data/splits/train.jsonl
  
  # Check all splits
  python scripts/sanity_check_dataset.py data/splits/*.jsonl
        """
    )
    parser.add_argument("files", nargs="+", help="JSONL files to check")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    total_examples = 0
    total_errors = 0
    role_counts = {"system": 0, "user": 0, "assistant": 0}
    total_tokens = 0
    longest_example = None
    longest_tokens = 0
    
    for file_path in args.files:
        logger.info(f"Checking {file_path}")
        
        try:
            examples = load_jsonl(file_path)
            logger.info(f"  Loaded {len(examples)} examples")
            
            for ex in examples:
                total_examples += 1
                
                # Validate
                errors = validate_example(ex)
                if errors:
                    total_errors += len(errors)
                    logger.warning(f"  Example {ex.id} has errors: {', '.join(errors)}")
                
                # Count roles
                for msg in ex.messages:
                    role_counts[msg.role] += 1
                    total_tokens += estimate_tokens(msg.content)
                
                # Track longest
                example_text = " ".join(msg.content for msg in ex.messages)
                tokens = estimate_tokens(example_text)
                if tokens > longest_tokens:
                    longest_tokens = tokens
                    longest_example = (ex.id, file_path)
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            total_errors += 1
    
    # Report
    logger.info("\n" + "="*50)
    logger.info("Dataset Statistics")
    logger.info("="*50)
    logger.info(f"Total examples: {total_examples}")
    logger.info(f"Total errors: {total_errors}")
    logger.info(f"\nRole distribution:")
    for role, count in role_counts.items():
        logger.info(f"  {role}: {count}")
    logger.info(f"\nEstimated total tokens: {total_tokens:,}")
    if total_examples > 0:
        logger.info(f"Average tokens per example: {total_tokens / total_examples:.1f}")
    if longest_example:
        logger.info(f"Longest example: {longest_example[0]} ({longest_tokens} tokens) in {longest_example[1]}")
    
    if total_errors == 0:
        logger.info("\n✓ Dataset validation passed!")
        return 0
    else:
        logger.warning(f"\n✗ Dataset validation found {total_errors} errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())

