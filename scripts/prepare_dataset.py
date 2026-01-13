#!/usr/bin/env python3
"""Prepare dataset from raw format to standardized JSONL splits."""
import argparse
import json
import random
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loaders import load_examples
from src.dataset_schema import save_jsonl, Example
from src.utils import setup_logging, set_seed, ensure_dir, save_json


def split_examples(
    examples: list,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> tuple:
    """Split examples into train/val/test sets.
    
    Args:
        examples: List of examples
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
    
    Returns:
        Tuple of (train, val, test) lists
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    set_seed(seed)
    random.shuffle(examples)
    
    total = len(examples)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train = examples[:train_end]
    val = examples[train_end:val_end]
    test = examples[val_end:]
    
    # Assign split metadata
    for ex in train:
        if ex.meta is None:
            ex.meta = {}
        ex.meta["split"] = "train"
    
    for ex in val:
        if ex.meta is None:
            ex.meta = {}
        ex.meta["split"] = "val"
    
    for ex in test:
        if ex.meta is None:
            ex.meta = {}
        ex.meta["split"] = "test"
    
    return train, val, test


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token average).
    
    Args:
        text: Input text
    
    Returns:
        Estimated token count
    """
    return len(text) // 4


def generate_stats(examples: list, output_dir: Path) -> dict:
    """Generate dataset statistics.
    
    Args:
        examples: List of examples
        output_dir: Output directory for stats file
    
    Returns:
        Statistics dictionary
    """
    stats = {
        "total_examples": len(examples),
        "role_distribution": {"system": 0, "user": 0, "assistant": 0},
        "total_messages": 0,
        "total_estimated_tokens": 0,
        "longest_example": None,
        "longest_example_tokens": 0,
        "avg_messages_per_example": 0,
        "avg_tokens_per_example": 0
    }
    
    longest_example = None
    longest_tokens = 0
    
    for ex in examples:
        stats["total_messages"] += len(ex.messages)
        example_text = ""
        
        for msg in ex.messages:
            stats["role_distribution"][msg.role] += 1
            example_text += msg.content + " "
        
        tokens = estimate_tokens(example_text)
        stats["total_estimated_tokens"] += tokens
        
        if tokens > longest_tokens:
            longest_tokens = tokens
            longest_example = ex.id
    
    if len(examples) > 0:
        stats["avg_messages_per_example"] = stats["total_messages"] / len(examples)
        stats["avg_tokens_per_example"] = stats["total_estimated_tokens"] / len(examples)
    
    stats["longest_example"] = longest_example
    stats["longest_example_tokens"] = longest_tokens
    
    # Save stats
    ensure_dir(output_dir)
    stats_path = output_dir / "stats.json"
    save_json(stats, str(stats_path))
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset from raw format to standardized JSONL splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in data/raw
  python scripts/prepare_dataset.py --input data/raw --output data/splits --seed 42
  
  # Custom split ratios
  python scripts/prepare_dataset.py --input data/raw --output data/splits --train-ratio 0.9 --val-ratio 0.05 --test-ratio 0.05
        """
    )
    parser.add_argument("--input", type=str, required=True, help="Input directory or file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    set_seed(args.seed)
    
    # Load examples
    logger.info(f"Loading examples from {args.input}")
    examples = load_examples(args.input)
    logger.info(f"Loaded {len(examples)} examples")
    
    if len(examples) == 0:
        logger.error("No examples loaded. Check input path and file formats.")
        return 1
    
    # Split
    logger.info("Splitting examples...")
    train, val, test = split_examples(
        examples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    logger.info(f"Split: {len(train)} train, {len(val)} val, {len(test)} test")
    
    # Save splits
    output_dir = Path(args.output)
    ensure_dir(output_dir)
    
    save_jsonl(train, str(output_dir / "train.jsonl"))
    save_jsonl(val, str(output_dir / "val.jsonl"))
    save_jsonl(test, str(output_dir / "test.jsonl"))
    
    logger.info(f"Saved splits to {output_dir}")
    
    # Generate stats
    stats = generate_stats(examples, Path(args.output).parent / "interim")
    logger.info(f"Dataset statistics:")
    logger.info(f"  Total examples: {stats['total_examples']}")
    logger.info(f"  Total messages: {stats['total_messages']}")
    logger.info(f"  Estimated tokens: {stats['total_estimated_tokens']:,}")
    logger.info(f"  Avg tokens/example: {stats['avg_tokens_per_example']:.1f}")
    logger.info(f"  Longest example: {stats['longest_example']} ({stats['longest_example_tokens']} tokens)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

