#!/usr/bin/env python3
"""Generate synthetic dataset for testing the pipeline."""
import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, ensure_dir


SYNTHETIC_INSTRUCTIONS = [
    "Explain what machine learning is.",
    "Write a Python function to calculate the factorial of a number.",
    "What are the benefits of using version control?",
    "How does a neural network learn?",
    "Describe the difference between supervised and unsupervised learning.",
    "Write a SQL query to find all users who registered in the last month.",
    "What is the difference between a list and a tuple in Python?",
    "Explain the concept of recursion.",
    "How do you handle missing data in a dataset?",
    "What is the purpose of a database index?",
    "Describe the MVC architecture pattern.",
    "Explain how HTTP requests work.",
    "What is the difference between GET and POST requests?",
    "How does garbage collection work in Python?",
    "What are the principles of clean code?",
    "Explain the difference between shallow and deep copying.",
    "What is the purpose of unit testing?",
    "How does a hash table work?",
    "Describe the process of building a web application.",
    "What are the advantages of using cloud computing?",
]

SYNTHETIC_RESPONSES = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
    "Here's a Python function to calculate factorial:\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
    "Version control helps track changes, collaborate with teams, revert mistakes, and maintain project history.",
    "Neural networks learn by adjusting weights through backpropagation, minimizing a loss function using gradient descent.",
    "Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data.",
    "SELECT * FROM users WHERE registration_date >= DATE_SUB(NOW(), INTERVAL 1 MONTH);",
    "A list is mutable and ordered, while a tuple is immutable and ordered.",
    "Recursion is when a function calls itself to solve a problem by breaking it into smaller subproblems.",
    "Missing data can be handled by deletion, imputation (mean/median), or using algorithms that handle missing values.",
    "A database index improves query performance by creating a data structure that allows faster data retrieval.",
]


def generate_synthetic_example(idx: int, instruction: str, response: str) -> dict:
    """Generate a synthetic example in Alpaca format.
    
    Args:
        idx: Example index
        instruction: Instruction text
        response: Response text
    
    Returns:
        Example dictionary
    """
    return {
        "id": f"synthetic_{idx:04d}",
        "instruction": instruction,
        "input": "",
        "output": response,
        "system": "You are a helpful AI assistant."
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 examples
  python scripts/generate_synthetic_data.py --count 100
  
  # Generate 50 examples to a specific directory
  python scripts/generate_synthetic_data.py --count 50 --output data/raw/synthetic
        """
    )
    parser.add_argument("--count", type=int, default=100, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="data/raw/synthetic", help="Output directory")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Generate examples
    examples = []
    import random
    
    for i in range(args.count):
        instruction = random.choice(SYNTHETIC_INSTRUCTIONS)
        response = random.choice(SYNTHETIC_RESPONSES)
        example = generate_synthetic_example(i, instruction, response)
        examples.append(example)
    
    # Save to JSON file (Alpaca format)
    output_dir = Path(args.output)
    ensure_dir(output_dir)
    output_file = output_dir / "synthetic_alpaca.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Generated {len(examples)} synthetic examples")
    logger.info(f"Saved to {output_file}")
    logger.info(f"Process with: python scripts/prepare_dataset.py --input {output_file.parent} --output data/splits")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

