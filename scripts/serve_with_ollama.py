#!/usr/bin/env python3
"""Interactive script to serve trained models with Ollama."""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, load_json


def find_available_runs(runs_dir: Path) -> List[Tuple[str, Path]]:
    """Find all available training runs."""
    runs = []
    if not runs_dir.exists():
        return runs
    
    for run_path in sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_path.is_dir():
            continue
        
        # Check if this looks like a valid run directory
        config_path = run_path / "config.json"
        adapter_safetensors = run_path / "adapter_model.safetensors"
        adapter_config = run_path / "adapter_config.json"
        
        if config_path.exists() and (adapter_safetensors.exists() or adapter_config.exists()):
            runs.append((run_path.name, run_path))
    
    return runs


def check_merged_model(run_dir: Path) -> Optional[Path]:
    """Check if merged model exists in run directory."""
    merged_dir = run_dir / "merged_model"
    if merged_dir.exists() and (merged_dir / "config.json").exists():
        # Check for model files
        has_model_files = any(
            f.suffix in [".safetensors", ".bin"] 
            for f in merged_dir.iterdir()
        )
        if has_model_files:
            return merged_dir
    return None


def get_model_info(run_dir: Path) -> dict:
    """Get model information from run directory."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    
    config = load_json(str(config_path))
    base_model = config.get("model", {}).get("base_model", "unknown")
    run_name = config.get("run", {}).get("name", run_dir.name)
    
    return {
        "base_model": base_model,
        "run_name": run_name,
        "run_dir": run_dir,
    }


def generate_modelfile(merged_model_dir: Path, model_name: str, output_path: Path) -> None:
    """Generate an Ollama Modelfile from a merged HuggingFace model."""
    # Use absolute path for the FROM directive
    abs_model_dir = merged_model_dir.resolve()
    
    modelfile_content = f"""FROM {abs_model_dir}

# Model created from training run: {model_name}
# To use this model:
#   ollama create {model_name} -f {output_path.resolve()}
#   ollama run {model_name}
"""
    
    output_path.write_text(modelfile_content)


def main():
    parser = argparse.ArgumentParser(
        description="Serve trained models with Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode - list and select model
  python scripts/serve_with_ollama.py
  
  # Direct mode - specify run directory
  python scripts/serve_with_ollama.py --run runs/qlora_sft_20260113_101843
  
  # Skip merge check (if already merged)
  python scripts/serve_with_ollama.py --run runs/qlora_sft_20260113_101843 --skip-merge
        """
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Path to training run directory (if not provided, will list available runs)"
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory containing training runs (default: runs)"
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merge check (assume merged model exists or will merge separately)"
    )
    parser.add_argument(
        "--merge-now",
        action="store_true",
        help="Automatically merge adapter if merged model doesn't exist"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Ollama model name (default: sanitized run directory name)"
    )
    
    args = parser.parse_args()
    
    logger = setup_logging()
    runs_dir = Path(args.runs_dir)
    
    # Find or select run
    if args.run:
        run_dir = Path(args.run)
        if not run_dir.exists():
            logger.error(f"Run directory not found: {run_dir}")
            return 1
    else:
        # Interactive mode - list available runs
        runs = find_available_runs(runs_dir)
        if not runs:
            logger.error(f"No training runs found in {runs_dir}")
            logger.error("Make sure you have completed training first.")
            return 1
        
        print("\nAvailable trained models:")
        print("-" * 60)
        for i, (run_name, run_path) in enumerate(runs, 1):
            info = get_model_info(run_path)
            base_model = info.get("base_model", "unknown")
            merged = "[merged]" if check_merged_model(run_path) else "[adapter only - needs merge]"
            print(f"{i}. {run_name}")
            print(f"   Base model: {base_model}")
            print(f"   Status: {merged}")
            print()
        
        try:
            choice = input("Select a model (number): ").strip()
            idx = int(choice) - 1
            if idx < 0 or idx >= len(runs):
                logger.error("Invalid selection")
                return 1
            run_dir = runs[idx][1]
        except (ValueError, KeyboardInterrupt):
            logger.error("Invalid input or cancelled")
            return 1
    
    logger.info(f"Selected run: {run_dir.name}")
    
    # Get model info
    info = get_model_info(run_dir)
    base_model = info.get("base_model", "unknown")
    
    # Check for merged model
    merged_model_dir = check_merged_model(run_dir)
    
    if not merged_model_dir and not args.skip_merge:
        logger.warning("Merged model not found!")
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Base model: {base_model}")
        logger.info("\nTo merge the adapter with the base model, run:")
        print(f"  python scripts/export_merge_adapter.py --run {run_dir} --merge")
        print()
        
        if args.merge_now:
            logger.info("Merging adapter now...")
            import subprocess
            result = subprocess.run(
                [sys.executable, "scripts/export_merge_adapter.py", "--run", str(run_dir), "--merge"],
                cwd=Path(__file__).parent.parent
            )
            if result.returncode != 0:
                logger.error("Merge failed!")
                return 1
            merged_model_dir = check_merged_model(run_dir)
            if not merged_model_dir:
                logger.error("Merge completed but merged model not found")
                return 1
        else:
            logger.info("Re-run with --merge-now to automatically merge, or merge manually first.")
            return 1
    
    if not merged_model_dir:
        logger.error("No merged model found and --skip-merge was used")
        logger.error("Please merge the adapter first or remove --skip-merge flag")
        return 1
    
    logger.info(f"Found merged model at: {merged_model_dir}")
    
    # Determine Ollama model name
    if args.model_name:
        ollama_model_name = args.model_name
    else:
        # Sanitize run directory name for Ollama (no special chars, lowercase)
        ollama_model_name = run_dir.name.lower().replace(" ", "-").replace("_", "-")
        # Remove any remaining invalid characters
        ollama_model_name = "".join(c for c in ollama_model_name if c.isalnum() or c == "-")
    
    # Generate Modelfile
    modelfile_path = run_dir / "Modelfile"
    generate_modelfile(merged_model_dir, run_dir.name, modelfile_path)
    logger.info(f"Generated Modelfile: {modelfile_path}")
    
    # Generate commands
    print("\n" + "=" * 70)
    print("OLLAMA SETUP COMMANDS")
    print("=" * 70)
    print()
    print(f"# 1. Create Ollama model from Modelfile")
    print(f"ollama create {ollama_model_name} -f {modelfile_path.resolve()}")
    print()
    print(f"# 2. Run/chat with the model")
    print(f"ollama run {ollama_model_name}")
    print()
    print(f"# Or use the API:")
    print(f"curl http://localhost:11434/api/generate -d '{{\"model\": \"{ollama_model_name}\", \"prompt\": \"Hello!\"}}'")
    print()
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

