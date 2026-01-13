#!/usr/bin/env python3
"""Quick verification script to check if setup is correct."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_python():
    """Check Python version."""
    print("✓ Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  ✗ Python {version.major}.{version.minor} is too old. Need 3.8+")
        return False
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_torch():
    """Check PyTorch and CUDA."""
    print("\n✓ Checking PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.version.cuda}")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Test tensor operation
            x = torch.randn(10, 10).cuda()
            y = torch.randn(10, 10).cuda()
            z = torch.matmul(x, y)
            print("  ✓ CUDA tensor operations working")
            return True
        else:
            print("  ✗ CUDA not available")
            return False
    except ImportError:
        print("  ✗ PyTorch not installed")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def check_dependencies():
    """Check required packages."""
    print("\n✓ Checking dependencies...")
    required = [
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "sentencepiece",
        "safetensors",
        "yaml",
        "dotenv",
    ]
    
    missing = []
    for pkg in required:
        try:
            if pkg == "yaml":
                __import__("yaml")
            elif pkg == "dotenv":
                __import__("dotenv")
            else:
                __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} not installed")
            missing.append(pkg)
    
    # Check bitsandbytes (optional for QLoRA)
    try:
        import bitsandbytes
        print(f"  ✓ bitsandbytes (optional, for QLoRA)")
    except ImportError:
        print(f"  ⚠ bitsandbytes not installed (needed for QLoRA, but LoRA will work)")
    
    return len(missing) == 0

def check_bitsandbytes():
    """Check bitsandbytes (optional)."""
    print("\n✓ Checking bitsandbytes...")
    try:
        import bitsandbytes as bnb
        print(f"  ✓ bitsandbytes {bnb.__version__}")
        return True
    except ImportError:
        print("  ⚠ bitsandbytes not installed (QLoRA won't work, use LoRA instead)")
        return False

def check_files():
    """Check required files exist."""
    print("\n✓ Checking project files...")
    repo_root = Path(__file__).parent.parent
    
    required_files = [
        "scripts/train_sft.py",
        "scripts/prepare_dataset.py",
        "scripts/evaluate.py",
        "scripts/generate_synthetic_data.py",
        "configs/sft_qlora.yaml",
        "configs/sft_lora.yaml",
        "src/utils.py",
        "src/formatting.py",
    ]
    
    missing = []
    for file_path in required_files:
        full_path = repo_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} missing")
            missing.append(file_path)
    
    return len(missing) == 0

def check_env():
    """Check environment variables."""
    print("\n✓ Checking environment variables...")
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and hf_token != "your_huggingface_token_here":
        print("  ✓ HF_TOKEN is set")
        return True
    else:
        print("  ⚠ HF_TOKEN not set (needed for gated models like Llama)")
        print("    You can still use public models without it")
        return True  # Not critical

def check_data_dirs():
    """Check data directories exist."""
    print("\n✓ Checking data directories...")
    repo_root = Path(__file__).parent.parent
    
    dirs = ["data/raw", "data/splits", "data/interim", "runs"]
    all_exist = True
    
    for dir_path in dirs:
        full_path = repo_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}/ exists")
        else:
            print(f"  ⚠ {dir_path}/ doesn't exist (will be created when needed)")
    
    return True

def check_accelerate():
    """Check accelerate configuration."""
    print("\n✓ Checking accelerate configuration...")
    try:
        from accelerate.utils import write_basic_config
        import os
        config_path = os.path.expanduser("~/.cache/huggingface/accelerate/default_config.yaml")
        if os.path.exists(config_path):
            print("  ✓ Accelerate config exists")
            return True
        else:
            print("  ⚠ Accelerate config not found")
            print("    Run: bash scripts/setup_accelerate.sh")
            return False
    except Exception as e:
        print(f"  ⚠ Could not check accelerate config: {e}")
        return False

def main():
    """Run all checks."""
    print("=" * 50)
    print("PyTorch Fine-Tuning Pipeline - Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python", check_python),
        ("PyTorch & CUDA", check_torch),
        ("Dependencies", check_dependencies),
        ("bitsandbytes", check_bitsandbytes),
        ("Project Files", check_files),
        ("Environment Variables", check_env),
        ("Data Directories", check_data_dirs),
        ("Accelerate Config", check_accelerate),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ✗ Error during {name} check: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All critical checks passed! You're ready to train.")
        print("\nNext steps:")
        print("  1. python scripts/generate_synthetic_data.py --count 100")
        print("  2. python scripts/prepare_dataset.py --input data/raw --output data/splits")
        print("  3. python scripts/train_sft.py --config configs/sft_qlora.yaml --smoke")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Run: bash scripts/setup_wsl.sh")
        print("  - Run: bash scripts/setup_accelerate.sh")
        print("  - Create .env file with HF_TOKEN (if needed)")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

