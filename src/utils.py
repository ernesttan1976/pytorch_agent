"""Common utilities for the fine-tuning pipeline."""
import os
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up structured logging with timestamps.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("llm_ft")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file with validation.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    
    return config


def ensure_dir(path: str) -> Path:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(data: Dict[str, Any], path: str, indent: int = 2) -> None:
    """Save dictionary to JSON file.
    
    Args:
        data: Data to save
        path: Output file path
        indent: JSON indentation
    """
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file.
    
    Args:
        path: JSON file path
    
    Returns:
        Loaded dictionary
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_run_dir(base_dir: str, run_name: str, create: bool = True) -> Path:
    """Get run directory path with timestamp.
    
    Args:
        base_dir: Base output directory
        run_name: Run name
        create: Whether to create the directory
    
    Returns:
        Path to run directory
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{run_name}_{timestamp}"
    
    if create:
        ensure_dir(run_dir)
    
    return run_dir


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token average).
    
    Args:
        text: Input text
    
    Returns:
        Estimated token count
    """
    return len(text) // 4
