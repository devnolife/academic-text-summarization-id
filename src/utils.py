# -*- coding: utf-8 -*-
"""
Utility functions for the NLP Summarization Pipeline.

Provides helper functions for file I/O, logging, reproducibility,
and text statistics computation.
"""

import json
import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """
    Configure the logging system for the entire pipeline.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger.info("Logging configured at level: %s", level)


def save_json(data: Union[Dict, List], path: str) -> None:
    """
    Save a dictionary or list to a JSON file.

    Args:
        data: Dictionary or list to serialize.
        path: File path to save the JSON output.

    Raises:
        IOError: If the file cannot be written.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("JSON saved to: %s", path)
    except IOError as e:
        logger.error("Failed to save JSON to %s: %s", path, e)
        raise


def load_json(path: str) -> Union[Dict, List]:
    """
    Load a JSON file and return its contents.

    Args:
        path: File path to the JSON file.

    Returns:
        Parsed JSON content as a dictionary or list.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not os.path.exists(path):
        logger.error("JSON file not found: %s", path)
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("JSON loaded from: %s", path)
        return data
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in file %s: %s", path, e)
        raise


def save_summaries(summaries: List[str], path: str) -> None:
    """
    Save a list of summaries to a text file, one summary per block.

    Args:
        summaries: List of summary strings to save.
        path: File path to save the text output.

    Raises:
        IOError: If the file cannot be written.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for i, summary in enumerate(summaries):
                f.write(f"--- Summary {i + 1} ---\n")
                f.write(summary.strip() + "\n\n")
        logger.info("Summaries saved to: %s (%d summaries)", path, len(summaries))
    except IOError as e:
        logger.error("Failed to save summaries to %s: %s", path, e)
        raise


def log_message(message: str, level: str = "INFO") -> None:
    """
    Simple logging wrapper that logs a message at the specified level.

    Args:
        message: The message to log.
        level: Logging level string (INFO, WARNING, ERROR, DEBUG).
    """
    level_upper = level.upper()
    if level_upper == "DEBUG":
        logger.debug(message)
    elif level_upper == "WARNING":
        logger.warning(message)
    elif level_upper == "ERROR":
        logger.error(message)
    else:
        logger.info(message)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info("Random seed set to %d (Python, NumPy, PyTorch)", seed)
    except ImportError:
        logger.info(
            "Random seed set to %d (Python, NumPy only — PyTorch not available)", seed
        )


def compute_text_stats(texts: List[str]) -> Dict[str, float]:
    """
    Compute statistics on text lengths for a list of text strings.

    Args:
        texts: List of text strings.

    Returns:
        Dictionary with keys: 'count', 'avg_length', 'min_length',
        'max_length', 'median_length'.
    """
    if not texts:
        return {
            "count": 0,
            "avg_length": 0.0,
            "min_length": 0,
            "max_length": 0,
            "median_length": 0.0,
        }

    lengths = [len(text) for text in texts]
    return {
        "count": len(texts),
        "avg_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "median_length": float(np.median(lengths)),
    }


def truncate_text(text: str, max_chars: int) -> str:
    """
    Safely truncate text to a maximum character limit.

    Truncation occurs at the nearest word boundary before the limit
    to avoid cutting words in half.

    Args:
        text: Input text string.
        max_chars: Maximum number of characters allowed.

    Returns:
        Truncated text string.
    """
    if not text or len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Try to truncate at a word boundary
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:  # Only use word boundary if reasonable
        truncated = truncated[:last_space]

    logger.debug("Text truncated from %d to %d characters", len(text), len(truncated))
    return truncated


if __name__ == "__main__":
    """Test utility functions when run directly."""
    setup_logging("DEBUG")

    # Test set_seed
    set_seed(42)

    # Test text stats
    sample_texts = [
        "Ini adalah contoh teks pertama.",
        "Teks kedua lebih panjang dari teks pertama yang ada.",
        "Teks ketiga.",
    ]
    stats = compute_text_stats(sample_texts)
    print(f"Text stats: {stats}")

    # Test truncation
    long_text = "Ini adalah contoh teks yang sangat panjang " * 10
    truncated = truncate_text(long_text, 100)
    print(f"Original length: {len(long_text)}, Truncated length: {len(truncated)}")
    print(f"Truncated: {truncated}")

    # Test JSON save/load
    test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
    test_path = "test_output.json"
    save_json(test_data, test_path)
    loaded = load_json(test_path)
    print(f"Loaded JSON: {loaded}")
    os.remove(test_path)

    print("\nAll utility functions tested successfully.")
