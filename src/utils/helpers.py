"""
src/utils/helpers.py — Shared utility functions used across the project.
"""

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with timestamp + level format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s — %(levelname)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def ensure_dirs(*paths: str | Path) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def memory_usage_mb(df) -> float:
    """Return DataFrame memory usage in MB."""
    return df.memory_usage(deep=True).sum() / (1024 ** 2)


def print_shape_report(df, label: str = "DataFrame") -> None:
    """Print shape and memory usage."""
    mb = memory_usage_mb(df)
    print(f"{label}: {df.shape[0]:,} rows × {df.shape[1]} columns | {mb:.1f} MB")
