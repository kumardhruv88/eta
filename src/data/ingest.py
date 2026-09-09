"""
src/data/ingest.py — Data download and raw loading utilities.

Functions
---------
download_raw_data(url, save_path) → Path
    Downloads a remote file with progress bar; skips if already present.
load_raw_parquet(file_path, sample_size, random_state) → pd.DataFrame
    Loads a parquet file and optionally samples it.
"""

import logging
import os
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_raw_data(url: str, save_path: str | Path) -> Path:
    """
    Download a remote file to save_path with a tqdm progress bar.

    Skips download if the file already exists (idempotent).

    Parameters
    ----------
    url       : str  — full download URL
    save_path : str | Path — local destination (including filename)

    Returns
    -------
    Path — absolute path to the downloaded (or existing) file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        size_mb = save_path.stat().st_size / (1024 ** 2)
        logger.info("File already exists: %s (%.1f MB) — skipping download.", save_path, size_mb)
        return save_path

    logger.info("Downloading %s → %s", url, save_path)

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", 0))

    with open(save_path, "wb") as f, tqdm(
        desc=save_path.name,
        total=total_bytes,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

    size_mb = save_path.stat().st_size / (1024 ** 2)
    logger.info("Download complete: %s (%.1f MB)", save_path, size_mb)
    return save_path


def load_raw_parquet(
    file_path: str | Path,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load a parquet file into a DataFrame, with optional sampling.

    Parameters
    ----------
    file_path    : path to parquet file
    sample_size  : if set, randomly sample this many rows
    random_state : reproducibility seed

    Returns
    -------
    pd.DataFrame
    """
    file_path = Path(file_path)
    logger.info("Loading parquet: %s", file_path)

    df = pd.read_parquet(file_path)
    logger.info("Full dataset shape: %s", df.shape)

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        logger.info("Sampled %d rows (random_state=%d).", sample_size, random_state)

    return df
