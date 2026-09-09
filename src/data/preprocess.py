"""
src/data/preprocess.py — Data cleaning and outlier removal.

Functions
---------
clean_data(df)                         → pd.DataFrame
remove_outliers(df, column, method)    → pd.DataFrame
construct_target(df)                   → pd.DataFrame
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# TARGET CONSTRUCTION
# ─────────────────────────────────────────────────────────────

def construct_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive eta_seconds = on_scene_datetime - request_datetime.

    Drops rows where on_scene_datetime is null (MNAR — not safe to impute).
    Drops physically impossible values (< 30 s or > 3600 s).
    """
    df = df.copy()

    # Convert to datetime if not already
    for col in ["request_datetime", "on_scene_datetime"]:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["on_scene_datetime"])
    logger.info("Dropped %d rows with null on_scene_datetime.", before - len(df))

    df["eta_seconds"] = (
        df["on_scene_datetime"] - df["request_datetime"]
    ).dt.total_seconds()

    # Remove physically impossible ETAs
    for condition, label in [
        (df["eta_seconds"] <= 0,    "non-positive ETA"),
        (df["eta_seconds"] > 3600,  "ETA > 1 hour"),
        ((df["trip_miles"] == 0) & (df.get("trip_time", pd.Series([1])) == 0),
         "ghost trips (0 miles, 0 time)"),
    ]:
        n = condition.sum()
        if n:
            df = df[~condition]
            logger.info("Removed %d rows: %s.", n, label)

    logger.info("Rows remaining after target construction: %d", len(df))
    return df


# ─────────────────────────────────────────────────────────────
# GENERAL CLEANING
# ─────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply general cleaning steps:
    - Parse datetime columns
    - Drop fully duplicate rows
    - Enforce sensible dtypes
    """
    df = df.copy()

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate rows.", before - len(df))

    # Ensure datetime columns are parsed
    datetime_cols = [c for c in df.columns if "datetime" in c.lower() or "time" in c.lower()]
    for col in datetime_cols:
        if df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            logger.info("Parsed %s as datetime.", col)

    return df


# ─────────────────────────────────────────────────────────────
# OUTLIER REMOVAL
# ─────────────────────────────────────────────────────────────

def remove_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "IQR",
    z_thresh: float = 3.0,
) -> pd.DataFrame:
    """
    Remove outliers from a numeric column.

    Parameters
    ----------
    df       : input DataFrame
    column   : column to check
    method   : 'IQR' (default, robust) or 'zscore' (assumes normality)
    z_thresh : threshold for z-score method

    Returns
    -------
    Filtered DataFrame + logs how many rows were removed.
    """
    df = df.copy()
    before = len(df)

    if method == "IQR":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = (df[column] >= lower) & (df[column] <= upper)
        logger.info(
            "[IQR] %s: bounds=[%.2f, %.2f], removing %d rows.",
            column, lower, upper, (~mask).sum()
        )

    elif method == "zscore":
        mean = df[column].mean()
        std  = df[column].std()
        z    = (df[column] - mean) / std
        mask = z.abs() <= z_thresh
        logger.info(
            "[Z-score] %s: thresh=%.1f, removing %d rows.",
            column, z_thresh, (~mask).sum()
        )

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'IQR' or 'zscore'.")

    df = df[mask].reset_index(drop=True)
    logger.info("Rows remaining: %d (removed %d).", len(df), before - len(df))
    return df
