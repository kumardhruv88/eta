"""
src/pipeline/full_pipeline.py — Orchestrates the full end-to-end pipeline.
Run this script to reproduce all steps from raw data → saved model.
"""

import logging
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from xgboost import XGBRegressor

import config
from src.data.ingest import download_raw_data, load_raw_parquet
from src.data.preprocess import clean_data, construct_target, remove_outliers
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.utils.helpers import setup_logging, ensure_dirs

setup_logging()
logger = logging.getLogger(__name__)


def run():
    ensure_dirs(
        config.RAW_DATA_PATH,
        config.INTERIM_DATA_PATH,
        config.PROCESSED_DATA_PATH,
        config.SAVED_MODELS_PATH,
    )

    # 1. Download
    raw_path = download_raw_data(
        config.HVFHV_URL,
        Path(config.RAW_DATA_PATH) / config.RAW_FILE_NAME,
    )

    # 2. Load + sample
    df = load_raw_parquet(raw_path, sample_size=config.SAMPLE_SIZE, random_state=config.RANDOM_STATE)

    # 3. Clean + construct target
    df = clean_data(df)
    df = construct_target(df)
    df = remove_outliers(df, "trip_miles", method="IQR")

    # 4. Feature engineering (inline for pipeline script)
    df["pickup_hour"]        = df["request_datetime"].dt.hour
    df["pickup_day_of_week"] = df["request_datetime"].dt.dayofweek
    df["is_weekend"]         = df["pickup_day_of_week"].isin([5, 6]).astype(int)
    df["is_rush_hour"]       = df["pickup_hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    FEATURES = [
        "trip_miles", "pickup_hour", "pickup_day_of_week",
        "is_weekend", "is_rush_hour", "PULocationID", "DOLocationID",
    ]
    X = df[FEATURES]
    y = df[config.TARGET_COLUMN]

    # Sort by time for temporal split
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    split = int(len(X) * 0.70)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # 5. Train XGBoost
    model = XGBRegressor(**config.XGB_DEFAULTS)
    result = train_model(X_train, y_train, model, cv_strategy=TimeSeriesSplit(n_splits=config.CV_N_SPLITS))

    # 6. Evaluate
    metrics = evaluate_model(result["model"], X_test, y_test)
    logger.info("Test metrics: %s", metrics)

    # 7. Save
    model_path = Path(config.SAVED_MODELS_PATH) / config.MODEL_ARTIFACT_NAME
    joblib.dump(result["model"], model_path)
    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    run()
