"""
src/pipeline/full_pipeline.py
Fast end-to-end ML pipeline:
  raw parquet -> filtered interim -> features -> train -> evaluate
Column-pruned load, XGBoost hist, single command to reproduce everything.

Usage:
    python src/pipeline/full_pipeline.py
    python src/pipeline/full_pipeline.py --sample 200000
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# Project root = two levels up from this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _log(msg: str, t0: float | None = None) -> float:
    elapsed = f"  [{time.time()-t0:.1f}s]" if t0 else ""
    print(f"[pipeline]{elapsed}  {msg}", flush=True)
    return time.time()


def _save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────
# Step 1 – Ingest (column-pruned for speed)
# ─────────────────────────────────────────────
def step_ingest(sample_size: int) -> pd.DataFrame:
    t0 = _log("Step 1/5 — Ingest raw parquet (column-pruned)")
    raw_path = ROOT / config.RAW_DATA_PATH / config.RAW_FILE_NAME
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw file not found: {raw_path}\n"
            "Either download it (run Notebook 01) or place the parquet in data/raw/"
        )

    # Only load the 9 columns we actually need — reduces I/O from 500 MB to ~60 MB
    needed_cols = config.RAW_COLUMNS_NEEDED
    df = pd.read_parquet(raw_path, columns=needed_cols)
    _log(f"  Full dataset: {df.shape[0]:,} rows  (loaded {len(needed_cols)} columns)", t0)

    # Deterministic sample
    df = df.sample(n=min(sample_size, len(df)), random_state=config.RANDOM_STATE)
    _log(f"  Sampled: {len(df):,} rows", t0)

    # Parse datetimes
    for col in ["request_datetime", "on_scene_datetime"]:
        if df[col].dtype == object:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


# ─────────────────────────────────────────────
# Step 2 – Preprocess + filter
# ─────────────────────────────────────────────
def step_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    t0 = _log("Step 2/5 — Preprocess and filter")
    n_start = len(df)

    df["eta_seconds"] = (
        df["on_scene_datetime"] - df["request_datetime"]
    ).dt.total_seconds()

    # Sequential quality filters
    df = df.dropna(subset=["on_scene_datetime"])
    df = df[df["eta_seconds"] > 0]
    df = df[df["eta_seconds"].between(config.ETA_MIN_SECONDS, config.ETA_MAX_SECONDS)]

    # Ghost trips
    if "trip_time" in df.columns:
        df = df[~((df["trip_miles"] == 0) & (df["trip_time"] == 0))]

    n_kept = len(df)
    _log(f"  Kept: {n_kept:,} / {n_start:,} ({100*n_kept/n_start:.1f}%)", t0)

    # Save interim
    interim_dir = ROOT / config.INTERIM_DATA_PATH
    interim_dir.mkdir(parents=True, exist_ok=True)
    out = interim_dir / config.INTERIM_FILTERED_NAME
    df.to_parquet(out, index=False)
    _log(f"  Saved -> {out}", t0)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# Step 3 – Feature engineering
# ─────────────────────────────────────────────
def step_features(df: pd.DataFrame) -> tuple:
    t0 = _log("Step 3/5 — Feature engineering")
    try:
        from category_encoders import TargetEncoder
    except ImportError:
        raise ImportError("pip install category_encoders")

    # Time features
    df["pickup_hour"]        = df["request_datetime"].dt.hour
    df["pickup_day_of_week"] = df["request_datetime"].dt.dayofweek
    df["is_weekend"]         = (df["pickup_day_of_week"] >= 5).astype(int)
    df["is_rush_hour"]       = df["pickup_hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_night"]           = df["pickup_hour"].isin([22, 23, 0, 1, 2, 3, 4]).astype(int)

    # Operator
    df["operator"] = df["hvfhs_license_num"].map({"HV0003": 0, "HV0005": 1}).fillna(-1).astype(int)

    # Flags
    for col in ["shared_request_flag", "wav_request_flag"]:
        df[col] = (df.get(col, pd.Series("N", index=df.index)) == "Y").astype(int)

    # Log target
    df["log_eta"] = np.log1p(df["eta_seconds"])

    FEATURES = config.FEATURE_NAMES
    TARGET   = "log_eta"

    # Chronological split (no shuffle — prevents temporal leakage)
    df_sorted  = df.sort_values("request_datetime").reset_index(drop=True)
    split_idx  = int(len(df_sorted) * 0.80)
    X_train    = df_sorted[FEATURES].iloc[:split_idx].copy()
    X_test     = df_sorted[FEATURES].iloc[split_idx:].copy()
    y_train    = df_sorted[TARGET].iloc[:split_idx].copy()
    y_test     = df_sorted[TARGET].iloc[split_idx:].copy()

    # Preprocessor
    numeric_features = ["pickup_hour", "pickup_day_of_week"]
    binary_features  = ["is_weekend", "is_rush_hour", "is_night",
                        "operator", "shared_request_flag", "wav_request_flag"]
    zone_features    = ["PULocationID", "DOLocationID"]

    preprocessor = ColumnTransformer(transformers=[
        ("num",  StandardScaler(),          numeric_features),
        ("bin",  "passthrough",             binary_features),
        ("zone", TargetEncoder(smoothing=10), zone_features),
    ], remainder="drop")

    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc  = preprocessor.transform(X_test)

    # Save processed arrays
    proc_dir = ROOT / config.PROCESSED_DATA_PATH
    proc_dir.mkdir(parents=True, exist_ok=True)
    np.save(proc_dir / "X_train.npy", X_train_proc)
    np.save(proc_dir / "X_test.npy",  X_test_proc)
    np.save(proc_dir / "y_train.npy", y_train.values)
    np.save(proc_dir / "y_test.npy",  y_test.values)

    # Save preprocessor
    models_dir = ROOT / config.SAVED_MODELS_PATH
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")

    _log(f"  Train: {X_train_proc.shape}  Test: {X_test_proc.shape}", t0)
    return X_train_proc, X_test_proc, y_train.values, y_test.values


# ─────────────────────────────────────────────
# Step 4 – Train XGBoost
# ─────────────────────────────────────────────
def step_train(X_train, y_train) -> XGBRegressor:
    t0 = _log("Step 4/5 — Train XGBoost (tree_method=hist)")
    params = {
        **config.XGB_DEFAULTS,
        "n_estimators": 300,         # fast default; Notebook 05 tunes further
        "tree_method":  "hist",      # fastest CPU training
        "verbosity":    0,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    models_dir = ROOT / config.SAVED_MODELS_PATH
    joblib.dump(model, models_dir / config.MODEL_ARTIFACT_NAME)
    _log(f"  Saved -> {models_dir / config.MODEL_ARTIFACT_NAME}", t0)
    return model


# ─────────────────────────────────────────────
# Step 5 – Evaluate
# ─────────────────────────────────────────────
def step_evaluate(model, X_test, y_test) -> dict:
    t0 = _log("Step 5/5 — Evaluate on holdout test set")
    y_pred_log = model.predict(X_test)
    y_pred     = np.expm1(y_pred_log)
    y_true     = np.expm1(y_test)
    abs_err    = np.abs(y_true - y_pred)

    metrics = {
        "MAE_seconds":   float(mean_absolute_error(y_true, y_pred)),
        "RMSE_seconds":  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_pct":      float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2":            float(r2_score(y_true, y_pred)),
        "p90_error_s":   float(np.percentile(abs_err, 90)),
        "p95_error_s":   float(np.percentile(abs_err, 95)),
        "n_test":        int(len(y_true)),
        "feature_names": config.FEATURE_NAMES,
        "shap_importance": [],  # populated by Notebook 06
    }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _save_json(metrics, reports_dir / "evaluation_report.json")

    print()
    print("=" * 50)
    print("  PIPELINE RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<22} {v:.3f}")
    print("=" * 50)
    _log("  Pipeline complete!", t0)
    return metrics


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Driver ETA full pipeline")
    parser.add_argument("--sample", type=int, default=config.SAMPLE_SIZE,
                        help=f"Sample size (default: {config.SAMPLE_SIZE})")
    args = parser.parse_args()

    total_t0 = time.time()
    print(f"\n{'='*55}")
    print(f"  Driver Pickup ETA Prediction -- Full Pipeline")
    print(f"  Sample size: {args.sample:,}  |  random_state={config.RANDOM_STATE}")
    print(f"{'='*55}\n")

    df              = step_ingest(args.sample)
    df              = step_preprocess(df)
    X_tr, X_te, y_tr, y_te = step_features(df)
    model           = step_train(X_tr, y_tr)
    metrics         = step_evaluate(model, X_te, y_te)

    total = time.time() - total_t0
    print(f"\nTotal wall time: {total:.1f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
