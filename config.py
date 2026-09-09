"""
config.py — Central configuration for Driver Pickup ETA Prediction project.
All paths, constants, and hyperparameter defaults live here.
Import this in every notebook and src file to keep things DRY.
"""

# ─────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────
RAW_DATA_PATH       = "data/raw/"
INTERIM_DATA_PATH   = "data/interim/"
PROCESSED_DATA_PATH = "data/processed/"
SAVED_MODELS_PATH   = "saved_models/"
REPORTS_PATH        = "reports/figures/"

# ─────────────────────────────────────────────
# DATA SOURCE
# ─────────────────────────────────────────────
# NYC TLC High Volume For-Hire Vehicle (HVFHV) — Jan 2024
# HV0003 = Uber, HV0005 = Lyft
HVFHV_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "fhvhv_tripdata_2024-01.parquet"
)
RAW_FILE_NAME       = "fhvhv_2024_01.parquet"
INTERIM_SAMPLE_NAME = "sample.parquet"
INTERIM_FILTERED_NAME = "sample_filtered.parquet"
PROCESSED_FILE_NAME = "features.parquet"

# ─────────────────────────────────────────────
# SAMPLING
# ─────────────────────────────────────────────
# Full dataset ≈ 20M rows. We cap at 500k for local development.
# In production this would run on Spark/BigQuery.
SAMPLE_SIZE  = 500_000
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# TARGET VARIABLE
# ─────────────────────────────────────────────
TARGET_COLUMN = "eta_seconds"

# Bounds for valid ETA (seconds).
# < 30 s  → driver was already on-site (GPS error or pre-positioned)
# > 3600 s → over 1 hour, almost certainly a data error or app crash
ETA_MIN_SECONDS = 30
ETA_MAX_SECONDS = 3600

# ─────────────────────────────────────────────
# FEATURE LISTS
# ─────────────────────────────────────────────
NUMERIC_FEATURES = [
    "trip_miles",
    "passenger_count",  # not in HVFHV, kept for schema parity
    "pickup_hour",
    "pickup_day_of_week",
    "is_weekend",
    "is_rush_hour",
]

CATEGORICAL_FEATURES = [
    "hvfhs_license_num",   # Uber vs Lyft
    "PULocationID",        # pickup zone  (~260 unique)
    "DOLocationID",        # dropoff zone (~260 unique)
]

# Columns that must NEVER be used as features (post-trip / target-leaking)
LEAKY_COLUMNS = [
    "trip_time",          # total trip seconds — only known after drop-off
    "base_passenger_fare",
    "driver_pay",
    "tips",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "dropoff_datetime",
    "on_scene_datetime",  # used only to construct target, then dropped
]

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
TEST_SIZE    = 0.30
CV_N_SPLITS  = 5

# ─────────────────────────────────────────────
# XGBOOST DEFAULTS (overridden by Optuna)
# ─────────────────────────────────────────────
XGB_DEFAULTS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     RANDOM_STATE,
    "tree_method":      "hist",   # fast CPU training
    "eval_metric":      "mae",
}

# ─────────────────────────────────────────────
# OPTUNA TUNING
# ─────────────────────────────────────────────
OPTUNA_N_TRIALS = 100
OPTUNA_DIRECTION = "minimize"   # minimise MAE

# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_ARTIFACT_NAME = "best_xgb_model.joblib"
