"""
api/main.py — FastAPI serving endpoint for Driver Pickup ETA Prediction.

Endpoints
---------
GET  /health        liveness check
GET  /model-info    feature names, version, training metadata
GET  /metrics       evaluation metrics from reports/evaluation_report.json
GET  /zones         list of valid TLC zone IDs for frontend dropdowns
POST /predict       predict ETA given trip features
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, field_validator

import config

# ─────────────────────────────────────────────
# Startup: load model artifacts once
# ─────────────────────────────────────────────
ARTIFACTS: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and preprocessor at startup."""
    model_path = Path(config.SAVED_MODELS_PATH) / config.MODEL_ARTIFACT_NAME
    prep_path  = Path(config.SAVED_MODELS_PATH) / "preprocessor.joblib"
    report_path = Path("reports") / "evaluation_report.json"

    ARTIFACTS["model"]       = joblib.load(model_path) if model_path.exists() else None
    ARTIFACTS["preprocessor"]= joblib.load(prep_path)  if prep_path.exists()  else None
    ARTIFACTS["model_loaded"]= ARTIFACTS["model"] is not None
    ARTIFACTS["loaded_at"]   = datetime.utcnow().isoformat()
    ARTIFACTS["metrics"]     = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists() else {}
    )
    print(f"[API] Model loaded: {ARTIFACTS['model_loaded']}")
    yield

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Driver Pickup ETA Prediction API",
    description="Predicts driver pickup ETA in seconds using XGBoost + NYC TLC data.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend at /ui
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class TripRequest(BaseModel):
    pickup_hour:        int   = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    pickup_day_of_week: int   = Field(..., ge=0, le=6,  description="Day of week (0=Mon)")
    is_weekend:         int   = Field(..., ge=0, le=1,  description="1 if Sat/Sun")
    is_rush_hour:       int   = Field(..., ge=0, le=1,  description="1 if 7-9am or 5-7pm")
    is_night:           int   = Field(0,  ge=0, le=1,  description="1 if 10pm-5am")
    PULocationID:       int   = Field(..., ge=1, le=265, description="TLC pickup zone")
    DOLocationID:       int   = Field(..., ge=1, le=265, description="TLC dropoff zone")
    operator:           int   = Field(0,  ge=0, le=1,  description="0=Uber, 1=Lyft")
    shared_request_flag:int   = Field(0,  ge=0, le=1,  description="1 if pooled ride")
    wav_request_flag:   int   = Field(0,  ge=0, le=1,  description="1 if wheelchair")

    model_config = {"json_schema_extra": {
        "example": {
            "pickup_hour": 8, "pickup_day_of_week": 1, "is_weekend": 0,
            "is_rush_hour": 1, "is_night": 0, "PULocationID": 161,
            "DOLocationID": 236, "operator": 0, "shared_request_flag": 0,
            "wav_request_flag": 0
        }
    }}


class ETAResponse(BaseModel):
    predicted_eta_seconds: float
    predicted_eta_minutes: float
    eta_range_low_s:       float
    eta_range_high_s:      float
    confidence:            str
    interpretation:        str


# ─────────────────────────────────────────────
# Helper: raw feature vector -> model input
# ─────────────────────────────────────────────
FEATURE_ORDER = config.FEATURE_NAMES  # must match preprocessor column order


def _build_feature_df(req: TripRequest) -> pd.DataFrame:
    row = {
        "pickup_hour":        req.pickup_hour,
        "pickup_day_of_week": req.pickup_day_of_week,
        "is_weekend":         req.is_weekend,
        "is_rush_hour":       req.is_rush_hour,
        "is_night":           req.is_night,
        "PULocationID":       req.PULocationID,
        "DOLocationID":       req.DOLocationID,
        "operator":           req.operator,
        "shared_request_flag":req.shared_request_flag,
        "wav_request_flag":   req.wav_request_flag,
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    return {"message": "Driver ETA API", "docs": "/docs", "ui": "/ui/"}


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "model_loaded": ARTIFACTS.get("model_loaded", False),
        "loaded_at": ARTIFACTS.get("loaded_at"),
        "version": "2.0.0",
    }


@app.get("/model-info", tags=["System"])
def model_info():
    return {
        "feature_names": config.FEATURE_NAMES,
        "n_features": len(config.FEATURE_NAMES),
        "target": "log1p(eta_seconds) -> expm1 for seconds",
        "model_type": "XGBoostRegressor",
        "model_loaded": ARTIFACTS.get("model_loaded", False),
        "shap_importance": ARTIFACTS.get("metrics", {}).get("shap_importance", []),
    }


@app.get("/metrics", tags=["System"])
def get_metrics():
    m = ARTIFACTS.get("metrics", {})
    if not m:
        raise HTTPException(
            status_code=404,
            detail="Evaluation report not found. Run the full pipeline or Notebook 06 first."
        )
    return m


@app.get("/zones", tags=["Reference"])
def get_zones():
    """Return all valid TLC zone IDs (1-265) for frontend dropdowns."""
    # Common named zones for better UX
    named = {
        1: "Newark Airport", 4: "Alphabet City", 12: "Battery Park",
        13: "Battery Park City", 24: "Bloomingdale", 41: "Central Park",
        42: "Central Park (W)", 43: "Charlotte", 45: "Chinatown",
        48: "Clinton East", 50: "Clinton Hill", 68: "East Chelsea",
        79: "East Village", 87: "Financial District N", 88: "Financial District S",
        100: "Garment District", 107: "Hamilton Heights", 113: "Hell's Kitchen N",
        114: "Hell's Kitchen S", 125: "Hudson Sq", 127: "Highbridge Park",
        128: "Hollis Hills", 132: "JFK Airport", 138: "LaGuardia Airport",
        140: "Lenox Hill E", 141: "Lenox Hill W", 143: "Lincoln Sq E",
        144: "Lincoln Sq W", 148: "Little Italy/NoLiTa", 151: "Manhattan Valley",
        152: "Manhattanville", 153: "Marble Hill", 158: "Meatpacking/West Village W",
        161: "Midtown Center", 162: "Midtown East", 163: "Midtown North",
        164: "Midtown South", 166: "Morningside Heights", 186: "Penn Station/Madison Sq W",
        230: "Times Sq/Theatre District", 231: "TriBeCa/Civic Center",
        232: "Two Bridges/Seaport", 233: "UN/Turtle Bay S", 234: "Union Sq",
        236: "Upper East Side N", 237: "Upper East Side S",
        238: "Upper West Side N", 239: "Upper West Side S",
        243: "Washington Heights N", 244: "Washington Heights S",
        246: "West Chelsea/Hudson Yards", 249: "West Village",
        261: "World Trade Center", 262: "Yorkville E", 263: "Yorkville W",
    }
    zones = [
        {"id": i, "name": named.get(i, f"Zone {i}")}
        for i in range(1, 266)
    ]
    return {"zones": zones, "count": len(zones)}


@app.post("/predict", response_model=ETAResponse, tags=["Prediction"])
def predict(request: TripRequest):
    if not ARTIFACTS.get("model_loaded"):
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the full pipeline first: python src/pipeline/full_pipeline.py"
        )

    try:
        preprocessor = ARTIFACTS["preprocessor"]
        model        = ARTIFACTS["model"]

        X_df    = _build_feature_df(request)
        X_proc  = preprocessor.transform(X_df) if preprocessor else X_df.values
        log_pred = model.predict(X_proc)[0]
        eta_sec  = float(np.expm1(log_pred))

        # Approximate ±1 std confidence interval (from training residual std)
        residual_std = ARTIFACTS.get("metrics", {}).get(
            "residual_stats", {}).get("std", eta_sec * 0.25)
        low  = max(0.0, eta_sec - residual_std)
        high = eta_sec + residual_std

        if eta_sec < 180:
            confidence = "High"
            interp = "Driver is very close — expect quick pickup."
        elif eta_sec < 420:
            confidence = "Medium"
            interp = "Typical urban pickup time."
        else:
            confidence = "Low"
            interp = "Long ETA — possible high demand or far driver."

        return ETAResponse(
            predicted_eta_seconds=round(eta_sec, 1),
            predicted_eta_minutes=round(eta_sec / 60, 2),
            eta_range_low_s=round(low, 1),
            eta_range_high_s=round(high, 1),
            confidence=confidence,
            interpretation=interp,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
