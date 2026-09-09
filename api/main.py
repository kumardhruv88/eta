"""
api/main.py — FastAPI serving endpoint for Driver Pickup ETA Prediction.

Endpoints
---------
GET  /health       — liveness check
POST /predict      — predict ETA given trip features
"""

import os
import sys
from pathlib import Path

# Allow imports from project root when run from /api
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import config
from src.models.predict import predict_eta

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="Driver Pickup ETA Prediction API",
    description=(
        "Predicts how many seconds a driver will take to reach the pickup location, "
        "given trip request features."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Model loading (done once at startup)
# ─────────────────────────────────────────────
MODEL = None
MODEL_PATH = Path(config.SAVED_MODELS_PATH) / config.MODEL_ARTIFACT_NAME


@app.on_event("startup")
def load_model():
    global MODEL
    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)
    else:
        # Non-fatal: API can start but /predict will return 503
        MODEL = None


# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class TripRequest(BaseModel):
    pickup_hour: int        = Field(..., ge=0, le=23, description="Hour of pickup (0-23)")
    pickup_day_of_week: int = Field(..., ge=0, le=6,  description="Day of week (0=Mon, 6=Sun)")
    is_weekend: int         = Field(..., ge=0, le=1,  description="1 if Sat/Sun, else 0")
    is_rush_hour: int       = Field(..., ge=0, le=1,  description="1 if 7-9am or 5-7pm, else 0")
    trip_miles: float       = Field(..., gt=0, description="Estimated trip distance in miles")
    PULocationID: int       = Field(..., ge=1, le=265, description="TLC pickup zone ID")
    DOLocationID: int       = Field(..., ge=1, le=265, description="TLC dropoff zone ID")
    hvfhs_license_num: str  = Field("HV0003", description="HV0003=Uber, HV0005=Lyft")

    @field_validator("hvfhs_license_num")
    @classmethod
    def validate_operator(cls, v):
        allowed = {"HV0003", "HV0005"}
        if v not in allowed:
            raise ValueError(f"hvfhs_license_num must be one of {allowed}")
        return v


class ETAResponse(BaseModel):
    predicted_eta_seconds: float
    predicted_eta_minutes: float
    confidence_note: str


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=ETAResponse, tags=["Prediction"])
def predict(request: TripRequest):
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model not loaded. Run the full pipeline first to generate "
                f"{MODEL_PATH}."
            ),
        )

    try:
        result = predict_eta(MODEL, request.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ETAResponse(**result)
