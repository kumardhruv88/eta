"""
src/models/predict.py — Single-row prediction for API serving.
"""

import numpy as np
import pandas as pd


def predict_eta(model, input_dict: dict) -> dict:
    """
    Run a single prediction from a raw input dictionary.

    Parameters
    ----------
    model      : trained sklearn Pipeline (loaded from joblib)
    input_dict : dict matching the model's expected feature schema

    Returns
    -------
    dict with predicted_eta_seconds, predicted_eta_minutes, confidence_note
    """
    X = pd.DataFrame([input_dict])
    eta_seconds = float(model.predict(X)[0])

    # Clip to valid range (model could extrapolate outside training bounds)
    eta_seconds = max(30.0, min(eta_seconds, 3600.0))

    eta_minutes = round(eta_seconds / 60, 1)

    # Confidence note based on input plausibility
    if input_dict.get("trip_miles", 1) > 20:
        note = "Long trip detected — ETA estimate may have higher uncertainty."
    elif input_dict.get("is_rush_hour", 0) == 1:
        note = "Rush hour — ETA may be understated due to traffic variability."
    else:
        note = "Prediction within normal operating range."

    return {
        "predicted_eta_seconds": round(eta_seconds, 1),
        "predicted_eta_minutes": eta_minutes,
        "confidence_note": note,
    }
