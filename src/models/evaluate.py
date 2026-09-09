"""
src/models/evaluate.py — Compute regression evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a trained model on hold-out test data.

    Returns
    -------
    dict with MAE, RMSE, MAPE, R²
    """
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # MAPE — guard against division by zero
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

    return {
        "MAE":  round(mae,  2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "R2":   round(r2,   4),
    }
