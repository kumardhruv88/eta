"""
src/models/train.py — Model training with cross-validation.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

logger = logging.getLogger(__name__)


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
    cv_strategy: TimeSeriesSplit | None = None,
    scoring: str = "neg_mean_absolute_error",
) -> dict:
    """
    Train a model with TimeSeriesSplit cross-validation and return CV scores.

    Parameters
    ----------
    X           : feature matrix
    y           : target series
    model       : sklearn-compatible estimator (already wrapped in Pipeline if needed)
    cv_strategy : TimeSeriesSplit instance; defaults to 5-fold
    scoring     : sklearn scoring string

    Returns
    -------
    dict with keys: model, cv_scores, cv_mean, cv_std
    """
    if cv_strategy is None:
        cv_strategy = TimeSeriesSplit(n_splits=5)

    logger.info("Starting cross-validation: %s", model.__class__.__name__)
    scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)
    mae_scores = -scores  # flip sign: sklearn returns negative MAE

    logger.info("CV MAE: %.2f ± %.2f", mae_scores.mean(), mae_scores.std())

    # Final fit on full training data
    model.fit(X, y)
    logger.info("Model trained on full training set.")

    return {
        "model":     model,
        "cv_scores": mae_scores,
        "cv_mean":   mae_scores.mean(),
        "cv_std":    mae_scores.std(),
    }
