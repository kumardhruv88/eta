"""
src/features/build_features.py — sklearn Pipeline for feature engineering.

build_feature_pipeline() returns a fully configured Pipeline object
that can be fit on training data and transform new data consistently.
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import category_encoders as ce


def build_feature_pipeline(
    numeric_features: list[str],
    categorical_low: list[str],
    categorical_high: list[str],
    target_col: str = "eta_seconds",
) -> Pipeline:
    """
    Build a sklearn feature engineering pipeline.

    Parameters
    ----------
    numeric_features   : columns to scale (StandardScaler)
    categorical_low    : low-cardinality categoricals (OrdinalEncoder)
    categorical_high   : high-cardinality categoricals (TargetEncoder)
    target_col         : name of the target column (for TargetEncoder)

    Returns
    -------
    sklearn Pipeline ready for .fit_transform(X, y)
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    low_card_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    # TargetEncoder handles high-cardinality features (e.g., 260 taxi zones)
    # without the dimensionality explosion of one-hot encoding.
    high_card_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", ce.TargetEncoder(smoothing=10)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num",      numeric_transformer,   numeric_features),
        ("cat_low",  low_card_transformer,  categorical_low),
        ("cat_high", high_card_transformer, categorical_high),
    ], remainder="drop")

    return Pipeline([("preprocessor", preprocessor)])
