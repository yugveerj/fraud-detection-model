"""Leakage-safe preprocessing (SPEC Section 3).

All encoders are **fit on TRAIN only** and applied to val/holdout — no statistic
crosses the temporal boundary. Two flavours, both as scikit-learn transformers so
models compose as ``Pipeline(preprocessor, estimator)`` with a uniform interface:

* **tree**: ordinal-encode categoricals (unseen -> -1, missing -> -2), pass numerics
  through with NaN intact (XGBoost/LightGBM handle missing natively).
* **linear**: median-impute + standardize numerics; one-hot the categoricals with a
  rare-category floor so high-cardinality columns stay bounded and target-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from pipeline import features, schema

# Categoricals fed to the models (kept modest; high-cardinality ids stay numeric).
MODEL_CATEGORICALS = [
    schema.PRODUCT_COL,
    "card4",
    "card6",
    *schema.EMAIL_COLS,
    *schema.M_COLS,
    "DeviceType",
]

# Non-feature columns never fed to a model.
_EXCLUDE = {schema.ID_COL, schema.TARGET, schema.TIME_COL}


def model_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) present in ``df`` for modelling.

    Numerics = engineered base + causal aggregates + raw numeric schema columns.
    Categoricals = the modest MODEL_CATEGORICALS list (those present).
    """
    categorical = [c for c in MODEL_CATEGORICALS if c in df.columns]
    cat_set = set(categorical)
    numeric = [
        c
        for c in df.columns
        if c not in _EXCLUDE and c not in cat_set and pd.api.types.is_numeric_dtype(df[c])
    ]
    return numeric, categorical


def build_preprocessor(numeric: list[str], categorical: list[str], kind: str) -> ColumnTransformer:
    if kind == "tree":
        cat = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
        return ColumnTransformer(
            [("num", "passthrough", numeric), ("cat", cat, categorical)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
    if kind == "linear":
        num_pipe = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value="__NA__")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=0.02,
                        max_categories=20,
                        sparse_output=True,
                    ),
                ),
            ]
        )
        return ColumnTransformer(
            [("num", num_pipe, numeric), ("cat", cat_pipe, categorical)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
    raise ValueError(f"unknown preprocessor kind: {kind!r}")


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build modelling features (base + causal) from a raw merged frame."""
    return features.build_features(df)


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Split a featured frame into (X with all columns, y)."""
    y = df[schema.TARGET].to_numpy().astype(int)
    return df, y
