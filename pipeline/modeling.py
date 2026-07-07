"""Model factory + calibration (SPEC Section 3).

Every model is a scikit-learn ``Pipeline(preprocessor, estimator)`` so training,
prediction, and calibration share one interface. Class imbalance is handled with
cost-sensitive weights (``scale_pos_weight`` / ``class_weight='balanced'``) — never
SMOTE (see docs/decisions.md D-001). The registered production model is the
calibrated XGBoost.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from pipeline import encoders

# NOTE: lightgbm is imported lazily inside make_model, not at module level. The serving
# image (xgboost only) unpickles a bundle that references this module (for _FeatureBinder);
# a top-level lightgbm import would make that fail with ModuleNotFoundError.

SEED = 42
MODEL_NAMES = ("logreg", "xgboost", "lightgbm")

# Modest, hand-set hyperparameters (tuned on VALIDATION only, per spec). "smoke"
# variants are tiny so CI validates the plumbing in seconds.
_PARAMS = {
    "xgboost": {
        "full": {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
        },
        "smoke": {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.2},
    },
    "lightgbm": {
        "full": {
            "n_estimators": 400,
            "num_leaves": 64,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
        },
        "smoke": {"n_estimators": 40, "num_leaves": 15, "learning_rate": 0.2},
    },
    "logreg": {
        "full": {"C": 1.0, "max_iter": 2000},
        "smoke": {"C": 1.0, "max_iter": 300},
    },
}


def scale_pos_weight(y) -> float:
    """neg/pos ratio for cost-sensitive GBM training."""
    y = np.asarray(y)
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    return neg / pos if pos > 0 else 1.0


def make_model(name: str, y_train=None, profile: str = "full", seed: int = SEED):
    """Build an unfitted ``Pipeline`` for ``name`` ('logreg'|'xgboost'|'lightgbm').

    ``y_train`` (if given) sets the cost-sensitive weight for the GBMs.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model: {name!r}")

    params = _PARAMS[name][profile]
    spw = scale_pos_weight(y_train) if y_train is not None else 1.0

    if name == "logreg":
        est = LogisticRegression(class_weight="balanced", solver="lbfgs", n_jobs=-1, **params)
        kind = "linear"
    elif name == "xgboost":
        est = XGBClassifier(
            tree_method="hist",
            eval_metric="aucpr",
            scale_pos_weight=spw,
            random_state=seed,
            n_jobs=-1,
            **params,
        )
        kind = "tree"
    else:  # lightgbm
        from lightgbm import LGBMClassifier  # lazy: keep it out of the serving image

        est = LGBMClassifier(
            scale_pos_weight=spw,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
        kind = "tree"

    # Preprocessor columns are bound at fit time by _FeatureBinder.
    return Pipeline([("prep", _FeatureBinder(kind)), ("est", est)])


class _FeatureBinder(BaseEstimator, TransformerMixin):
    """Column-selecting preprocessor that resolves feature columns from the fitted
    DataFrame, then delegates to the tree/linear ColumnTransformer. Keeps model
    construction independent of the exact column list (which depends on the data)."""

    def __init__(self, kind: str = "tree"):
        self.kind = kind

    def fit(self, X, y=None):
        numeric, categorical = encoders.model_feature_columns(X)
        self.ct_ = encoders.build_preprocessor(numeric, categorical, self.kind)
        self.ct_.fit(X, y)
        return self

    def transform(self, X):
        return self.ct_.transform(X)


def positive_proba(model, X) -> np.ndarray:
    """P(fraud) from a fitted classifier/pipeline."""
    return model.predict_proba(X)[:, 1]


def calibrate(fitted_model, X_val, y_val, method: str = "isotonic"):
    """Calibrate a PREFIT model on the validation set (isotonic default, or 'sigmoid').

    Uses a frozen base estimator so only the calibration map is fit on val — the base
    model is never refit, preserving the temporal train/val boundary.
    """
    try:
        from sklearn.frozen import FrozenEstimator

        cal = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
    except ImportError:  # older sklearn
        cal = CalibratedClassifierCV(fitted_model, method=method, cv="prefit")
    cal.fit(X_val, y_val)
    return cal
