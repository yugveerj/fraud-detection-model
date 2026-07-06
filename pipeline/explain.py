"""SHAP explainability (SPEC Section 4).

Explanations are computed on the **base XGBoost** (a fast, exact TreeExplainer). The
calibration step is a monotonic transform of the score, so tree SHAP values explain
the ranking the decision uses. Three case studies (fraud / legit / borderline) drawn
from HOLDOUT become the demo presets (Phase D).

**Interpretation caveat:** many features are anonymized Vesta ``V*`` columns and
``id_*`` fields with no published semantics, so SHAP shows *which* engineered inputs
move a score, not a mechanistic business reason. This limit is stated wherever SHAP
output appears.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CaseStudy:
    kind: str  # "fraud" | "legit" | "borderline"
    index: int
    label: int
    proba: float
    amount: float
    top_factors: list[dict]  # [{feature, value, shap}]


def _base_estimator(pipeline):
    return pipeline.named_steps["est"]


def _transform(pipeline, X):
    prep = pipeline.named_steps["prep"]
    Xt = prep.transform(X)
    names = list(prep.ct_.get_feature_names_out())
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    return np.asarray(Xt), names


def compute_shap(pipeline, X: pd.DataFrame):
    """Return (shap_values [n,f], transformed X [n,f], feature_names, base_value)."""
    import shap

    est = _base_estimator(pipeline)
    Xt, names = _transform(pipeline, X)
    explainer = shap.TreeExplainer(est)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):  # some SHAP versions return per-class lists
        sv = sv[1] if len(sv) > 1 else sv[0]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.ravel(base_value)[-1])
    return np.asarray(sv), Xt, names, float(base_value)


def global_importance(shap_values: np.ndarray, names: list[str], top_k: int = 20) -> pd.DataFrame:
    """Mean absolute SHAP per feature, descending."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)[:top_k]
    return pd.DataFrame({"feature": [names[i] for i in order], "mean_abs_shap": mean_abs[order]})


def select_case_studies(
    proba: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    """Pick representative HOLDOUT rows: a confident fraud, a clear legit, a borderline."""
    proba = np.asarray(proba)
    y = np.asarray(y)
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]
    # Confident true fraud = highest-probability actual fraud.
    fraud = int(fraud_idx[np.argmax(proba[fraud_idx])]) if len(fraud_idx) else 0
    # Clear legit = lowest-probability actual legit.
    legit = int(legit_idx[np.argmin(proba[legit_idx])]) if len(legit_idx) else 0
    # Borderline = closest probability to the operating threshold.
    borderline = int(np.argmin(np.abs(proba - threshold)))
    return {"fraud": fraud, "legit": legit, "borderline": borderline}


def case_study(
    kind: str,
    idx: int,
    shap_values: np.ndarray,
    Xt: np.ndarray,
    names: list[str],
    proba: np.ndarray,
    y: np.ndarray,
    amounts: np.ndarray,
    top_k: int = 6,
) -> CaseStudy:
    contrib = shap_values[idx]
    order = np.argsort(-np.abs(contrib))[:top_k]
    factors = [
        {"feature": names[i], "value": float(Xt[idx, i]), "shap": float(contrib[i])} for i in order
    ]
    return CaseStudy(
        kind=kind,
        index=int(idx),
        label=int(y[idx]),
        proba=float(proba[idx]),
        amount=float(amounts[idx]),
        top_factors=factors,
    )
