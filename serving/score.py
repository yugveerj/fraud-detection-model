"""Shared scoring logic for the serving path (handler + preset precompute).

A single transaction has no entity history, so ``build_features`` yields first-sighting
causal aggregates (count 0) — correct for a novel transaction. SHAP is computed on the
base gradient-boosted model (LightGBM) via a TreeExplainer built once at init (fast for
trees); the calibrated model supplies the displayed probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import features, schema


def build_feature_row(payload: dict) -> pd.DataFrame:
    """Turn an API payload into a one-row featured frame with the full schema."""
    cols = schema.transaction_columns() + [
        c for c in schema.identity_columns() if c != schema.ID_COL
    ]
    row: dict = dict.fromkeys(cols, np.nan)
    row[schema.ID_COL] = 0
    row[schema.TARGET] = 0  # placeholder; never read for scoring
    for key, value in payload.items():
        if key in row and value is not None:
            row[key] = value
    df = pd.DataFrame([row])
    df[schema.TIME_COL] = (
        pd.to_numeric(df[schema.TIME_COL], errors="coerce").fillna(0).astype("int64")
    )
    df[schema.AMT_COL] = pd.to_numeric(df[schema.AMT_COL], errors="coerce")
    # Categorical columns must be object dtype (matching training): an all-NaN column
    # otherwise infers to float64 and breaks the OrdinalEncoder (which expects the
    # string categories it was fit on).
    for col in schema.CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(object)
    return features.build_features(df)


def build_explainer(base_pipeline):
    """Build a TreeExplainer on the base gradient-boosted model once (reused per request)."""
    import shap

    return shap.TreeExplainer(base_pipeline.named_steps["est"])


def top_factors(base_pipeline, explainer, feat: pd.DataFrame, k: int = 5) -> list[dict]:
    prep = base_pipeline.named_steps["prep"]
    Xt = prep.transform(feat)
    names = list(prep.ct_.get_feature_names_out())
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    contrib = np.asarray(sv)[0]
    order = np.argsort(-np.abs(contrib))[:k]
    return [
        {
            "feature": names[i],
            # None (JSON null) when the feature is missing, so the response is valid JSON.
            "value": None if np.isnan(Xt[0, i]) else float(Xt[0, i]),
            "shap": float(contrib[i]),
            "direction": "increases" if contrib[i] > 0 else "decreases",
        }
        for i in order
    ]


def score(bundle: dict, explainer, payload: dict) -> dict:
    """Score one transaction. Returns a JSON-serializable dict."""
    feat = build_feature_row(payload)
    proba = float(bundle["calibrated"].predict_proba(feat)[:, 1][0])
    threshold = float(bundle["threshold"])
    meta = bundle.get("meta", {})
    return {
        "fraud_probability": round(proba, 6),
        "decision": "review" if proba >= threshold else "approve",
        "threshold": round(threshold, 6),
        "top_factors": top_factors(bundle["base"], explainer, feat),
        "model_version": meta.get("version"),
        "run_id": meta.get("run_id"),
        "calibration": meta.get("calibration", "isotonic"),
        "provenance": meta.get("provenance", "synthetic"),
    }
