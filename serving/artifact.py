"""Build the serving bundle + demo presets (SPEC Section 5).

Trains the base XGBoost, calibrates (isotonic) on VAL, freezes the operating point,
registers the calibrated model in MLflow (for the version + run id ``/healthz``
reports), and writes:

* ``artifacts/serving/bundle.joblib`` — {calibrated, base, threshold, meta} loaded by
  the Lambda handler at init (gitignored; rebuilt at image-build time).
* ``artifacts/serving/presets.json`` — three representative demo transactions
  (fraud / legit / borderline) with rounded values and their scored output, committed
  for the static demo page.

    uv run python -m serving.artifact                 # real data if present, else synthetic
    uv run python -m serving.artifact --source synthetic
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from pipeline import data as data_mod
from pipeline import decisions, encoders, experiments, explain, modeling, schema
from serving import score as score_mod

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVING_DIR = REPO_ROOT / "artifacts" / "serving"
BUNDLE_PATH = SERVING_DIR / "bundle.joblib"
PRESETS_PATH = SERVING_DIR / "presets.json"
META_PATH = SERVING_DIR / "meta.json"

# Human-readable fields surfaced on the demo card (the rest ride along in the payload).
DISPLAY_FIELDS = ["TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain", "DeviceType"]


def build(source: str = "auto", profile: str = "full", review_cost: float = 25.0) -> dict:
    ds = data_mod.load_dataset(source)
    feat = encoders.feature_frame(ds.frame)
    sp = data_mod.temporal_split(feat, provenance=ds.provenance)
    y_tr = sp.train[schema.TARGET].to_numpy()
    y_val = sp.val[schema.TARGET].to_numpy()

    base = modeling.make_model("xgboost", y_train=y_tr, profile=profile).fit(sp.train, y_tr)
    cal = modeling.calibrate(base, sp.val, y_val, method="isotonic")
    op = decisions.optimize_operating_point(
        y_val, cal.predict_proba(sp.val)[:, 1], sp.val[schema.AMT_COL].to_numpy(), review_cost
    )

    mlflow.set_tracking_uri(experiments.MLFLOW_URI)
    mlflow.set_experiment(experiments.EXPERIMENT_NAME)
    reg = experiments._register(cal, "isotonic", ds.provenance, profile)

    numeric, categorical = encoders.model_feature_columns(sp.train)
    meta = {
        "version": reg.get("version"),
        "run_id": reg.get("run_id"),
        "model_uri": reg.get("model_uri"),
        "calibration": "isotonic",
        "provenance": ds.provenance,
        "profile": profile,
        "threshold": op["threshold"],
        "review_cost": review_cost,
        "feature_columns": {"numeric": numeric, "categorical": categorical},
    }
    bundle = {
        "calibrated": cal,
        "base": base,
        "threshold": op["threshold"],
        "review_cost": review_cost,
        "meta": meta,
    }
    SERVING_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    presets = _build_presets(bundle, sp.holdout, op["threshold"], ds.provenance)
    # allow_nan=False: fail loudly rather than emit `NaN` (invalid JSON) if any NaN leaks.
    PRESETS_PATH.write_text(json.dumps(presets, indent=2, allow_nan=False), encoding="utf-8")
    return meta


def _build_presets(bundle, holdout, threshold, provenance) -> dict:
    """Representative fraud / legit / borderline transactions (rounded), each scored."""
    explainer = score_mod.build_explainer(bundle["base"])
    p_hold = bundle["calibrated"].predict_proba(holdout)[:, 1]
    y_hold = holdout[schema.TARGET].to_numpy()
    sel = explain.select_case_studies(p_hold, y_hold, threshold)

    presets = []
    for kind, idx in sel.items():
        row = holdout.iloc[idx]
        payload = _representative_payload(row)
        scored = score_mod.score(bundle, explainer, payload)
        presets.append(
            {
                "name": kind,
                "label": {
                    "fraud": "Likely fraud",
                    "legit": "Clearly legitimate",
                    "borderline": "Borderline",
                }[kind],
                "display": {k: payload.get(k) for k in DISPLAY_FIELDS},
                "payload": payload,
                "expected": scored,
            }
        )
    return {
        "provenance": provenance,
        "threshold": round(float(threshold), 6),
        "disclaimer": (
            "Demonstration system on a public research dataset — not a production fraud decision."
        ),
        "presets": presets,
    }


def _representative_payload(row) -> dict:
    """Non-null transaction fields with a rounded amount (representative, not a raw row)."""
    payload: dict = {}
    for col, val in row.items():
        if col in (schema.ID_COL, schema.TARGET):
            continue
        if pd.isna(val):  # catches float / float32 / None NaN so no NaN reaches the JSON
            continue
        payload[col] = _json_safe(val)
    payload[schema.AMT_COL] = round(float(row[schema.AMT_COL]), 2)
    payload[schema.TIME_COL] = int(row[schema.TIME_COL])
    return payload


def _json_safe(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return round(float(val), 4)
    return val


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the serving bundle + presets")
    parser.add_argument("--source", default="auto", choices=["auto", "real", "synthetic"])
    parser.add_argument("--profile", default="full", choices=["full", "smoke"])
    parser.add_argument("--review-cost", type=float, default=25.0)
    args = parser.parse_args(argv)
    meta = build(source=args.source, profile=args.profile, review_cost=args.review_cost)
    print(
        f"Wrote {BUNDLE_PATH.relative_to(REPO_ROOT)} + presets "
        f"(version {meta['version']}, provenance {meta['provenance']}, "
        f"threshold {meta['threshold']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
