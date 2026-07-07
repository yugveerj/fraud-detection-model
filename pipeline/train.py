"""Reproducible training entrypoint (SPEC Commands).

Runs the full experiment grid, registers the isotonic-calibrated production model
(``modeling.PRODUCTION_MODEL``) in MLflow, exports ``docs/experiments.md``, and writes
the committed champion record (``artifacts/model/metadata.json``). The Lambda serves the
separate bundle built by ``serving.artifact`` (same seeded config). Seeded and deterministic.

    uv run python -m pipeline.train --full     # full grid + register + artifact
    uv run python -m pipeline.train --smoke     # tiny seeded run for CI plumbing
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib

from pipeline import experiments, modeling

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "model"
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


def save_serving_artifact(result: experiments.GridResult) -> dict:
    """Persist the calibrated model + metadata for serving (Phase D)."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, ARTIFACT_DIR / MODEL_FILE, compress=3)

    reg = result.registry
    holdout = next((c for c in result.calibration if c["method"] == "isotonic"), {})
    metadata = {
        "registered_model": reg.get("registered_model", experiments.REGISTERED_MODEL),
        "version": reg.get("version"),
        "run_id": reg.get("run_id"),
        "model_uri": reg.get("model_uri"),
        "calibration": "isotonic",
        "model_type": f"isotonic-calibrated {modeling.PRODUCTION_MODEL} pipeline",
        "provenance": result.provenance,
        "profile": result.profile,
        "feature_columns": result.feature_columns,
        "operating_threshold": None,  # frozen in Phase C (decision layer)
        "holdout_brier": holdout.get("holdout_brier"),
        "holdout_pr_auc": holdout.get("holdout_pr_auc"),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": (
            "Trained on SYNTHETIC data — placeholder; retrain on real IEEE-CIS."
            if result.provenance != "real"
            else "Trained on real IEEE-CIS data."
        ),
    }
    (ARTIFACT_DIR / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train + register the fraud model")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="full experiment grid (default)")
    mode.add_argument("--smoke", action="store_true", help="tiny seeded run for CI")
    parser.add_argument(
        "--source",
        default="auto",
        choices=["auto", "real", "synthetic"],
        help="data source (auto = real if present, else synthetic)",
    )
    parser.add_argument("--no-register", action="store_true", help="skip MLflow registration")
    args = parser.parse_args(argv)

    profile = "smoke" if args.smoke else "full"
    print(f"Training: profile={profile} source={args.source}")
    result = experiments.run_grid(
        source=args.source, profile=profile, register=not args.no_register
    )

    doc = experiments.write_experiments_md(result)
    metadata = save_serving_artifact(result)

    print(f"\nProvenance: {result.provenance}  ·  registered version: {metadata['version']}")
    print("Leakage (temporal-val vs random-CV PR-AUC):")
    for lk in result.leakage:
        print(
            f"  {lk['model']:9s} {lk['temporal_val_pr_auc']:.4f} -> {lk['random_cv_pr_auc']:.4f}"
            f"  (+{lk['inflation_abs']:.4f}, {lk['inflation_pct']:+.1f}%)"
        )
    print(f"\nWrote {doc.relative_to(REPO_ROOT)} and {ARTIFACT_DIR.relative_to(REPO_ROOT)}/")
    if result.provenance != "real":
        print("LABEL: synthetic run — numbers are illustrative, not results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
