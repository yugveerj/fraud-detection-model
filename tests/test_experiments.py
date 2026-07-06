"""Experiment grid orchestration test (slow: trains the full smoke grid)."""

from __future__ import annotations

import pytest

from pipeline import experiments


@pytest.mark.slow
def test_run_grid_smoke_structure(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    r = experiments.run_grid(source="synthetic", profile="smoke", register=False, tracking_uri=uri)
    # 3 models x 2 protocols.
    assert len(r.cells) == 6
    assert {c["protocol"] for c in r.cells} == {"temporal", "random_cv"}
    # Leakage row per model with the required keys.
    assert len(r.leakage) == len(experiments.modeling.MODEL_NAMES)
    for lk in r.leakage:
        assert {
            "temporal_val_pr_auc",
            "random_cv_pr_auc",
            "inflation_abs",
            "inflation_pct",
        } <= lk.keys()
    # Calibration includes none + isotonic + platt, each with val + holdout Brier.
    methods = {c["method"] for c in r.calibration}
    assert {"none", "isotonic", "platt"} <= methods
    assert all("val_brier" in c for c in r.calibration)
    # A calibrated model is exposed for serving even without registration.
    assert r.model is not None
    assert r.feature_counts["numeric"] > 0

    md = experiments.render_experiments_md(r)
    assert "Leakage experiment" in md
    assert "Registered model" in md
