"""Monitoring engine tests (slow ones train a model + run Evidently)."""

from __future__ import annotations

import json

import pytest

from monitoring import replay, replay_week


def test_extract_dataset_drift_from_evidently_dict():
    drifted = {
        "metrics": [
            {
                "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                "config": {"drift_share": 0.5},
                "value": {"count": 12.0, "share": 0.6},
            }
        ]
    }
    stable = {
        "metrics": [
            {
                "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                "config": {"drift_share": 0.5},
                "value": {"count": 3.0, "share": 0.3},
            }
        ]
    }
    assert replay._extract_dataset_drift(drifted) is True
    assert replay._extract_dataset_drift(stable) is False
    assert replay._extract_dataset_drift({"metrics": []}) is False


@pytest.mark.slow
def test_prepare_excludes_causal_aggregates_and_sparse():
    ctx = replay.prepare(source="synthetic", profile="smoke", n_batches=5)
    # Causal aggregates are excluded (structural non-stationarity, not drift).
    assert not any(c.startswith("ent_") for c in ctx.numeric_cols)
    assert ctx.numeric_cols  # some dense features remain
    assert 0.0 <= ctx.baseline_value_capture <= 1.0


@pytest.mark.slow
def test_run_batch_produces_report_and_record(tmp_path):
    ctx = replay.prepare(source="synthetic", profile="smoke", n_batches=5)
    result = replay.run_batch(ctx, 0, tmp_path)
    assert {"pr_auc", "value_capture", "max_psi", "breaches", "evidently_drift"} <= result.keys()
    assert (tmp_path / "week-00" / "report.html").exists()
    assert (tmp_path / "week-00" / "evidently.html").exists()


@pytest.mark.slow
def test_injected_drift_triggers_breach(tmp_path):
    replay_week.main(
        [
            "--inject-drift",
            "--reset",
            "--source",
            "synthetic",
            "--profile",
            "smoke",
            "--n-batches",
            "5",
            "--out",
            str(tmp_path),
        ]
    )
    breach_path = tmp_path / "breach.json"
    assert breach_path.exists()
    breach = json.loads(breach_path.read_text())
    assert breach["breaches"]  # a breach was recorded
    assert any("PSI" in b for b in breach["breaches"])  # the injected shift shows as PSI


@pytest.mark.slow
def test_stream_freezes_when_exhausted(tmp_path):
    replay_week.main(
        [
            "--all",
            "--reset",
            "--source",
            "synthetic",
            "--profile",
            "smoke",
            "--n-batches",
            "3",
            "--out",
            str(tmp_path),
        ]
    )
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["frozen"] is True
    assert state["cursor"] == 3
    assert len(state["history"]) == 3
    assert (tmp_path / "index.html").exists()
