"""web/metrics.json must stay in sync with the committed source artifacts (H1).

Guards the demo results-card numbers against silent drift: if the model, operating
point, or leakage figure changes and metrics.json is not regenerated, this fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tooling import build_web_metrics as bwm

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = REPO_ROOT / "web" / "metrics.json"


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


def test_metrics_file_matches_generator(metrics: dict):
    """The committed file equals a fresh build from the sources (no manual drift)."""
    assert metrics == bwm.build()


def test_value_capture_matches_operating_point(metrics: dict):
    op = json.loads((REPO_ROOT / "artifacts" / "operating_point.json").read_text())
    vc = next(h for h in metrics["headline"] if "value captured" in h["label"])
    assert vc["raw"] == pytest.approx(op["holdout"]["value_capture_rate"])
    assert vc["value"] == f"{op['holdout']['value_capture_rate'] * 100:.1f}%"


def test_net_value_matches_operating_point(metrics: dict):
    op = json.loads((REPO_ROOT / "artifacts" / "operating_point.json").read_text())
    net = next(h for h in metrics["headline"] if "net value" in h["label"])
    assert net["raw"] == pytest.approx(op["holdout"]["net_per_100k"])


def test_registry_version_matches_metadata(metrics: dict):
    meta = json.loads((REPO_ROOT / "artifacts" / "model" / "metadata.json").read_text())
    assert metrics["registry_version"] == int(meta["version"])
    assert metrics["model"].lower() in meta["model_type"].lower()


def test_threshold_matches_operating_point(metrics: dict):
    op = json.loads((REPO_ROOT / "artifacts" / "operating_point.json").read_text())
    assert metrics["threshold"] == round(op["threshold"], 4)


def test_total_transactions_matches_manifest(metrics: dict):
    manifest = (REPO_ROOT / "docs" / "data_manifest.md").read_text()
    total = next(h for h in metrics["headline"] if "transactions analyzed" in h["label"])
    assert total["raw"] == bwm._manifest_total_rows(manifest)
