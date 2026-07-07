"""web/curves/*.json must stay in sync with the committed source artifacts.

Guards the demo's three charts (net-value, reliability, leakage) against silent drift:
if a retrain shifts a curve and the web JSON is not regenerated, this fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tooling import build_web_curves as bwc

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_CURVES = REPO_ROOT / "web" / "curves"


def _load(name: str) -> dict:
    return json.loads((WEB_CURVES / f"{name}.json").read_text())


def test_web_curves_match_generator():
    """Every committed web/curves file equals a fresh build from the sources."""
    built = bwc.build()
    for name in ("net_value", "reliability", "leakage"):
        assert _load(name) == built[name], f"{name}.json is stale — rerun build_web_curves"


def test_net_value_op_matches_operating_point():
    op = json.loads((REPO_ROOT / "artifacts" / "operating_point.json").read_text())
    nv = _load("net_value")
    assert nv["basis"] == "validation"
    assert nv["op"]["t"] == pytest.approx(round(op["threshold"], 5))
    assert nv["op"]["net_val"] == round(op["validation"]["net_per_100k"])
    assert nv["op"]["net_holdout"] == round(op["holdout"]["net_per_100k"])
    assert nv["op"]["net_val"] > nv["op"]["net_holdout"]  # out-of-time drop, shown honestly
    assert len(nv["series"]) >= 20  # enough points to draw a smooth curve


def test_reliability_has_both_series():
    rel = _load("reliability")
    assert rel["basis"] == "holdout"
    assert len(rel["uncalibrated"]) == rel["n_bins"]
    assert len(rel["isotonic"]) == rel["n_bins"]
    # empirical fraud rates are probabilities
    for series in ("uncalibrated", "isotonic"):
        for pt in rel[series]:
            assert 0.0 <= pt["pred"] <= 1.0 and 0.0 <= pt["emp"] <= 1.0


def test_leakage_lightgbm_matches_experiments():
    """The LightGBM inflation on the chart is the +23.2% headline the page already cites."""
    lk = _load("leakage")
    lgbm = next(r for r in lk["rows"] if r["model"] == "LightGBM")
    assert lgbm["random_cv"] > lgbm["temporal"]  # random-CV is the (inflated) anti-pattern
    assert lgbm["inflation"] == pytest.approx(23.2, abs=0.1)
