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
    for name in ("net_value", "reliability", "score_histogram", "leakage"):
        assert _load(name) == built[name], f"{name}.json is stale — rerun build_web_curves"


def test_score_histogram_is_the_real_holdout():
    """The hero mark-field is real data: 64 bins whose counts sum to the holdout n."""
    hist = _load("score_histogram")
    op = json.loads((REPO_ROOT / "artifacts" / "operating_point.json").read_text())
    assert hist["basis"] == "holdout"
    assert len(hist["bins"]) == 64
    assert sum(b["count"] for b in hist["bins"]) == hist["n"]
    assert hist["n"] == op["holdout"]["n"]
    los = [b["lo"] for b in hist["bins"]]
    assert los == sorted(los) and los[0] == 0.0 and hist["bins"][-1]["hi"] == 1.0


def test_net_value_carries_validation_n():
    """The E1 foot's sample size hydrates from the curve payload, guarded here."""
    op = json.loads((REPO_ROOT / "artifacts" / "operating_point.json").read_text())
    assert _load("net_value")["n"] == op["validation"]["n"]


def test_leakage_range_fallback_prose_matches_data():
    """index.html's static +lo–hi% fallback (JS re-hydrates it) must match leakage.json."""
    lk = _load("leakage")
    infl = [r["inflation"] for r in lk["rows"] if r["inflation"] > 0]
    expected = f"+{round(min(infl))}–{round(max(infl))}%"
    html = (REPO_ROOT / "web" / "index.html").read_text()
    assert expected in html, f"stale leakage-range fallback; expected {expected}"


def test_threshold_fallback_prose_matches_data():
    """The hero/rule-tag static fallbacks (JS re-hydrates them) must match the artifact."""
    op = json.loads((REPO_ROOT / "artifacts" / "operating_point.json").read_text())
    pct = f"{op['threshold'] * 100:.1f}%"
    html = (REPO_ROOT / "web" / "index.html").read_text()
    assert html.count(pct) >= 2, f"hero/rule fallback text diverged from threshold {pct}"


def test_chart_axis_domains_still_hold():
    """E1/E3 axis bounds are intentional design constants — assert the data fits them,
    so a retrain that outgrows a domain fails CI instead of silently clipping."""
    nv = _load("net_value")
    assert max(q["net"] for q in nv["series"]) <= 320000  # E1 yMax
    rel = _load("reliability")
    for series in ("uncalibrated", "isotonic"):
        for q in rel[series]:
            assert q["pred"] <= 0.62 and q["emp"] <= 0.62  # E3 domain


def test_signal_red_defined_exactly_once():
    """The design's red discipline is mechanical: #C8102E exists only as the --signal
    token in style.css — nothing else may restate the hex."""
    web = REPO_ROOT / "web"
    css = (web / "style.css").read_text().lower()
    assert css.count("#c8102e") == 1, "--signal must be the only definition of the red"
    for name in ("index.html", "app.js"):
        assert "#c8102e" not in (web / name).read_text().lower(), f"{name} restates --signal"


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
