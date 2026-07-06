"""Dollar-framed decision layer tests — cost model correctness."""

from __future__ import annotations

import numpy as np

from pipeline import decisions

# Hand example: 2 fraud (amounts 100, 200) and 2 legit (50, 30).
Y = np.array([1, 0, 1, 0])
PROBA = np.array([0.9, 0.8, 0.4, 0.1])
AMOUNTS = np.array([100.0, 50.0, 200.0, 30.0])


def test_net_value_curve_matches_hand_computation():
    curve = decisions.net_value_curve(Y, PROBA, AMOUNTS, review_cost=25.0)
    # Sorted by proba desc: alerts 1..4, fraud caught value 100,100,300,300.
    assert list(curve["fraud_value_caught"]) == [100.0, 100.0, 300.0, 300.0]
    # net = caught - 25*alerts.
    assert list(curve["net_value"]) == [75.0, 50.0, 225.0, 200.0]
    assert curve["fraud_value_total"].iloc[0] == 300.0


def test_optimize_picks_max_net_value():
    op = decisions.optimize_operating_point(Y, PROBA, AMOUNTS, review_cost=25.0)
    assert op["net_value"] == 225.0
    assert op["threshold"] == 0.4  # third-ranked probability
    assert op["value_capture_rate"] == 1.0  # all fraud value caught by then


def test_evaluate_at_threshold_consistency():
    stats = decisions.evaluate_at_threshold(Y, PROBA, AMOUNTS, threshold=0.4, review_cost=25.0)
    assert stats["alerts"] == 3
    assert stats["fraud_value_caught"] == 300.0
    assert stats["net_value"] == 225.0
    assert stats["recall"] == 1.0
    assert stats["fpr"] == 0.5  # 1 of 2 legit alerted


def test_net_identity_holds(featured_frame):
    """net == fraud_value_caught - review_cost * alerts, on real-ish featured data."""
    from pipeline import data, schema

    sp = data.temporal_split(featured_frame, provenance="synthetic")
    y = sp.holdout[schema.TARGET].to_numpy()
    rng = np.random.default_rng(0)
    proba = rng.random(len(y))
    amt = sp.holdout[schema.AMT_COL].to_numpy()
    s = decisions.evaluate_at_threshold(y, proba, amt, threshold=0.5, review_cost=25.0)
    assert abs(s["net_value"] - (s["fraud_value_caught"] - 25.0 * s["alerts"])) < 1e-6


def test_sensitivity_table_threshold_monotonic_in_cost():
    """Higher review cost should not lower the optimal threshold."""
    table = decisions.sensitivity_table(Y, PROBA, AMOUNTS, review_costs=(5, 25, 75))
    thr = list(table["opt_threshold"])
    assert all(a <= b + 1e-9 for a, b in zip(thr, thr[1:], strict=False))


def test_statement_mentions_capture_fpr_and_net():
    stats = decisions.evaluate_at_threshold(Y, PROBA, AMOUNTS, threshold=0.4, review_cost=25.0)
    s = decisions.operating_point_statement(stats, provenance="synthetic")
    assert "captures" in s and "false-positive" in s and "net value" in s
    assert "SYNTHETIC" in s
