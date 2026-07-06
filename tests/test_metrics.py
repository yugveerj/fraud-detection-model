"""Metric correctness tests."""

from __future__ import annotations

import numpy as np

from pipeline import metrics


def test_perfect_separation_scores():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    ev = metrics.evaluate(y, p)
    assert ev.pr_auc == 1.0
    assert ev.roc_auc == 1.0
    assert ev.n == 4 and ev.n_pos == 2


def test_brier_of_constant_prediction():
    y = np.array([0, 0, 0, 1])  # base rate 0.25
    ev = metrics.evaluate(y, np.full(4, 0.25))
    assert abs(ev.brier - 0.25 * 0.75) < 1e-9  # constant-at-baserate Brier


def test_recall_at_fpr_is_monotonic_in_fpr():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.1).astype(int)
    p = 0.3 * y + rng.random(1000) * 0.7  # separable-ish
    rec = metrics.recall_at_fpr(y, p, fpr_points=(0.01, 0.05, 0.1, 0.2))
    vals = [rec[k] for k in (0.01, 0.05, 0.1, 0.2)]
    assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:], strict=False))  # recall grows with FPR
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_reliability_curve_shapes_and_range():
    rng = np.random.default_rng(1)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)  # well-calibrated by construction
    mean_pred, obs_freq, counts = metrics.reliability_curve(y, p, n_bins=5)
    assert len(mean_pred) == len(obs_freq) == len(counts)
    assert counts.sum() == 500
    assert np.all((obs_freq >= 0) & (obs_freq <= 1))
