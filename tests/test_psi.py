"""PSI correctness tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from monitoring import psi


def test_identical_distribution_is_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 3000)
    assert psi.numeric_psi(x, x) < 0.01


def test_shifted_distribution_breaches():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 3000)
    cur = rng.normal(3, 1, 1000)  # large mean shift
    assert psi.numeric_psi(ref, cur) > psi.BREACH_THRESHOLD


def test_min_sample_guard_returns_zero():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 3000)
    cur = rng.normal(3, 1, 10)  # fewer than MIN_NONNULL -> no measurable drift
    assert psi.numeric_psi(ref, cur) == 0.0


def test_numeric_psi_handles_all_nan_reference():
    assert psi.numeric_psi(np.full(100, np.nan), np.random.default_rng(0).normal(0, 1, 100)) == 0.0


def test_categorical_psi_detects_proportion_shift():
    ref = pd.Series(["a"] * 800 + ["b"] * 200)
    cur = pd.Series(["a"] * 100 + ["b"] * 400)
    assert psi.categorical_psi(ref, cur) > psi.BREACH_THRESHOLD
    assert psi.categorical_psi(ref, ref) < 0.01


def test_psi_table_flags_breach_and_sorts():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"stable": rng.normal(0, 1, 2000), "shifted": rng.normal(0, 1, 2000)})
    cur = pd.DataFrame({"stable": rng.normal(0, 1, 800), "shifted": rng.normal(4, 1, 800)})
    table = psi.psi_table(ref, cur, ["stable", "shifted"], [])
    assert list(table["feature"])[0] == "shifted"  # most drifted first
    assert bool(table.loc[table["feature"] == "shifted", "breach"].iloc[0]) is True
    assert bool(table.loc[table["feature"] == "stable", "breach"].iloc[0]) is False
