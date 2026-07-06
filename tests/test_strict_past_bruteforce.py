"""Strict-past aggregates validated against an independent O(n^2) ground truth.

This is the strongest causality check: the fast vectorized ``_strict_past_stats``
is compared, row by row, to a dead-simple brute-force reference on a deliberately
adversarial frame — tied timestamps, entity-first ties, an entity that only ever
appears in a single tied block, a null entity key, a null amount in the prior set,
unsorted input, and a non-default index. Regression coverage for the audit's
same-timestamp recency leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import features, schema
from tests.conftest import equal_with_nan


def _bruteforce_strict_past(df: pd.DataFrame, entity: str):
    """Reference: for each row, aggregate over rows with strictly-earlier DT and the
    same (non-null) entity. ``count`` counts all strictly-prior rows; ``mean`` is the
    NaN-skipping mean of prior amounts; ``recency`` is time since the last prior row."""
    t = df[schema.TIME_COL].to_numpy()
    a = df[schema.AMT_COL].to_numpy(dtype=float)
    e = df[entity].to_numpy()
    n = len(df)
    count = np.zeros(n)
    mean = np.full(n, np.nan)
    recency = np.full(n, np.nan)
    for i in range(n):
        if pd.isna(e[i]):
            continue
        prior_idx = [j for j in range(n) if e[j] == e[i] and t[j] < t[i]]
        count[i] = len(prior_idx)
        present_amts = [a[j] for j in prior_idx if not np.isnan(a[j])]
        if present_amts:
            mean[i] = np.mean(present_amts)
        if prior_idx:
            recency[i] = t[i] - max(t[j] for j in prior_idx)
    return count, mean, recency


def _adversarial_frame() -> pd.DataFrame:
    rows = [
        # (card1, DT, Amt) — interleaved and unsorted on purpose.
        (1, 10, 1.0),  # entity 1: first appearance is a tie at t=10
        (1, 10, 2.0),
        (3, 15, 7.0),  # entity 3: only ever appears in one tied block
        (3, 15, 8.0),
        (3, 15, 9.0),
        (2, 10, 5.0),  # entity 2 first at t=10
        (1, 20, 3.0),
        (np.nan, 25, 10.0),  # null entity key -> no aggregates
        (4, 20, np.nan),  # entity 4: prior amount is NaN
        (4, 20, 11.0),
        (4, 30, 12.0),
        (2, 40, 6.0),
        (1, 30, 4.0),
    ]
    df = pd.DataFrame(rows, columns=["card1", schema.TIME_COL, schema.AMT_COL])
    df[schema.ID_COL] = np.arange(100, 100 + len(df))
    # Shuffle and give a non-default index to stress alignment/sorting.
    return df.sample(frac=1.0, random_state=5).set_index(np.arange(1000, 1000 + len(df)))


def test_vectorized_matches_bruteforce_on_adversarial_frame():
    df = _adversarial_frame()
    stats = features._strict_past_stats(df, "card1")
    exp_count, exp_mean, exp_rec = _bruteforce_strict_past(df, "card1")
    assert equal_with_nan(stats["prior_count"].to_numpy(), exp_count)
    assert equal_with_nan(stats["prior_amt_mean"].to_numpy(), exp_mean)
    assert equal_with_nan(stats["recency"].to_numpy(), exp_rec)


def test_entity_first_tied_block_recency_is_nan():
    """The audit's exact failure case: recency must be NaN on a first-appearance tie."""
    df = pd.DataFrame(
        {
            "card1": [7, 7, 7],
            schema.TIME_COL: [500, 500, 900],
            schema.AMT_COL: [10.0, 20.0, 30.0],
        }
    )
    rec = features._strict_past_stats(df, "card1")["recency"].to_numpy()
    assert equal_with_nan(rec, np.array([np.nan, np.nan, 400.0]))


def test_recency_independent_of_same_timestamp_sibling():
    """Removing a same-timestamp sibling must not change a past row's recency."""
    with_sibling = pd.DataFrame(
        {"card1": [7, 7, 7], schema.TIME_COL: [500, 500, 900], schema.AMT_COL: [1.0, 2.0, 3.0]}
    )
    without = with_sibling.drop(index=1).reset_index(drop=True)
    r_with = features._strict_past_stats(with_sibling, "card1")["recency"].iloc[0]
    r_without = features._strict_past_stats(without, "card1")["recency"].iloc[0]
    assert np.isnan(r_with) and np.isnan(r_without)


def test_fixture_actually_contains_entity_ties(synthetic_frame: pd.DataFrame):
    """Guard against a hollow suite: the fixture must exercise the strict-past tie path."""
    block_sizes = synthetic_frame.groupby(["card1", schema.TIME_COL]).size()
    assert (block_sizes >= 2).sum() >= 1, "fixture has no (entity, DT) ties — tie path uncovered"
