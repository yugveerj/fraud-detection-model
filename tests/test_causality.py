"""Property-style causality tests — the project's headline guarantee (CLAUDE.md).

Every causal aggregate for a past row must be invariant to what happens in the
future. We prove it three ways: truncating the future, shuffling the future, and
appending brand-new future rows. In all cases the aggregate values of the earlier
rows must be byte-for-byte unchanged. We also pin the strict-past tie semantics on
a hand-built frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import features, schema
from tests.conftest import equal_with_nan

CUT_POINTS = [500, 1500, 3000, 4200]


def _causal(df: pd.DataFrame) -> pd.DataFrame:
    return features.add_causal_aggregates(features.add_base_features(df))[
        features.causal_feature_names()
    ]


@pytest.mark.causality
@pytest.mark.parametrize("cut", CUT_POINTS)
def test_truncating_future_leaves_past_unchanged(synthetic_frame: pd.DataFrame, cut: int):
    full = _causal(synthetic_frame)
    truncated = _causal(synthetic_frame.iloc[:cut].copy())
    assert equal_with_nan(full.iloc[:cut].to_numpy(), truncated.to_numpy())


@pytest.mark.causality
@pytest.mark.parametrize("cut", CUT_POINTS)
def test_shuffling_future_leaves_past_unchanged(synthetic_frame: pd.DataFrame, cut: int):
    full = _causal(synthetic_frame)
    future = synthetic_frame.iloc[cut:].sample(frac=1.0, random_state=7)
    shuffled = pd.concat([synthetic_frame.iloc[:cut], future]).reset_index(drop=True)
    recomputed = _causal(shuffled)
    assert equal_with_nan(full.iloc[:cut].to_numpy(), recomputed.iloc[:cut].to_numpy())


@pytest.mark.causality
def test_appending_new_future_rows_leaves_past_unchanged(synthetic_frame: pd.DataFrame):
    """Adding transactions strictly after the last DT must not touch any past row."""
    cut = len(synthetic_frame)
    full = _causal(synthetic_frame)
    # Fabricate future rows reusing existing entities at later timestamps.
    extra = synthetic_frame.iloc[:200].copy()
    extra[schema.TIME_COL] = synthetic_frame[schema.TIME_COL].max() + 1 + np.arange(len(extra))
    appended = pd.concat([synthetic_frame, extra]).reset_index(drop=True)
    recomputed = _causal(appended)
    assert equal_with_nan(full.to_numpy(), recomputed.iloc[:cut].to_numpy())


@pytest.mark.causality
def test_strict_past_tie_semantics():
    """Rows sharing a TransactionDT must not see each other (strict past)."""
    df = pd.DataFrame(
        {
            schema.ID_COL: [1, 2, 3, 4, 5],
            schema.TIME_COL: [100, 100, 200, 200, 300],
            schema.AMT_COL: [10.0, 20.0, 30.0, 40.0, 50.0],
            "card1": [7, 7, 7, 7, 7],
        }
    )
    stats = features._strict_past_stats(df, "card1")
    # DT=100 block: no prior -> count 0, mean/recency NaN.
    assert list(stats["prior_count"]) == [0, 0, 2, 2, 4]
    assert np.isnan(stats["prior_amt_mean"].iloc[0])
    assert np.isnan(stats["prior_amt_mean"].iloc[1])
    # Regression (audit finding): recency on the entity-FIRST tied block must be NaN,
    # not fabricated from a same-timestamp sibling.
    assert np.isnan(stats["recency"].iloc[0])
    assert np.isnan(stats["recency"].iloc[1])
    # DT=200 block sees only the two DT=100 rows: mean = (10+20)/2 = 15, recency = 100.
    assert stats["prior_amt_mean"].iloc[2] == pytest.approx(15.0)
    assert stats["prior_amt_mean"].iloc[3] == pytest.approx(15.0)
    assert stats["recency"].iloc[2] == pytest.approx(100.0)
    assert stats["recency"].iloc[3] == pytest.approx(100.0)
    # DT=300 sees all four earlier: mean = (10+20+30+40)/4 = 25, recency = 300-200 = 100.
    assert stats["prior_amt_mean"].iloc[4] == pytest.approx(25.0)
    assert stats["recency"].iloc[4] == pytest.approx(100.0)


@pytest.mark.causality
def test_first_occurrence_has_zero_count_and_nan_stats(synthetic_frame: pd.DataFrame):
    full = features.add_causal_aggregates(synthetic_frame)
    for entity in schema.ENTITY_COLS:
        prefix = f"ent_{entity}__"
        first_mask = ~synthetic_frame[entity].duplicated() & synthetic_frame[entity].notna()
        counts = full.loc[first_mask, prefix + "prior_count"]
        assert (counts == 0).all()
        assert full.loc[first_mask, prefix + "prior_amt_mean"].isna().all()


@pytest.mark.causality
def test_recency_is_nonnegative(featured_frame: pd.DataFrame):
    for col in [c for c in features.causal_feature_names() if c.endswith("recency")]:
        vals = featured_frame[col].dropna()
        assert (vals >= 0).all()


@pytest.mark.causality
def test_prior_count_never_exceeds_entity_history(synthetic_frame: pd.DataFrame):
    full = features.add_causal_aggregates(synthetic_frame)
    for entity in schema.ENTITY_COLS:
        size = synthetic_frame.groupby(entity)[schema.ID_COL].transform("size")
        prior = full[f"ent_{entity}__prior_count"]
        # A strictly-past count is at most (group size - 1).
        assert (
            prior[synthetic_frame[entity].notna()] <= size[synthetic_frame[entity].notna()] - 1
        ).all()
