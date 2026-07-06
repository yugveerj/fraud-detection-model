"""Temporal segmentation tests — splits are strictly by TransactionDT."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import data, schema


def test_temporal_split_raises_on_null_transactiondt():
    """A null timestamp would match no split mask and vanish — must raise, not drop."""
    df = pd.DataFrame(
        {
            schema.ID_COL: range(10),
            schema.TIME_COL: [10, 20, 30, np.nan, 50, 60, 70, 80, 90, 100],
            schema.TARGET: [0] * 10,
        }
    )
    with pytest.raises(ValueError, match="null"):
        data.temporal_split(df, provenance="synthetic")


def test_splits_are_temporally_disjoint_and_ordered(featured_frame: pd.DataFrame):
    sp = data.temporal_split(featured_frame, provenance="synthetic")
    assert sp.train[schema.TIME_COL].max() < sp.val[schema.TIME_COL].min()
    assert sp.val[schema.TIME_COL].max() < sp.holdout[schema.TIME_COL].min()


def test_no_transaction_id_appears_in_two_splits(featured_frame: pd.DataFrame):
    sp = data.temporal_split(featured_frame, provenance="synthetic")
    ids = [set(part[schema.ID_COL]) for part in (sp.train, sp.val, sp.holdout)]
    assert ids[0].isdisjoint(ids[1])
    assert ids[1].isdisjoint(ids[2])
    assert ids[0].isdisjoint(ids[2])
    assert len(ids[0]) + len(ids[1]) + len(ids[2]) == len(featured_frame)


def test_fractions_are_approximately_60_15_25(featured_frame: pd.DataFrame):
    sp = data.temporal_split(featured_frame, provenance="synthetic")
    total = len(featured_frame)
    assert abs(len(sp.train) / total - 0.60) < 0.03
    assert abs(len(sp.val) / total - 0.15) < 0.03
    assert abs(len(sp.holdout) / total - 0.25) < 0.03


def test_holdout_is_the_latest_slice(featured_frame: pd.DataFrame):
    sp = data.temporal_split(featured_frame, provenance="synthetic")
    assert sp.holdout[schema.TIME_COL].max() == featured_frame[schema.TIME_COL].max()
    assert sp.train[schema.TIME_COL].min() == featured_frame[schema.TIME_COL].min()


def test_summary_shape(featured_frame: pd.DataFrame):
    sp = data.temporal_split(featured_frame, provenance="synthetic")
    summary = sp.summary()
    assert list(summary["split"]) == ["train", "val", "holdout"]
    assert summary["rows"].sum() == len(featured_frame)
