"""Schema fidelity + base-feature tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import features, schema
from tests.conftest import equal_with_nan


def test_transaction_schema_column_count():
    # 4 core + ProductCD + 6 card + 2 addr + 2 dist + 2 email + 14 C + 15 D + 9 M + 339 V.
    assert len(schema.transaction_columns()) == 4 + 1 + 6 + 2 + 2 + 2 + 14 + 15 + 9 + 339
    assert schema.transaction_columns()[0] == schema.ID_COL
    assert schema.TARGET in schema.transaction_columns()


def test_identity_schema_column_count():
    # TransactionID + id_01..id_38 (38) + DeviceType + DeviceInfo.
    assert len(schema.identity_columns()) == 1 + 38 + 2
    assert [f"id_{i:02d}" for i in range(1, 12)] == schema.ID_NUM_COLS
    assert [f"id_{i:02d}" for i in range(12, 39)] == schema.ID_CAT_COLS


def test_synthetic_frame_matches_schema(synthetic_frame: pd.DataFrame):
    expected = set(schema.transaction_columns()) | set(schema.identity_columns())
    assert set(synthetic_frame.columns) == expected
    assert synthetic_frame[schema.ID_COL].is_unique


def test_fraud_rate_is_plausible(synthetic_frame: pd.DataFrame):
    rate = synthetic_frame[schema.TARGET].mean()
    assert 0.01 < rate < 0.10  # synthetic base rate ~3.5%


def test_fraud_targets_higher_amounts(synthetic_frame: pd.DataFrame):
    """Fraud amounts skew higher than legit (economics: review must beat fraud caught)."""
    med = synthetic_frame.groupby(schema.TARGET)[schema.AMT_COL].median()
    assert med[1] > med[0] * 1.5


def test_base_features_are_row_local(synthetic_frame: pd.DataFrame):
    """Permuting rows must permute base features identically (no cross-row dependence)."""
    base_cols = ["amt_log", "amt_cents", "amt_is_round", "dt_hour", "dt_dow", "dt_hour_sin"]
    original = features.add_base_features(synthetic_frame)
    perm = synthetic_frame.sample(frac=1.0, random_state=3)
    permuted = features.add_base_features(perm)
    # Realign permuted rows back to original order and compare.
    permuted_realigned = permuted.set_index(schema.ID_COL).loc[synthetic_frame[schema.ID_COL]]
    for col in base_cols:
        assert equal_with_nan(original[col].to_numpy(), permuted_realigned[col].to_numpy())


def test_amt_is_round_flag():
    df = pd.DataFrame({schema.AMT_COL: [100.00, 59.99, 25.0, 25.5], schema.TIME_COL: [1, 2, 3, 4]})
    out = features.add_base_features(df)
    assert list(out["amt_is_round"]) == [1, 0, 1, 0]


def test_time_decomposition_ranges(featured_frame: pd.DataFrame):
    assert featured_frame["dt_hour"].between(0, 23).all()
    assert featured_frame["dt_dow"].between(0, 6).all()
    assert np.isfinite(featured_frame["dt_hour_sin"]).all()
