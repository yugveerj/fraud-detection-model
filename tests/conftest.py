"""Shared pytest fixtures. All tests run on the synthetic fixture (never real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import data, features


@pytest.fixture(scope="session")
def synthetic_dataset() -> data.Dataset:
    """Merged synthetic dataset (transactions + identity), sorted by TransactionDT."""
    return data.load_dataset("synthetic")


@pytest.fixture(scope="session")
def synthetic_frame(synthetic_dataset: data.Dataset) -> pd.DataFrame:
    return synthetic_dataset.frame


@pytest.fixture(scope="session")
def featured_frame(synthetic_frame: pd.DataFrame) -> pd.DataFrame:
    """Synthetic frame with base + causal features."""
    return features.build_features(synthetic_frame)


def equal_with_nan(a: np.ndarray, b: np.ndarray) -> bool:
    """Array equality treating NaN == NaN (for float feature columns)."""
    return np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True)
