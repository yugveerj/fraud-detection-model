"""Population Stability Index (SPEC Section 6).

PSI quantifies distribution shift between a reference (the model's training window) and
a current batch. Convention: PSI < 0.1 = stable, 0.1–0.2 = moderate shift, **> 0.2 =
significant shift** (the monitoring breach threshold). Reference bin edges are frozen
so successive batches are comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BREACH_THRESHOLD = 0.2
MIN_NONNULL = 30  # skip features with too few current non-null values (small-sample noise)
_EPS = 1e-6


def numeric_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI for a numeric feature, using reference quantile bins (NaN treated as a bin).

    Returns 0.0 (no measurable drift) when the current batch has fewer than
    ``MIN_NONNULL`` non-null values — PSI on a handful of points is noise, not signal.
    Bin count adapts down for small batches for the same reason.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref_valid = ref[~np.isnan(ref)]
    cur_valid = cur[~np.isnan(cur)]
    if ref_valid.size == 0 or cur_valid.size < MIN_NONNULL:
        return 0.0
    bins = min(bins, max(4, cur_valid.size // 25))  # adaptive: fewer bins for small batches
    edges = np.unique(np.quantile(ref_valid, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:  # degenerate (constant reference): fall back to a trivial split
        edges = np.array([ref_valid.min() - 1e-9, ref_valid.max() + 1e-9])
    # Interior edges for digitize; add explicit NaN buckets.
    interior = edges[1:-1]
    ref_bins = np.digitize(ref, interior)
    cur_bins = np.digitize(cur, interior)
    n_buckets = len(interior) + 1
    e_prop = _proportions(ref_bins, n_buckets, np.isnan(ref))
    a_prop = _proportions(cur_bins, n_buckets, np.isnan(cur))
    return _psi_from_props(e_prop, a_prop)


def categorical_psi(reference: pd.Series, current: pd.Series) -> float:
    """PSI for a categorical feature over the union of categories (NaN is a category)."""
    if current.notna().sum() < MIN_NONNULL:
        return 0.0
    ref = reference.astype("object").where(reference.notna(), "__nan__")
    cur = current.astype("object").where(current.notna(), "__nan__")
    cats = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    e = ref.value_counts().reindex(cats).fillna(0).to_numpy()
    a = cur.value_counts().reindex(cats).fillna(0).to_numpy()
    return _psi_from_props(e / max(e.sum(), 1), a / max(a.sum(), 1))


def _proportions(bin_idx: np.ndarray, n_buckets: int, nan_mask: np.ndarray) -> np.ndarray:
    counts = np.bincount(bin_idx[~nan_mask], minlength=n_buckets)[:n_buckets].astype(float)
    counts = np.append(counts, float(nan_mask.sum()))  # explicit NaN bucket
    total = counts.sum()
    return counts / total if total else counts


def _psi_from_props(e: np.ndarray, a: np.ndarray) -> float:
    e = np.clip(e, _EPS, None)
    a = np.clip(a, _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    """PSI per feature (numeric + categorical), sorted descending, with a breach flag."""
    rows = []
    for col in numeric_cols:
        if col in reference.columns and col in current.columns:
            rows.append(
                (col, "numeric", numeric_psi(reference[col].to_numpy(), current[col].to_numpy()))
            )
    for col in categorical_cols:
        if col in reference.columns and col in current.columns:
            rows.append((col, "categorical", categorical_psi(reference[col], current[col])))
    table = pd.DataFrame(rows, columns=["feature", "kind", "psi"]).sort_values(
        "psi", ascending=False, ignore_index=True
    )
    table["breach"] = table["psi"] > BREACH_THRESHOLD
    return table
