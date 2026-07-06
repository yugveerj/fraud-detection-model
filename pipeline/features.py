"""Feature engineering — causality is the headline constraint (CLAUDE.md).

Two kinds of features:

* **Base features** — row-local transforms of a single transaction (amount, time
  decomposition). Trivially causal: a row's value depends only on that row.
* **Causal aggregates** — per-entity expanding statistics (count, mean/std amount,
  recency) computed over **strictly earlier** transactions only. Rows that share a
  ``TransactionDT`` do not see one another: the aggregate for every row in a
  ``(entity, TransactionDT)`` block is the value as of the instant *before* that
  block. This strict-past rule is what the causality tests in
  ``tests/test_causality.py`` enforce by shuffling/truncating future rows.

Aggregates are computed on the full time-ordered dataset *before* splitting: a
validation/holdout row legitimately uses its entity's entire past (including train
rows), because "past" is defined by time, not by split membership.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import schema

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400

# Names of the aggregate statistics produced per entity.
AGG_STATS = ["prior_count", "prior_amt_mean", "prior_amt_std", "recency"]


# ---------------------------------------------------------------- base features


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-local transaction/time transforms (no cross-row dependence)."""
    df = df.copy()
    amt = df[schema.AMT_COL].astype(float)
    df["amt_log"] = np.log1p(amt)
    cents = (amt - np.floor(amt)).round(2)
    df["amt_cents"] = cents
    df["amt_is_round"] = (cents == 0).astype("int8")

    dt = df[schema.TIME_COL].astype("int64")
    day_index = dt // SECONDS_PER_DAY
    df["dt_day"] = day_index
    df["dt_hour"] = (dt // SECONDS_PER_HOUR) % 24
    df["dt_dow"] = day_index % 7
    df["dt_is_night"] = ((df["dt_hour"] < 6) | (df["dt_hour"] >= 22)).astype("int8")
    # Cyclical encoding of hour for linear models.
    df["dt_hour_sin"] = np.sin(2 * np.pi * df["dt_hour"] / 24)
    df["dt_hour_cos"] = np.cos(2 * np.pi * df["dt_hour"] / 24)
    return df


# ------------------------------------------------------------ causal aggregates


def _strict_past_stats(
    df: pd.DataFrame,
    entity: str,
    time_col: str = schema.TIME_COL,
    amt_col: str = schema.AMT_COL,
) -> pd.DataFrame:
    """Strictly-past per-entity stats, aligned to ``df.index`` (original order).

    Returns columns ``prior_count``, ``prior_amt_mean``, ``prior_amt_std``,
    ``recency``. Rows whose entity is null, or whose entity has no strictly-earlier
    transaction, get 0 count and NaN mean/std/recency.
    """
    n = len(df)
    work = pd.DataFrame(
        {
            "_orig": np.arange(n),
            "_ent": df[entity].to_numpy(),
            "_t": df[time_col].to_numpy(),
            "_a": df[amt_col].to_numpy(dtype=float),
        }
    )
    order = work.sort_values(["_t", "_orig"], kind="stable")
    valid = order["_ent"].notna().to_numpy()
    sub = order.loc[valid].copy()
    # Amount summation fills NaN -> 0 and counts present amounts separately, so the
    # mean is a consistent NaN-skipping mean (sum of present amounts / count of
    # present amounts) even for rows whose own amount is missing. `prior_count`
    # counts ALL strictly-prior rows regardless of amount presence. On real
    # IEEE-CIS data TransactionAmt is never null, so present == count throughout.
    sub["_af"] = sub["_a"].fillna(0.0)
    sub["_present"] = sub["_a"].notna().astype(float)
    sub["_sqf"] = sub["_af"] ** 2

    g = sub.groupby("_ent", sort=False)
    inc_count = g.cumcount() + 1  # inclusive count of all rows in sort order
    inc_present = g["_present"].cumsum()
    inc_sum = g["_af"].cumsum()
    inc_sumsq = g["_sqf"].cumsum()
    prior_time_incl = g["_t"].shift(1)  # time of prior-in-sort row (may tie on _t)

    # Exclusive-of-self, but still includes same-timestamp earlier-in-sort rows.
    sub["_excl_count"] = inc_count - 1
    sub["_excl_present"] = inc_present - sub["_present"]
    sub["_excl_sum"] = inc_sum - sub["_af"]
    sub["_excl_sumsq"] = inc_sumsq - sub["_sqf"]
    sub["_prior_t"] = prior_time_incl

    # Enforce STRICT past: every row in a (entity, time) block takes the exclusive
    # stats of the block's first row — i.e. the state before any same-time row.
    # NOTE: pandas transform("first") returns the first NON-NULL value, not the
    # first positional row. That is harmless for count/present/sum/sumsq (their
    # block-first values are 0.0 / real numbers, never NaN), but `_prior_t` IS NaN
    # on a block's first row when the entity has no strictly-earlier history, so
    # transform("first") would reach past it to a same-timestamp sibling. We
    # therefore gate recency on count > 0 below, which forces NaN exactly when no
    # strictly-past row exists — independent of any same-timestamp sibling.
    blk = sub.groupby(["_ent", "_t"], sort=False)
    strict_count = blk["_excl_count"].transform("first").to_numpy()
    strict_present = blk["_excl_present"].transform("first").to_numpy()
    strict_sum = blk["_excl_sum"].transform("first").to_numpy()
    strict_sumsq = blk["_excl_sumsq"].transform("first").to_numpy()
    strict_prior_t = blk["_prior_t"].transform("first").to_numpy()

    with np.errstate(invalid="ignore", divide="ignore"):
        count = strict_count
        has_prior = count > 0
        has_amt = strict_present > 0
        denom = np.where(has_amt, strict_present, 1.0)  # avoid 0-division; masked out below
        mean = np.where(has_amt, strict_sum / denom, np.nan)
        var = np.clip(np.where(has_amt, strict_sumsq / denom - mean**2, np.nan), 0.0, None)
        std = np.sqrt(var)
        recency = np.where(has_prior, sub["_t"].to_numpy() - strict_prior_t, np.nan)

    stats = pd.DataFrame(
        {
            "prior_count": count.astype(float),
            "prior_amt_mean": mean,
            "prior_amt_std": std,
            "recency": recency,
        },
        index=sub["_orig"].to_numpy(),
    )
    # Reindex to all rows (null-entity rows -> NaN stats, count 0), restore order.
    stats = stats.reindex(range(n))
    stats["prior_count"] = stats["prior_count"].fillna(0.0)
    return stats.reset_index(drop=True)


def add_causal_aggregates(
    df: pd.DataFrame,
    entity_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add strict-past aggregate features for each entity in ``entity_cols``."""
    entity_cols = entity_cols or schema.ENTITY_COLS
    df = df.reset_index(drop=True).copy()
    new_cols: dict[str, np.ndarray] = {}
    for entity in entity_cols:
        stats = _strict_past_stats(df, entity)
        prefix = f"ent_{entity}__"
        for stat in AGG_STATS:
            new_cols[prefix + stat] = stats[stat].to_numpy()
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def build_features(df: pd.DataFrame, entity_cols: list[str] | None = None) -> pd.DataFrame:
    """Full feature build: base transforms + causal aggregates."""
    return add_causal_aggregates(add_base_features(df), entity_cols=entity_cols)


def causal_feature_names(entity_cols: list[str] | None = None) -> list[str]:
    """Names of the causal aggregate columns produced by ``add_causal_aggregates``."""
    entity_cols = entity_cols or schema.ENTITY_COLS
    return [f"ent_{e}__{stat}" for e in entity_cols for stat in AGG_STATS]
