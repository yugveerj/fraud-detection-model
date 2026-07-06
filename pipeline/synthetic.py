"""Synthetic, schema-identical IEEE-CIS fixture (SPEC Section 10 fallback).

Generates a small transaction + identity pair that matches the real column
inventory and dtypes, with a deterministic embedded fraud signal, realistic block
missingness, and repeating entities so causal aggregates are exercised.

**This is NOT real data.** It exists solely so the full pipeline and the causality
tests run in CI and locally without Kaggle credentials. It must NEVER be used for
reported metrics — those require the real competition data (see the manifest's
`provenance` field). Every artifact derived from it is labelled synthetic.

    uv run python -m pipeline.synthetic --out tests/fixtures --n 5000 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import schema

FIXTURE_TX = "tx_synthetic.parquet"
FIXTURE_ID = "id_synthetic.parquet"

# Base fraud rate roughly matching the real dataset (~3.5%).
FRAUD_RATE = 0.035


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    return (x - mu) / (sd if sd > 0 else 1.0)


def _apply_missing(rng: np.random.Generator, col: np.ndarray, frac: float) -> np.ndarray:
    col = col.astype(float)
    mask = rng.random(len(col)) < frac
    col[mask] = np.nan
    return col


def make_synthetic(
    n: int = 5000,
    seed: int = 42,
    identity_fraction: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (transaction_df, identity_df) matching the IEEE-CIS schema.

    Rows are sorted ascending by ``TransactionDT``. Identity is present for a
    random ``identity_fraction`` of transactions.
    """
    rng = np.random.default_rng(seed)

    # --- Time: increasing DT with irregular gaps (seconds), starting at ~1 day.
    # Same-second ties are injected below (real IEEE-CIS has many same-second
    # transactions per entity) so the strict-past tie path is genuinely exercised.
    gaps = rng.integers(1, 400, size=n)
    dt = 86400 + np.cumsum(gaps)

    # --- Amount: heavy-tailed, cents preserved.
    amt = np.round(np.exp(rng.normal(4.0, 1.0, size=n)) + rng.random(n), 2)

    tx = pd.DataFrame(
        {
            schema.ID_COL: np.arange(2_987_000, 2_987_000 + n, dtype=np.int64),
            schema.TIME_COL: dt.astype(np.int64),
            schema.AMT_COL: amt,
        }
    )

    # --- Categoricals with repeating entities.
    tx[schema.PRODUCT_COL] = rng.choice(
        schema.PRODUCT_VOCAB, size=n, p=[0.72, 0.11, 0.08, 0.06, 0.03]
    )
    # card1: ~800 distinct issuers so entities repeat (no NaN -> clean entity key).
    tx["card1"] = rng.integers(1000, 1800, size=n)
    tx["card2"] = _apply_missing(rng, rng.integers(100, 600, size=n).astype(float), 0.02)
    tx["card3"] = _apply_missing(rng, np.full(n, 150.0) + rng.integers(0, 40, n), 0.003)
    tx["card4"] = rng.choice(schema.CARD4_VOCAB, size=n, p=[0.65, 0.30, 0.03, 0.02])
    tx["card5"] = _apply_missing(rng, rng.integers(100, 240, size=n).astype(float), 0.05)
    tx["card6"] = rng.choice(schema.CARD6_VOCAB, size=n, p=[0.75, 0.235, 0.01, 0.005])

    tx["addr1"] = _apply_missing(rng, rng.integers(100, 540, size=n).astype(float), 0.12)
    tx["addr2"] = _apply_missing(rng, np.full(n, 87.0) + rng.integers(0, 12, n), 0.12)
    tx["dist1"] = _apply_missing(rng, rng.integers(0, 2000, size=n).astype(float), 0.60)
    tx["dist2"] = _apply_missing(rng, rng.integers(0, 3000, size=n).astype(float), 0.93)

    tx["P_emaildomain"] = _choice_with_na(rng, schema.EMAIL_VOCAB, n, na_frac=0.16)
    tx["R_emaildomain"] = _choice_with_na(rng, schema.EMAIL_VOCAB, n, na_frac=0.77)

    # C1..C14 counting features (non-negative, some heavy-tailed).
    for c in schema.C_COLS:
        lam = rng.uniform(0.5, 6.0)
        tx[c] = rng.poisson(lam, size=n).astype(float)

    # D1..D15 timedelta features (days-ish, with NaN blocks).
    for d in schema.D_COLS:
        vals = np.abs(rng.normal(120, 90, size=n))
        tx[d] = _apply_missing(rng, vals, rng.uniform(0.1, 0.6))

    # M1..M9 match features (mostly T/F; M4 is M0/M1/M2).
    for m in schema.M_COLS:
        if m == "M4":
            tx[m] = _choice_with_na(rng, schema.M4_VOCAB, n, na_frac=0.47)
        else:
            tx[m] = _choice_with_na(rng, schema.M_BINARY_VOCAB, n, na_frac=0.30)

    # V1..V339 Vesta features with block missingness (built in one block to avoid
    # DataFrame fragmentation from 339 successive inserts).
    tx = pd.concat([tx, _v_block_frame(rng, n)], axis=1)

    # --- Identity (subset). --------------------------------------------------
    has_id = rng.random(n) < identity_fraction
    id_ids = tx.loc[has_id, schema.ID_COL].to_numpy()
    m = len(id_ids)
    id_df = pd.DataFrame({schema.ID_COL: id_ids})
    for col in schema.ID_NUM_COLS:
        id_df[col] = _apply_missing(rng, rng.normal(0, 50, size=m), rng.uniform(0.0, 0.4))
    for col in schema.ID_CAT_COLS:
        if col == "id_30":
            id_df[col] = _choice_with_na(rng, schema.ID_30_VOCAB, m, 0.55)
        elif col == "id_31":
            id_df[col] = _choice_with_na(rng, schema.ID_31_VOCAB, m, 0.20)
        elif col == "id_33":
            id_df[col] = _choice_with_na(rng, schema.ID_33_VOCAB, m, 0.55)
        else:
            id_df[col] = _choice_with_na(rng, ["T", "F", "Found", "NotFound"], m, 0.35)
    id_df["DeviceType"] = _choice_with_na(rng, schema.DEVICE_TYPE_VOCAB, m, 0.10)
    id_df["DeviceInfo"] = _choice_with_na(rng, schema.DEVICE_INFO_VOCAB, m, 0.20)

    # Inject same-(entity, TransactionDT) bursts so the causality tests exercise
    # the strict-past tie-collapse (each burst is an entity's first appearance).
    _inject_entity_ties(rng, tx)

    # --- Fraud label driven by a latent signal (deterministic threshold). ----
    tx[schema.TARGET] = _make_labels(rng, tx, has_id)

    # Reorder transaction columns to canonical header order.
    tx = tx[schema.transaction_columns()]
    id_df = id_df[schema.identity_columns()]
    return tx.reset_index(drop=True), id_df.reset_index(drop=True)


def _choice_with_na(
    rng: np.random.Generator, vocab: list[str], n: int, na_frac: float
) -> np.ndarray:
    out = rng.choice(np.array(vocab, dtype=object), size=n)
    mask = rng.random(n) < na_frac
    out[mask] = np.nan
    return out


def _inject_entity_ties(rng: np.random.Generator, tx: pd.DataFrame, n_bursts: int = 12) -> None:
    """Force same-(entity, TransactionDT) bursts in place.

    Each burst sets 2-3 consecutive rows to a fresh ``card1`` (outside the normal
    1000-1799 range, so it is that entity's first appearance) sharing one
    TransactionDT. This yields entity-first tied blocks — the exact strict-past
    tie case the causality tests must cover. Timestamps stay non-decreasing.
    """
    n = len(tx)
    dt = tx[schema.TIME_COL].to_numpy().copy()
    card1 = tx["card1"].to_numpy().copy()
    for k, i in enumerate(np.linspace(50, n - 50, n_bursts).astype(int)):
        size = int(rng.integers(2, 4))
        dt[i : i + size] = dt[i]
        card1[i : i + size] = 2000 + k
    tx[schema.TIME_COL] = dt
    tx["card1"] = card1


def _v_block_frame(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """V1..V339 in ~13 blocks; each block is NaN for a shared random subset of rows."""
    n_blocks = 13
    data: dict[str, np.ndarray] = {}
    for block in np.array_split(schema.V_COLS, n_blocks):
        block_na = rng.random(n) < rng.uniform(0.15, 0.9)
        for col in block:
            # float32 for the 339-column V block keeps the committed fixture ~half
            # size; anonymized Vesta features carry no meaningful sub-float32 precision.
            vals = (rng.normal(0, 1, size=n) + rng.uniform(-1, 1)).astype(np.float32)
            vals[block_na] = np.nan
            data[col] = vals
    return pd.DataFrame(data)


def _make_labels(rng: np.random.Generator, tx: pd.DataFrame, has_id: np.ndarray) -> np.ndarray:
    """Deterministic fraud signal: latent linear combo + noise, thresholded to FRAUD_RATE."""
    latent = (
        0.9 * _z(np.log1p(tx[schema.AMT_COL].to_numpy()))
        + 0.7 * (tx["card6"].to_numpy() == "credit").astype(float)
        + 1.1 * (tx["P_emaildomain"].to_numpy() == "anonymous.com").astype(float)
        + 0.5 * _z(np.nan_to_num(tx["C1"].to_numpy()))
        + 0.4 * _z(np.nan_to_num(tx["V1"].to_numpy()))
        + 0.3 * has_id.astype(float)
    )
    latent = latent + rng.normal(0, 1.0, size=len(tx))
    thresh = np.quantile(latent, 1.0 - FRAUD_RATE)
    return (latent > thresh).astype(np.int64)


def write_fixture(out_dir: Path, n: int, seed: int) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tx, id_df = make_synthetic(n=n, seed=seed)
    tx_path = out_dir / FIXTURE_TX
    id_path = out_dir / FIXTURE_ID
    # zstd keeps the committed fixture small (sparse V-column floats compress well).
    tx.to_parquet(tx_path, index=False, compression="zstd", compression_level=19)
    id_df.to_parquet(id_path, index=False, compression="zstd", compression_level=19)
    return tx_path, id_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic IEEE-CIS fixture")
    parser.add_argument("--out", default="tests/fixtures", help="output directory")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    tx_path, id_path = write_fixture(Path(args.out), args.n, args.seed)
    tx = pd.read_parquet(tx_path)
    rate = tx[schema.TARGET].mean()
    print(f"Wrote {tx_path} ({len(tx)} rows) and {id_path}")
    print(f"Synthetic fraud rate: {rate:.4f}  |  columns: {tx.shape[1]} tx / identity subset")
    print("LABEL: synthetic — never use for reported metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
