"""Data loading + temporal segmentation.

Loads the real competition CSVs from ``data/`` when present, otherwise falls back
to the synthetic fixture (SPEC Section 10). Every split is by ``TransactionDT`` —
the project's headline constraint (CLAUDE.md). The HOLDOUT slice is later replayed
as the monitoring stream (SPEC Section 6).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline import schema
from pipeline.synthetic import FIXTURE_ID, FIXTURE_TX, make_synthetic

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"

RAW_TX = "train_transaction.csv"
RAW_ID = "train_identity.csv"

# Temporal segmentation fractions (SPEC Section 1).
TRAIN_FRAC = 0.60
VAL_FRAC = 0.15
HOLDOUT_FRAC = 0.25


@dataclass(frozen=True)
class Dataset:
    """A loaded, joined dataset with provenance."""

    frame: pd.DataFrame
    provenance: str  # "real" | "synthetic"

    @property
    def is_real(self) -> bool:
        return self.provenance == "real"


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    holdout: pd.DataFrame
    cut_train_val: float  # TransactionDT boundary: train < cut <= val
    cut_val_holdout: float  # TransactionDT boundary: val < cut <= holdout
    provenance: str

    def summary(self) -> pd.DataFrame:
        rows = []
        total = len(self.train) + len(self.val) + len(self.holdout)
        for name, part in (("train", self.train), ("val", self.val), ("holdout", self.holdout)):
            rows.append(
                {
                    "split": name,
                    "rows": len(part),
                    "frac": round(len(part) / total, 4) if total else 0.0,
                    "dt_min": int(part[schema.TIME_COL].min()) if len(part) else None,
                    "dt_max": int(part[schema.TIME_COL].max()) if len(part) else None,
                    "fraud_rate": round(float(part[schema.TARGET].mean()), 5)
                    if len(part)
                    else None,
                }
            )
        return pd.DataFrame(rows)


def _merge(tx: pd.DataFrame, id_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join identity onto transactions and sort by TransactionDT (stable)."""
    merged = tx.merge(id_df, on=schema.ID_COL, how="left")
    merged = merged.sort_values([schema.TIME_COL, schema.ID_COL], kind="stable").reset_index(
        drop=True
    )
    return merged


def load_raw(data_dir: Path | str = DEFAULT_DATA_DIR) -> Dataset:
    """Load and join the real competition CSVs. Raises if they are absent."""
    data_dir = Path(data_dir)
    tx_path, id_path = data_dir / RAW_TX, data_dir / RAW_ID
    if not tx_path.exists() or not id_path.exists():
        raise FileNotFoundError(
            f"Raw data not found under {data_dir} (need {RAW_TX} + {RAW_ID}). "
            "Run `uv run python -m pipeline.fetch_data` with Kaggle credentials, "
            "or use load_dataset(source='synthetic')."
        )
    tx = pd.read_csv(tx_path)
    id_df = pd.read_csv(id_path)
    return Dataset(_merge(tx, id_df), provenance="real")


def load_synthetic(fixture_dir: Path | str = FIXTURE_DIR, n: int = 5000, seed: int = 42) -> Dataset:
    """Load the committed synthetic fixture, generating it in-memory if absent."""
    fixture_dir = Path(fixture_dir)
    tx_path, id_path = fixture_dir / FIXTURE_TX, fixture_dir / FIXTURE_ID
    if tx_path.exists() and id_path.exists():
        tx, id_df = pd.read_parquet(tx_path), pd.read_parquet(id_path)
    else:
        tx, id_df = make_synthetic(n=n, seed=seed)
    return Dataset(_merge(tx, id_df), provenance="synthetic")


def load_dataset(source: str = "auto", data_dir: Path | str = DEFAULT_DATA_DIR) -> Dataset:
    """Load the dataset.

    source="auto": real data if present under ``data_dir``, else synthetic fixture.
    source="real": real data only (raises if absent).
    source="synthetic": synthetic fixture only.
    """
    if source == "real":
        return load_raw(data_dir)
    if source == "synthetic":
        return load_synthetic()
    if source == "auto":
        data_dir = Path(data_dir)
        if (data_dir / RAW_TX).exists() and (data_dir / RAW_ID).exists():
            return load_raw(data_dir)
        return load_synthetic()
    raise ValueError(f"unknown source: {source!r}")


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    provenance: str = "unknown",
) -> SplitResult:
    """Split strictly by ``TransactionDT`` at time quantiles.

    train:   TransactionDT <  cut_train_val
    val:     cut_train_val <= TransactionDT < cut_val_holdout
    holdout: TransactionDT >= cut_val_holdout

    Assignment is by time value, so a transaction never appears in an earlier split
    than one that occurred before it. Ties at a boundary fall into the later split;
    realised fractions may drift slightly from the targets when timestamps repeat.

    Raises ``ValueError`` if any ``TransactionDT`` is null — a null timestamp would
    match no split mask and silently vanish, breaking the exact-partition contract.
    """
    n_null = int(df[schema.TIME_COL].isna().sum())
    if n_null:
        raise ValueError(
            f"{schema.TIME_COL} contains {n_null} null value(s); a temporal split "
            "requires a timestamp for every row (would otherwise drop rows silently)."
        )
    if not df[schema.TIME_COL].is_monotonic_increasing:
        df = df.sort_values([schema.TIME_COL, schema.ID_COL], kind="stable").reset_index(drop=True)

    dt = df[schema.TIME_COL]
    cut_train_val = float(dt.quantile(train_frac))
    cut_val_holdout = float(dt.quantile(train_frac + val_frac))

    train = df[dt < cut_train_val]
    val = df[(dt >= cut_train_val) & (dt < cut_val_holdout)]
    holdout = df[dt >= cut_val_holdout]

    return SplitResult(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        holdout=holdout.reset_index(drop=True),
        cut_train_val=cut_train_val,
        cut_val_holdout=cut_val_holdout,
        provenance=provenance,
    )
