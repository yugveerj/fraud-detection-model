"""Kaggle IEEE-CIS data pull + integrity manifest (SPEC Section 1).

Downloads ``train_transaction.csv`` and ``train_identity.csv`` via the Kaggle API
using owner credentials, verifies row counts, and writes ``docs/data_manifest.md``.
Raw data lands in ``data/`` and is gitignored — only the manifest is committed.

Requires ``kaggle`` (``uv sync --group data``), a readable ``~/.kaggle/kaggle.json``
(or ``KAGGLE_USERNAME``/``KAGGLE_KEY``), and accepted IEEE-CIS competition rules.

    uv run python -m pipeline.fetch_data              # pull real data + manifest
    uv run python -m pipeline.fetch_data --synthetic  # (re)build fixture + labelled manifest

If credentials are absent, the command prints the exact remediation and exits
non-zero — it never fabricates data (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import pandas as pd

from pipeline import manifest as manifest_mod
from pipeline import schema
from pipeline.data import DEFAULT_DATA_DIR, RAW_ID, RAW_TX
from pipeline.synthetic import write_fixture

COMPETITION = "ieee-fraud-detection"
FILES = [RAW_TX, RAW_ID]


def _kaggle_available() -> tuple[bool, str]:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        return False, "kaggle package not installed — run `uv sync --group data`"
    except OSError as exc:
        # kaggle raises on import if credentials are missing/misconfigured.
        return False, f"kaggle credentials not resolved: {exc}"
    return True, ""


def _download_real(data_dir: Path) -> None:
    import kaggle

    kaggle.api.authenticate()
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        print(f"Downloading {fname} …")
        kaggle.api.competition_download_file(COMPETITION, fname, path=str(data_dir))
        zpath = data_dir / f"{fname}.zip"
        if zpath.exists():
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(data_dir)
            zpath.unlink()


def fetch_real(data_dir: Path = DEFAULT_DATA_DIR) -> int:
    ok, why = _kaggle_available()
    if not ok:
        print(f"ERROR: {why}", file=sys.stderr)
        print(
            "\nRemediation:\n"
            "  1. uv sync --group data\n"
            "  2. Create a Kaggle API token; save to ~/.kaggle/kaggle.json (chmod 600)\n"
            "  3. Accept the IEEE-CIS competition rules on kaggle.com\n"
            "  4. Re-run: uv run python -m pipeline.fetch_data\n"
            "  (CI/offline: use the synthetic fixture — `--synthetic`.)",
            file=sys.stderr,
        )
        return 2

    _download_real(data_dir)
    tx_path, id_path = data_dir / RAW_TX, data_dir / RAW_ID
    if not tx_path.exists() or not id_path.exists():
        print("ERROR: expected files missing after download.", file=sys.stderr)
        return 3

    tx = pd.read_csv(tx_path)
    id_df = pd.read_csv(id_path)
    _sanity_check(tx, id_df)

    m = manifest_mod.build_manifest(tx, id_df, provenance="real", tx_path=tx_path, id_path=id_path)
    out = manifest_mod.write_manifest(m)
    print(f"Wrote {out} · {len(tx):,} transactions · fraud rate {tx[schema.TARGET].mean():.4%}")
    return 0


def build_synthetic(data_dir: Path, n: int = 5000, seed: int = 42) -> int:
    fixture_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    tx_path, id_path = write_fixture(fixture_dir, n=n, seed=seed)
    tx, id_df = pd.read_parquet(tx_path), pd.read_parquet(id_path)
    m = manifest_mod.build_manifest(
        tx,
        id_df,
        provenance="synthetic",
        generator={"module": "pipeline.synthetic", "n": n, "seed": seed},
    )
    out = manifest_mod.write_manifest(m)
    print(f"Wrote synthetic fixture + {out} (labelled synthetic).")
    return 0


def _sanity_check(tx: pd.DataFrame, id_df: pd.DataFrame) -> None:
    """Guardrails against a corrupt/partial download (not tuned to exact counts)."""
    assert schema.TARGET in tx.columns, "isFraud missing from transactions"
    assert tx[schema.ID_COL].is_unique, "duplicate TransactionID in transactions"
    assert 0 < tx[schema.TARGET].mean() < 0.2, "implausible fraud rate — check download"
    assert len(tx) > 100_000, "transaction count too low — partial download?"
    assert set(id_df[schema.ID_COL]).issubset(set(tx[schema.ID_COL])), "identity id mismatch"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch IEEE-CIS data + write manifest")
    parser.add_argument(
        "--synthetic", action="store_true", help="build the synthetic fixture instead"
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--n", type=int, default=5000, help="synthetic row count")
    parser.add_argument("--seed", type=int, default=42, help="synthetic seed")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if args.synthetic:
        return build_synthetic(data_dir, n=args.n, seed=args.seed)
    return fetch_real(data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
