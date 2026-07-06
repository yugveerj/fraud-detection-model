"""Data manifest: row counts, class balance, checksums (SPEC Section 1).

Written at fetch time to ``docs/data_manifest.md``. For real data it records file
SHA-256 checksums so the exact input is pinned; for the synthetic fixture it records
the generator parameters instead and labels the manifest as synthetic.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from pipeline import data as data_mod
from pipeline import schema

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "docs" / "data_manifest.md"


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_manifest(
    tx: pd.DataFrame,
    id_df: pd.DataFrame,
    provenance: str,
    tx_path: Path | None = None,
    id_path: Path | None = None,
    generator: dict | None = None,
) -> dict:
    manifest: dict = {
        "provenance": provenance,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "transaction": {
            "rows": int(len(tx)),
            "columns": int(tx.shape[1]),
            "fraud_count": int(tx[schema.TARGET].sum()),
            "fraud_rate": round(float(tx[schema.TARGET].mean()), 6),
            "dt_min": int(tx[schema.TIME_COL].min()),
            "dt_max": int(tx[schema.TIME_COL].max()),
            "amt_min": round(float(tx[schema.AMT_COL].min()), 2),
            "amt_max": round(float(tx[schema.AMT_COL].max()), 2),
        },
        "identity": {
            "rows": int(len(id_df)),
            "columns": int(id_df.shape[1]),
            "coverage": round(float(len(id_df) / len(tx)), 4),
        },
        "files": [],
        "generator": generator,
    }
    for label, path in (("train_transaction", tx_path), ("train_identity", id_path)):
        if path is not None and Path(path).exists():
            p = Path(path)
            manifest["files"].append(
                {
                    "name": p.name,
                    "label": label,
                    "size_bytes": p.stat().st_size,
                    "sha256": file_sha256(p),
                }
            )

    # Temporal split preview.
    sp = data_mod.temporal_split(data_mod._merge(tx, id_df), provenance=provenance)
    manifest["splits"] = sp.summary().to_dict(orient="records")
    manifest["split_cuts"] = {
        "cut_train_val": sp.cut_train_val,
        "cut_val_holdout": sp.cut_val_holdout,
    }
    return manifest


def render_manifest_md(m: dict) -> str:
    tx, idn = m["transaction"], m["identity"]
    lines = ["# Data manifest", ""]
    if m["provenance"] != "real":
        lines += [
            "> **SYNTHETIC FIXTURE — NOT REAL DATA.** These numbers describe the "
            "schema-identical synthetic fixture used for tests/CI (SPEC Section 10). "
            "They are regenerated from the real IEEE-CIS competition data at fetch "
            "time (`uv run python -m pipeline.fetch_data`) and must never be cited as "
            "results.",
            "",
        ]
    lines += [
        f"- **Provenance:** {m['provenance']}",
        f"- **Generated (UTC):** {m['generated_at']}",
        "",
        "## Transactions",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| rows | {tx['rows']:,} |",
        f"| columns | {tx['columns']} |",
        f"| fraud count | {tx['fraud_count']:,} |",
        f"| fraud rate | {tx['fraud_rate']:.4%} |",
        f"| TransactionDT range | {tx['dt_min']:,} – {tx['dt_max']:,} |",
        f"| TransactionAmt range | {tx['amt_min']:,} – {tx['amt_max']:,} |",
        "",
        "## Identity (left-joined subset)",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| rows | {idn['rows']:,} |",
        f"| columns | {idn['columns']} |",
        f"| coverage of transactions | {idn['coverage']:.2%} |",
        "",
        "## Temporal segmentation (by TransactionDT)",
        "",
        f"Cut points: train|val = `{m['split_cuts']['cut_train_val']}`, "
        f"val|holdout = `{m['split_cuts']['cut_val_holdout']}`.",
        "",
        "| split | rows | frac | dt_min | dt_max | fraud_rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in m["splits"]:
        lines.append(
            f"| {s['split']} | {s['rows']:,} | {s['frac']} | {s['dt_min']} | "
            f"{s['dt_max']} | {s['fraud_rate']} |"
        )
    lines += ["", "## Integrity"]
    if m["files"]:
        lines += ["", "| file | size (bytes) | sha256 |", "| --- | --- | --- |"]
        for f in m["files"]:
            lines.append(f"| {f['name']} | {f['size_bytes']:,} | `{f['sha256']}` |")
    elif m.get("generator"):
        g = m["generator"]
        lines += [
            "",
            "No raw file checksums (synthetic fixture). Generator parameters:",
            "",
            f"- module: `{g.get('module')}`  ·  n = {g.get('n')}  ·  seed = {g.get('seed')}",
        ]
    lines.append("")
    return "\n".join(lines)


def write_manifest(m: dict, path: Path = MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_manifest_md(m), encoding="utf-8")
    return path
