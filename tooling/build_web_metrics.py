"""Assemble ``web/metrics.json`` for the demo's results card (IMPROVEMENTS_2 H1).

The four headline numbers on the demo page are read from this file at runtime, NOT
hardcoded in the HTML where they could silently drift from the model. Every value is
sourced from a committed artifact:

- fraud value captured + net value + threshold  ← ``artifacts/operating_point.json``
- transactions analyzed                          ← ``docs/data_manifest.md``
- leakage inflation (random-CV vs temporal)      ← ``docs/experiments.md`` (leakage table)
- champion model + registry version              ← ``artifacts/model/metadata.json``

``tests/test_web_metrics.py`` asserts the emitted file still matches those sources, so a
model change that isn't reflected here fails CI rather than shipping a stale number.

    uv run python -m tooling.build_web_metrics
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OP_PATH = REPO_ROOT / "artifacts" / "operating_point.json"
META_PATH = REPO_ROOT / "artifacts" / "model" / "metadata.json"
MANIFEST_PATH = REPO_ROOT / "docs" / "data_manifest.md"
EXPERIMENTS_PATH = REPO_ROOT / "docs" / "experiments.md"
OUT_PATH = REPO_ROOT / "web" / "metrics.json"


def _manifest_total_rows(text: str) -> int:
    """First ``| rows | N |`` row in the manifest = the full transaction count."""
    m = re.search(r"\|\s*rows\s*\|\s*([\d,]+)\s*\|", text)
    if not m:
        raise ValueError("could not find total 'rows' in data_manifest.md")
    return int(m.group(1).replace(",", ""))


def _leakage_inflation_pct(text: str, model: str) -> float:
    """Signed inflation % for ``model`` in the leakage table (e.g. '+23.2%')."""
    m = re.search(rf"\|\s*{re.escape(model)}\s*\|[^|]*\|[^|]*\|[^|]*\|\s*([+-]?\d+\.\d+)%", text)
    if not m:
        raise ValueError(f"could not find leakage inflation for {model!r} in experiments.md")
    return float(m.group(1))


def build() -> dict:
    op = json.loads(OP_PATH.read_text())
    meta = json.loads(META_PATH.read_text())
    hold = op["holdout"]

    total_rows = _manifest_total_rows(MANIFEST_PATH.read_text())
    model = "LightGBM" if "lightgbm" in meta["model_type"].lower() else meta["model_type"]
    inflation = _leakage_inflation_pct(EXPERIMENTS_PATH.read_text(), model.lower())

    value_capture = hold["value_capture_rate"]
    net = hold["net_per_100k"]

    return {
        "model": model,
        "registry_version": int(meta["version"]),
        "calibration": meta.get("calibration", "isotonic"),
        "provenance": op.get("provenance", meta.get("provenance", "real")),
        "threshold": round(op["threshold"], 4),
        "review_cost": op.get("review_cost", 25.0),
        "headline": [
            {
                "value": f"{total_rows:,}",
                "label": "transactions analyzed",
                "raw": total_rows,
            },
            {
                "value": f"+{inflation:.1f}%" if inflation > 0 else f"{inflation:.1f}%",
                "label": "PR-AUC inflation caught (random-CV vs temporal)",
                "raw": inflation,
            },
            {
                "value": f"{value_capture * 100:.1f}%",
                "label": "fraud value captured (holdout)",
                "raw": value_capture,
            },
            {
                "value": f"≈${net / 1000:.0f}K",
                "label": "net value / 100k transactions",
                "raw": net,
            },
        ],
        "disclaimer": "Non-production demonstration on the public IEEE-CIS dataset.",
    }


def main() -> int:
    metrics = build()
    OUT_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}:")
    for h in metrics["headline"]:
        print(f"  {h['value']:>10}  {h['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
