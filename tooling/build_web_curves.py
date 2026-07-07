"""Assemble ``web/curves/*.json`` — the series behind the demo's three charts.

Like ``build_web_metrics`` (and guarded the same way by ``tests/test_web_curves.py``),
nothing is hardcoded: the numbers come from committed artifacts, so a retrain that
shifts a curve fails CI instead of shipping a stale chart.

- ``net_value.json``   ← ``artifacts/curves/net_value.json``   (pipeline.evaluate; validation)
- ``reliability.json`` ← ``artifacts/curves/reliability.json`` (pipeline.evaluate; holdout)
- ``leakage.json``     ← the leakage table in ``docs/experiments.md``

    uv run python -m tooling.build_web_curves
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CURVES_DIR = REPO_ROOT / "artifacts" / "curves"
EXPERIMENTS_PATH = REPO_ROOT / "docs" / "experiments.md"
OUT_DIR = REPO_ROOT / "web" / "curves"

# Leakage table row order + display labels (the temporal-vs-random-CV anti-pattern).
LEAKAGE_MODELS = [("logreg", "Logistic reg."), ("xgboost", "XGBoost"), ("lightgbm", "LightGBM")]


def leakage() -> dict:
    """Parse the experiments.md leakage table: model | temporal | random-CV | Δ | inflation%."""
    text = EXPERIMENTS_PATH.read_text()
    rows = []
    for key, label in LEAKAGE_MODELS:
        m = re.search(
            rf"\|\s*{key}\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[+-]?[\d.]+\s*\|\s*([+-]?\d+\.\d+)%",
            text,
        )
        if not m:
            raise ValueError(f"leakage row for {key!r} not found in experiments.md")
        rows.append(
            {
                "model": label,
                "temporal": float(m.group(1)),
                "random_cv": float(m.group(2)),
                "inflation": float(m.group(3)),
            }
        )
    return {"metric": "PR-AUC", "rows": rows}


def build() -> dict:
    return {
        "net_value": json.loads((CURVES_DIR / "net_value.json").read_text()),
        "reliability": json.loads((CURVES_DIR / "reliability.json").read_text()),
        "leakage": leakage(),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curves = build()
    for name, data in curves.items():
        (OUT_DIR / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    op = curves["net_value"]["op"]
    leak = "  ".join(f"{r['model']} {r['inflation']:+.1f}%" for r in curves["leakage"]["rows"])
    print("Wrote web/curves/{net_value,reliability,leakage}.json")
    print(
        f"  net_value: {len(curves['net_value']['series'])} pts, op val ${op['net_val']:,} / holdout ${op['net_holdout']:,}"
    )
    print(f"  reliability: {len(curves['reliability']['isotonic'])} isotonic bins")
    print(f"  leakage: {leak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
