"""Dollar-framed decision layer (SPEC Section 4).

The model outputs a calibrated fraud probability; the decision is whether to *review*
a transaction. The cost model:

* Reviewing a transaction costs a fixed ``review_cost`` (a business parameter,
  default $25).
* Catching a fraudulent transaction saves its ``TransactionAmt`` (the loss avoided).
* A missed fraud loses its ``TransactionAmt``.

**Net value** of an operating threshold, relative to reviewing nothing, is therefore

    net = (fraud $ caught by alerts) - review_cost * (number of alerts)

We optimize expected net value on **VALIDATION**, freeze the threshold, and report on
**HOLDOUT** — never tuning the threshold on the data we report. Amounts are the real
per-transaction ``TransactionAmt``; the framing is deliberately dollars, not accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def net_value_curve(
    y: np.ndarray,
    proba: np.ndarray,
    amounts: np.ndarray,
    review_cost: float,
) -> pd.DataFrame:
    """Full threshold trade-off, computed in one sort (O(n log n)).

    Each row corresponds to a candidate threshold = the probability of the lowest-ranked
    alerted transaction. Columns: threshold, alerts, alert_rate, recall, fpr,
    fraud_value_caught, fraud_value_total, value_capture_rate, review_cost_total,
    net_value, net_per_100k.
    """
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    n = len(y)

    order = np.argsort(-proba, kind="stable")  # highest probability first
    p_s, y_s, a_s = proba[order], y[order], amounts[order]

    alerts = np.arange(1, n + 1)
    fraud_value_caught = np.cumsum(y_s * a_s)
    fraud_caught_count = np.cumsum(y_s)
    false_positives = np.cumsum(1 - y_s)

    total_fraud_value = float(np.sum(y * amounts))
    total_pos = int(np.sum(y))
    total_neg = int(n - total_pos)

    review_cost_total = review_cost * alerts
    net_value = fraud_value_caught - review_cost_total

    return pd.DataFrame(
        {
            "threshold": p_s,
            "alerts": alerts,
            "alert_rate": alerts / n,
            "recall": fraud_caught_count / total_pos if total_pos else np.zeros(n),
            "fpr": false_positives / total_neg if total_neg else np.zeros(n),
            "fraud_value_caught": fraud_value_caught,
            "fraud_value_total": total_fraud_value,
            "value_capture_rate": (
                fraud_value_caught / total_fraud_value if total_fraud_value else np.zeros(n)
            ),
            "review_cost_total": review_cost_total,
            "net_value": net_value,
            "net_per_100k": net_value / n * 100_000,
        }
    )


def optimize_operating_point(
    y: np.ndarray,
    proba: np.ndarray,
    amounts: np.ndarray,
    review_cost: float = 25.0,
) -> dict:
    """Threshold maximizing net value (on VALIDATION). Returns the operating point."""
    curve = net_value_curve(y, proba, amounts, review_cost)
    best = curve.loc[curve["net_value"].idxmax()]
    return _point(best, review_cost, len(y))


def evaluate_at_threshold(
    y: np.ndarray,
    proba: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    review_cost: float = 25.0,
) -> dict:
    """Apply a FROZEN threshold (from VAL) and report the operating point (on HOLDOUT)."""
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    amounts = np.asarray(amounts, dtype=float)
    n = len(y)

    alert = proba >= threshold
    total_fraud_value = float(np.sum(y * amounts))
    total_pos = int(np.sum(y))
    total_neg = int(n - total_pos)

    fraud_value_caught = float(np.sum(amounts[alert & (y == 1)]))
    n_alerts = int(alert.sum())
    fp = int(np.sum(alert & (y == 0)))
    net_value = fraud_value_caught - review_cost * n_alerts

    return {
        "threshold": float(threshold),
        "review_cost": float(review_cost),
        "n": n,
        "alerts": n_alerts,
        "alert_rate": n_alerts / n if n else 0.0,
        "recall": (int(np.sum(alert & (y == 1))) / total_pos) if total_pos else 0.0,
        "fpr": (fp / total_neg) if total_neg else 0.0,
        "fraud_value_caught": fraud_value_caught,
        "fraud_value_total": total_fraud_value,
        "value_capture_rate": (fraud_value_caught / total_fraud_value)
        if total_fraud_value
        else 0.0,
        "net_value": net_value,
        "net_per_100k": net_value / n * 100_000 if n else 0.0,
    }


def sensitivity_table(
    y: np.ndarray,
    proba: np.ndarray,
    amounts: np.ndarray,
    review_costs=(5, 15, 25, 50, 75),
) -> pd.DataFrame:
    """How the *optimal* operating point shifts as the review-cost assumption varies."""
    rows = []
    for rc in review_costs:
        op = optimize_operating_point(y, proba, amounts, review_cost=float(rc))
        rows.append(
            {
                "review_cost": rc,
                "opt_threshold": round(op["threshold"], 4),
                "alert_rate": round(op["alert_rate"], 4),
                "recall": round(op["recall"], 4),
                "fpr": round(op["fpr"], 5),
                "value_capture_rate": round(op["value_capture_rate"], 4),
                "net_per_100k": round(op["net_per_100k"], 2),
            }
        )
    return pd.DataFrame(rows)


def operating_point_statement(stats: dict, provenance: str = "synthetic") -> str:
    """The §0.4 statement: value captured, FPR, and net dollars per 100k transactions."""
    label = "" if provenance == "real" else " (SYNTHETIC — illustrative)"
    return (
        f"At the frozen operating point (threshold {stats['threshold']:.4f}, "
        f"review cost ${stats['review_cost']:.0f}), the model captures "
        f"{stats['value_capture_rate']:.1%} of fraud value at a "
        f"{stats['fpr']:.2%} false-positive rate, for approximately "
        f"${stats['net_per_100k']:,.0f} net value per 100k transactions{label}."
    )


def _point(row: pd.Series, review_cost: float, n: int) -> dict:
    return {
        "threshold": float(row["threshold"]),
        "review_cost": float(review_cost),
        "n": n,
        "alerts": int(row["alerts"]),
        "alert_rate": float(row["alert_rate"]),
        "recall": float(row["recall"]),
        "fpr": float(row["fpr"]),
        "fraud_value_caught": float(row["fraud_value_caught"]),
        "fraud_value_total": float(row["fraud_value_total"]),
        "value_capture_rate": float(row["value_capture_rate"]),
        "net_value": float(row["net_value"]),
        "net_per_100k": float(row["net_per_100k"]),
    }
