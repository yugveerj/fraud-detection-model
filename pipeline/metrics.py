"""Evaluation metrics for rare-positive fraud detection (SPEC Section 0/3).

PR-AUC is primary (ROC-AUC is optimistic under heavy imbalance). We also report
recall at fixed false-positive-rate operating points, the Brier score, and a
reliability curve for calibration assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

# False-positive-rate operating points at which we report recall (fraction of fraud
# caught). Chosen to bracket a realistic review-capacity range.
DEFAULT_FPR_POINTS = (0.001, 0.005, 0.01, 0.05, 0.10)


@dataclass
class EvalResult:
    pr_auc: float
    roc_auc: float
    brier: float
    recall_at_fpr: dict[float, float] = field(default_factory=dict)
    n: int = 0
    n_pos: int = 0

    def to_row(self) -> dict:
        row = {
            "pr_auc": round(self.pr_auc, 5),
            "roc_auc": round(self.roc_auc, 5),
            "brier": round(self.brier, 6),
            "n": self.n,
            "n_pos": self.n_pos,
        }
        for fpr, rec in self.recall_at_fpr.items():
            row[f"recall@fpr={fpr:g}"] = round(rec, 4)
        return row


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr_points=DEFAULT_FPR_POINTS) -> dict:
    """Recall (TPR) at each target FPR, via interpolation on the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    out = {}
    for target in fpr_points:
        # np.interp needs increasing x; fpr from roc_curve is non-decreasing.
        out[target] = float(np.interp(target, fpr, tpr))
    return out


def evaluate(y_true, scores, fpr_points=DEFAULT_FPR_POINTS) -> EvalResult:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    return EvalResult(
        pr_auc=float(average_precision_score(y_true, scores)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        brier=float(brier_score_loss(y_true, scores)),
        recall_at_fpr=recall_at_fpr(y_true, scores, fpr_points),
        n=int(len(y_true)),
        n_pos=int(np.sum(y_true)),
    )


def reliability_curve(y_true, scores, n_bins: int = 10, strategy: str = "quantile"):
    """Reliability curve points: (mean predicted, observed frequency, bin count).

    Uses quantile bins by default so each bin holds a comparable number of points
    under heavy imbalance. Returns arrays aligned by bin.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(scores, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    # Guard against degenerate edges (all-equal scores).
    if len(edges) < 2:
        edges = np.array([scores.min(), scores.max() + 1e-9])
    idx = np.clip(np.digitize(scores, edges[1:-1]), 0, len(edges) - 2)
    mean_pred, obs_freq, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        if mask.sum() == 0:
            continue
        mean_pred.append(float(scores[mask].mean()))
        obs_freq.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(mean_pred), np.array(obs_freq), np.array(counts)
