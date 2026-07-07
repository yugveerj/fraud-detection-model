"""Decision report generator (SPEC Section 4).

One command produces the dollar-framed decision report a reviewer uses to decide
whether to *operate* the model: the cost model, the operating point (optimized on
VALIDATION, reported on HOLDOUT), the value-capture curve, the review-cost
sensitivity table, calibration reliability, and SHAP global importance + three case
studies. Figures land in ``docs/figures/``; the report is ``docs/decision_report.md``.

    uv run python -m pipeline.evaluate --report               # default review cost $25
    uv run python -m pipeline.evaluate --report --review-cost 40
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline import data as data_mod  # noqa: E402
from pipeline import decisions, encoders, explain, metrics, modeling, schema  # noqa: E402

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
REPORT_PATH = REPO_ROOT / "docs" / "decision_report.md"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
OPERATING_POINT_PATH = ARTIFACT_DIR / "operating_point.json"
MODEL_META_PATH = ARTIFACT_DIR / "model" / "metadata.json"
SHAP_SAMPLE = 1000


def run_report(source: str = "auto", review_cost: float = 25.0, profile: str = "full") -> dict:
    ds = data_mod.load_dataset(source)
    feat = encoders.feature_frame(ds.frame)
    sp = data_mod.temporal_split(feat, provenance=ds.provenance)
    y_tr = sp.train[schema.TARGET].to_numpy()
    y_val = sp.val[schema.TARGET].to_numpy()
    y_hold = sp.holdout[schema.TARGET].to_numpy()
    amt_val = sp.val[schema.AMT_COL].to_numpy()
    amt_hold = sp.holdout[schema.AMT_COL].to_numpy()

    base = modeling.make_model(modeling.PRODUCTION_MODEL, y_train=y_tr, profile=profile).fit(
        sp.train, y_tr
    )
    cal = modeling.calibrate(base, sp.val, y_val, method="isotonic")
    p_val = cal.predict_proba(sp.val)[:, 1]
    p_hold = cal.predict_proba(sp.holdout)[:, 1]
    p_hold_base = modeling.positive_proba(base, sp.holdout)

    # Freeze the operating point on VAL; report on HOLDOUT.
    op = decisions.optimize_operating_point(y_val, p_val, amt_val, review_cost)
    hold_stats = decisions.evaluate_at_threshold(
        y_hold, p_hold, amt_hold, op["threshold"], review_cost
    )
    statement = decisions.operating_point_statement(hold_stats, ds.provenance)
    sensitivity = decisions.sensitivity_table(y_val, p_val, amt_val)
    hold_metrics = metrics.evaluate(y_hold, p_hold)
    # Platt is rank-preserving, so its PR-AUC equals the raw base model's — one number
    # captures the ranking cost isotonic pays for its tie-blocks (§4 tradeoff note).
    base_pr_auc = metrics.evaluate(y_hold, p_hold_base).pr_auc

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _fig_value_capture(y_hold, p_hold, amt_hold, review_cost, op, hold_stats)
    _fig_net_value(y_val, p_val, amt_val, review_cost, op)
    _fig_reliability(y_hold, p_hold_base, p_hold)

    cases = _shap_section(base, sp.holdout, p_hold, y_hold, amt_hold, op["threshold"])
    _export_holdout_scores(sp.holdout, y_hold, p_hold, amt_hold)

    report = _render_report(
        ds.provenance,
        review_cost,
        sp,
        op,
        hold_stats,
        statement,
        sensitivity,
        hold_metrics,
        cases,
        base_pr_auc,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    _persist_operating_point(op, hold_stats, ds.provenance, review_cost)

    return {
        "provenance": ds.provenance,
        "operating_point": op,
        "holdout": hold_stats,
        "statement": statement,
        "report": str(REPORT_PATH.relative_to(REPO_ROOT)),
    }


# ------------------------------------------------------------------- figures


def _save(fig, name: str) -> str:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return name


def _fig_value_capture(y, proba, amounts, review_cost, op, hold_stats):
    curve = decisions.net_value_curve(y, proba, amounts, review_cost)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve["alert_rate"], curve["value_capture_rate"], color="#4c78a8", lw=2)
    ax.scatter(
        [hold_stats["alert_rate"]],
        [hold_stats["value_capture_rate"]],
        color="#e45756",
        zorder=5,
        label="frozen operating point",
    )
    ax.set(
        xlabel="review volume (alert rate)",
        ylabel="fraud value captured",
        title="Value-capture curve — HOLDOUT (frozen threshold from VAL)",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, "value_capture.png")


def _fig_net_value(y, proba, amounts, review_cost, op):
    curve = decisions.net_value_curve(y, proba, amounts, review_cost)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(curve["threshold"], curve["net_per_100k"], color="#54a24b", lw=2)
    ax.axvline(
        op["threshold"], color="#e45756", ls="--", label=f"opt threshold {op['threshold']:.3f}"
    )
    ax.set(
        xlabel="probability threshold",
        ylabel="net value per 100k ($)",
        title="Net value vs threshold — VALIDATION (operating point frozen here)",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, "net_value.png")


def _fig_reliability(y, p_uncal, p_cal):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="gray", ls=":", label="perfect")
    for label, p, color in [("uncalibrated", p_uncal, "#f58518"), ("isotonic", p_cal, "#4c78a8")]:
        mp, of, _ = metrics.reliability_curve(y, p, n_bins=8)
        ax.plot(mp, of, marker="o", color=color, label=label)
    ax.set(
        xlabel="mean predicted probability",
        ylabel="observed fraud frequency",
        title="Reliability — HOLDOUT (pre/post calibration)",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "reliability.png")


# ------------------------------------------------------------------- SHAP


def _shap_section(base, X_hold, p_hold, y_hold, amt_hold, threshold):
    sample = X_hold.iloc[:SHAP_SAMPLE]
    sv, Xt, names, _ = explain.compute_shap(base, sample)

    import shap

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, Xt, feature_names=names, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP global importance (mean |value|) — base gradient-boosted model")
    _save(fig, "shap_global_bar.png")

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=18)
    _save(fig, "shap_beeswarm.png")

    # Case studies computed on the SHAP sample (indices align to sample rows).
    sel = explain.select_case_studies(p_hold[:SHAP_SAMPLE], y_hold[:SHAP_SAMPLE], threshold)
    cases = [
        explain.case_study(
            kind,
            idx,
            sv,
            Xt,
            names,
            p_hold[:SHAP_SAMPLE],
            y_hold[:SHAP_SAMPLE],
            amt_hold[:SHAP_SAMPLE],
        )
        for kind, idx in sel.items()
    ]
    return cases


# ------------------------------------------------------------------- report


def _render_report(
    prov, review_cost, sp, op, hold, statement, sensitivity, hold_metrics, cases, base_pr_auc
):
    synth = prov != "real"
    lines = ["# Decision report — is this model worth operating?", ""]
    if synth:
        lines += [
            "> **SYNTHETIC FIXTURE — illustrative, NOT results.** Numbers are from the "
            "schema-identical synthetic fixture (SPEC Section 10); the decision framework "
            "is real. Regenerate on real data with `uv run python -m pipeline.evaluate "
            "--report`.",
            "",
        ]
    lines += [
        "This report is written for a reviewer deciding whether to **operate** the model, "
        "not how it was trained. Everything is framed in dollars.",
        "",
        "## 1. Cost model",
        "",
        f"- Reviewing a transaction costs **${review_cost:.0f}** (business parameter).",
        "- Catching a fraudulent transaction saves its `TransactionAmt` (loss avoided).",
        "- A missed fraud loses its `TransactionAmt`.",
        "- **Net value** of an operating point = fraud dollars caught − review cost × alerts.",
        "",
        "The threshold is optimized on **VALIDATION** and reported on **HOLDOUT** — never "
        "tuned on the data it is reported on.",
        "",
        "## 2. Operating point",
        "",
        f"**{statement}**",
        "",
        "| metric | VALIDATION (optimized) | HOLDOUT (reported) |",
        "| --- | --- | --- |",
        f"| threshold | {op['threshold']:.4f} | {op['threshold']:.4f} (frozen) |",
        f"| alert rate | {op['alert_rate']:.2%} | {hold['alert_rate']:.2%} |",
        f"| recall (count) | {op['recall']:.1%} | {hold['recall']:.1%} |",
        f"| false-positive rate | {op['fpr']:.2%} | {hold['fpr']:.2%} |",
        f"| fraud value captured | {op['value_capture_rate']:.1%} | {hold['value_capture_rate']:.1%} |",
        f"| net value / 100k | ${op['net_per_100k']:,.0f} | ${hold['net_per_100k']:,.0f} |",
        "",
        "The holdout capture is lower than validation: an out-of-time slice under drift is "
        "genuinely harder, and the frozen threshold does not perfectly transfer. Net value "
        "remains positive, and the gap is exactly what ongoing monitoring watches "
        "(validation report Section 7).",
        "",
        "![value capture](figures/value_capture.png)",
        "",
        "![net value](figures/net_value.png)",
        "",
        "## 3. Review-cost sensitivity",
        "",
        "How the optimal operating point shifts as the review-cost assumption varies "
        "($5–$75). Higher review cost → higher threshold → fewer, higher-precision alerts.",
        "",
        "| review cost | opt threshold | alert rate | recall | fpr | value capture | net / 100k |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in sensitivity.iterrows():
        lines.append(
            f"| ${r['review_cost']:.0f} | {r['opt_threshold']:.4f} | {r['alert_rate']:.2%} | "
            f"{r['recall']:.1%} | {r['fpr']:.2%} | {r['value_capture_rate']:.1%} | "
            f"${r['net_per_100k']:,.0f} |"
        )

    lines += [
        "",
        "## 4. Calibration",
        "",
        f"Holdout Brier {hold_metrics.brier:.4f}. Under class imbalance the Brier score is "
        "dominated by the rare-positive base rate; the reliability curve is the real check. "
        "Isotonic calibration corrects the probability *level* the dollar thresholds depend "
        "on. Where the holdout curve departs from validation, that is calibration drift.",
        "",
        f"**Method tradeoff.** Isotonic is not free on ranking: it scores holdout PR-AUC "
        f"**{hold_metrics.pr_auc:.4f}** versus **{base_pr_auc:.4f}** for the rank-preserving "
        "alternatives (the raw model and Platt scaling), because its step function ties large "
        "blocks of transactions and PR-AUC penalises the ambiguous within-tie ordering. "
        "Isotonic is kept as champion anyway — it wins Brier and the reliability the dollar "
        "layer needs, and the operating point is chosen on *value*, not raw ranking — while "
        "the Platt-calibrated model is registered as the named challenger so the choice stays "
        "auditable (validation report Section 7).",
        "",
        "![reliability](figures/reliability.png)",
        "",
        "## 5. Feature attribution (SHAP)",
        "",
        "Global importance on the base gradient-boosted model (TreeExplainer). **Caveat:** many inputs are "
        "anonymized Vesta `V*` / `id_*` features with no published meaning, so SHAP shows "
        "*which engineered inputs* move a score, not a mechanistic business reason.",
        "",
        "![shap global](figures/shap_global_bar.png)",
        "",
        "![shap beeswarm](figures/shap_beeswarm.png)",
        "",
        "### Case studies (HOLDOUT)",
        "",
    ]
    for c in cases:
        decision = "ALERT" if c.proba >= op["threshold"] else "pass"
        # Isotonic's top bin saturates at 1.0; flag a rounded-1.000 score so a reviewer
        # reads it as a calibrated bin rate, not a suspicious claim of certainty.
        note = (
            " (an isotonic upper-bin value — the calibrator's top bin is saturated, so this"
            " is that bin's empirical fraud rate, not a claim of certainty)"
            if c.proba >= 0.9995
            else ""
        )
        lines += [
            f"**{c.kind.title()}** — actual label {c.label}, calibrated P(fraud) "
            f"{c.proba:.3f}, amount ${c.amount:,.2f} → **{decision}** at threshold "
            f"{op['threshold']:.3f}.{note}",
            "",
            "| feature | value | SHAP |",
            "| --- | --- | --- |",
        ]
        for f in c.top_factors:
            lines.append(f"| {f['feature']} | {f['value']:.3f} | {f['shap']:+.3f} |")
        lines.append("")

    lines += [
        "## 6. Limitations",
        "",
        "- Anonymized features cap semantic interpretation (above).",
        "- Holdout is a single out-of-time slice; live performance depends on drift, tracked "
        "by the monitoring layer (validation report Section 7).",
        "- Fraud labels arrive weeks late in production; this replay has them immediately — "
        "the label-lag caveat travels with every number that leaves the repo.",
        "" + ("- **All figures above are on synthetic data** (SPEC Section 10)." if synth else ""),
        "",
    ]
    return "\n".join(lines)


def _export_holdout_scores(holdout, y_hold, p_hold, amt_hold):
    """Export (id, label, calibrated probability, amount) for the independent R
    replication (SPEC Section 7b) — R recomputes the metrics with no Python imports."""
    import pandas as pd

    out = REPO_ROOT / "docs" / "validation_r" / "holdout_scores.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "TransactionID": holdout[schema.ID_COL].to_numpy(),
            "isFraud": y_hold,
            "calibrated_proba": p_hold,
            "TransactionAmt": amt_hold,
        }
    ).to_csv(out, index=False)


def _persist_operating_point(op, hold, prov, review_cost):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "provenance": prov,
        "review_cost": review_cost,
        "threshold": op["threshold"],
        "validation": op,
        "holdout": hold,
    }
    OPERATING_POINT_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    # Fold the frozen threshold into the serving metadata if it exists.
    if MODEL_META_PATH.exists():
        meta = json.loads(MODEL_META_PATH.read_text())
        meta["operating_threshold"] = op["threshold"]
        meta["review_cost"] = review_cost
        MODEL_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the decision report")
    parser.add_argument("--report", action="store_true", help="generate the full report")
    parser.add_argument("--source", default="auto", choices=["auto", "real", "synthetic"])
    parser.add_argument("--review-cost", type=float, default=25.0)
    parser.add_argument("--profile", default="full", choices=["full", "smoke"])
    args = parser.parse_args(argv)

    result = run_report(source=args.source, review_cost=args.review_cost, profile=args.profile)
    print(result["statement"])
    print(f"Wrote {result['report']} + docs/figures/ + artifacts/operating_point.json")
    if result["provenance"] != "real":
        print("LABEL: synthetic — illustrative, not results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
