"""Replayed drift-monitoring engine (SPEC Section 6).

Each run replays one HOLDOUT slice as a simulated week: score it with the deployed
model, measure data/prediction drift (Evidently) + explicit PSI, and performance on
the (immediately available, in replay) labels — carrying the label-lag caveat. Breaches
(PSI > 0.2, Evidently dataset drift, or value-capture degradation > 10% vs baseline)
are surfaced so the cron can open a GitHub Issue.

**This is a replayed simulation on a static public dataset** — that banner rides on
every page and every number.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import data as data_mod
from pipeline import decisions, encoders, metrics, modeling, schema

warnings.filterwarnings("ignore")

DEFAULT_BATCHES = 10  # simulated weeks over the holdout stream (configurable)
VALUE_CAPTURE_DEGRADE = 0.10  # >10% relative drop vs baseline = performance breach
TOP_K_MONITORED = 20  # PSI/Evidently over the top-K most important DENSE numeric features
DENSE_THRESHOLD = 0.30  # only monitor features <30% missing (sparse block-NaN => small-N noise)
MIN_FRAUD_FOR_PERF = 4  # don't flag value-capture drift on batches with too few frauds
# Engineered calendar-position transforms excluded from drift PSI: they encode *when* a
# batch falls on the clock, not the transaction population, so a forward-replayed batch is
# trivially "shifted" against the multi-month reference (see _monitored_columns, D-013).
_CALENDAR_POSITION_FEATURES = frozenset({"dt_day", "dt_dow"})
BANNER = "Replayed simulation on a static public dataset — not live production monitoring."


@dataclass
class MonitorContext:
    reference: pd.DataFrame
    holdout: pd.DataFrame
    calibrated: object
    base: object
    threshold: float
    review_cost: float
    provenance: str
    n_batches: int
    numeric_cols: list[str]
    categorical_cols: list[str]
    baseline_value_capture: float
    reference_score: np.ndarray = field(default_factory=lambda: np.array([]))


def prepare(
    source: str = "auto",
    profile: str = "full",
    review_cost: float = 25.0,
    n_batches: int = DEFAULT_BATCHES,
) -> MonitorContext:
    ds = data_mod.load_dataset(source)
    feat = encoders.feature_frame(ds.frame)
    sp = data_mod.temporal_split(feat, provenance=ds.provenance)
    y_tr = sp.train[schema.TARGET].to_numpy()
    y_val = sp.val[schema.TARGET].to_numpy()

    base = modeling.make_model(modeling.PRODUCTION_MODEL, y_train=y_tr, profile=profile).fit(
        sp.train, y_tr
    )
    cal = modeling.calibrate(base, sp.val, y_val, method="isotonic")
    p_val = cal.predict_proba(sp.val)[:, 1]
    op = decisions.optimize_operating_point(
        y_val, p_val, sp.val[schema.AMT_COL].to_numpy(), review_cost
    )

    reference = pd.concat([sp.train, sp.val], ignore_index=True)
    numeric, categorical = _monitored_columns(
        base, encoders.model_feature_columns(sp.train), reference
    )
    # Baseline for the value-capture guard = the model's *out-of-time* value capture on the
    # full holdout at the frozen threshold, NOT the VAL-optimal rate. The VAL-optimal (~61%)
    # is the designed rate; the deployed model runs at the characterized out-of-time level
    # (~48%), so comparing weeks against VAL would breach every week on the known,
    # documented gap (validation report §4.3). Anchoring to the out-of-time norm makes the
    # guard fire on *new* degradation instead — the signal §7 escalates to retraining. (D-013)
    p_hold_all = cal.predict_proba(sp.holdout)[:, 1]
    baseline_vc = decisions.evaluate_at_threshold(
        sp.holdout[schema.TARGET].to_numpy(),
        p_hold_all,
        sp.holdout[schema.AMT_COL].to_numpy(),
        op["threshold"],
        review_cost,
    )["value_capture_rate"]
    return MonitorContext(
        reference=reference,
        holdout=sp.holdout,
        calibrated=cal,
        base=base,
        threshold=op["threshold"],
        review_cost=review_cost,
        provenance=ds.provenance,
        n_batches=n_batches,
        numeric_cols=numeric,
        categorical_cols=categorical,
        baseline_value_capture=baseline_vc,
        reference_score=cal.predict_proba(reference)[:, 1],
    )


def _monitored_columns(base, feature_cols, reference) -> tuple[list[str], list[str]]:
    """Top-K most-important DENSE numeric features + dense model categoricals.

    Sparse (block-NaN) V/id features produce PSI noise at small batch sizes, so
    monitoring is restricted to features with <DENSE_THRESHOLD missingness in the
    reference — the meaningful, stable input signals (amount, C*, ...). Causal
    aggregates (ent_*) are excluded (they drift structurally as history accumulates).
    """
    numeric_all, categorical = feature_cols
    prep = base.named_steps["prep"]
    est = base.named_steps["est"]
    names = list(prep.ct_.get_feature_names_out())
    order = np.argsort(-np.asarray(est.feature_importances_))
    ranked = [names[i] for i in order]
    numeric_set = set(numeric_all)
    miss = reference.isna().mean()
    top_numeric = [
        c
        for c in ranked
        # Exclude features whose PSI reflects *when the batch is* rather than drift in the
        # transaction population: causal aggregates (ent_*) grow with accumulating history,
        # and the calendar-position transforms (dt_day = absolute day index, dt_dow =
        # day-of-week) are trivially "shifted" by construction — a replay batch spans ~2
        # real days, so it never samples the day axis representatively against the
        # multi-month reference and would breach every week. Intra-day timing (dt_hour and
        # its cyclic encodings, dt_is_night) is well-sampled even in a short batch and is
        # kept, as are all real transaction features + the score distribution. (D-013)
        if c in numeric_set
        and not c.startswith("ent_")
        and c not in _CALENDAR_POSITION_FEATURES
        and miss.get(c, 0.0) < DENSE_THRESHOLD
    ][:TOP_K_MONITORED]
    dense_cat = [c for c in categorical if miss.get(c, 0.0) < DENSE_THRESHOLD]
    return top_numeric, dense_cat


def batches(ctx: MonitorContext) -> list[pd.DataFrame]:
    """Split the holdout into n_batches time-ordered slices (each a simulated week)."""
    holdout = ctx.holdout.sort_values(schema.TIME_COL, kind="stable")
    return [b for b in np.array_split(holdout, ctx.n_batches) if len(b)]


def run_batch(ctx: MonitorContext, index: int, out_dir: Path) -> dict:
    """Score + assess one simulated week; write its HTML. Returns a result record."""
    batch = batches(ctx)[index]
    y = batch[schema.TARGET].to_numpy()
    proba = ctx.calibrated.predict_proba(batch)[:, 1]
    amt = batch[schema.AMT_COL].to_numpy()

    # --- Explicit PSI (top-K features + score distribution). ---
    from monitoring import psi as psi_mod

    ref = ctx.reference.copy()
    cur = batch.copy()
    ref["__score__"] = ctx.reference_score
    cur["__score__"] = proba
    psi_table = psi_mod.psi_table(ref, cur, ctx.numeric_cols + ["__score__"], ctx.categorical_cols)

    # --- Evidently drift report (best-effort; PSI is the load-bearing signal). ---
    week_dir = out_dir / f"week-{index:02d}"
    week_dir.mkdir(parents=True, exist_ok=True)
    evidently_drift = _evidently_report(ctx, batch, week_dir / "evidently.html")

    # --- Performance on labels (replay has them immediately; production lags weeks). ---
    ev = metrics.evaluate(y, proba)
    hold_stats = decisions.evaluate_at_threshold(y, proba, amt, ctx.threshold, ctx.review_cost)
    value_capture = hold_stats["value_capture_rate"]
    vc_degradation = (
        (ctx.baseline_value_capture - value_capture) / ctx.baseline_value_capture
        if ctx.baseline_value_capture
        else 0.0
    )

    # --- Breaches. ---
    psi_breaches = psi_table[psi_table["breach"]]["feature"].tolist()
    breaches = []
    if psi_breaches:
        breaches.append(f"PSI>0.2 on: {', '.join(psi_breaches)}")
    if evidently_drift:
        breaches.append("Evidently flagged dataset drift")
    if vc_degradation > VALUE_CAPTURE_DEGRADE and int(y.sum()) >= MIN_FRAUD_FOR_PERF:
        breaches.append(f"value-capture down {vc_degradation:.0%} vs out-of-time baseline")

    result = {
        "index": index,
        "dt_min": int(batch[schema.TIME_COL].min()),
        "dt_max": int(batch[schema.TIME_COL].max()),
        "rows": int(len(batch)),
        "fraud_rate": round(float(y.mean()), 5),
        "pr_auc": round(ev.pr_auc, 4),
        "value_capture": round(value_capture, 4),
        "vc_degradation": round(vc_degradation, 4),
        "max_psi": round(float(psi_table["psi"].max()), 4) if len(psi_table) else 0.0,
        "evidently_drift": bool(evidently_drift),
        "breaches": breaches,
    }
    (week_dir / "report.html").write_text(
        _render_week_html(ctx, result, psi_table, ev), encoding="utf-8"
    )
    return result


def _evidently_report(ctx: MonitorContext, batch: pd.DataFrame, out_path: Path) -> bool:
    """Write the Evidently drift report; return whether dataset drift was flagged."""
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset

        cols = [c for c in ctx.numeric_cols if c != "__score__"]
        dd = DataDefinition(numerical_columns=cols, categorical_columns=ctx.categorical_cols)
        ref = Dataset.from_pandas(ctx.reference[cols + ctx.categorical_cols], data_definition=dd)
        cur = Dataset.from_pandas(batch[cols + ctx.categorical_cols], data_definition=dd)
        report = Report(metrics=[DataDriftPreset()])
        res = report.run(current_data=cur, reference_data=ref)
        res.save_html(str(out_path))
        _inject_banner(out_path)  # honesty label travels with this standalone drift page
        return _extract_dataset_drift(res.dict())
    except Exception as exc:  # noqa: BLE001 - Evidently is a best-effort enhancement
        out_path.write_text(f"<p>Evidently report unavailable: {exc}</p>", encoding="utf-8")
        return False


def _inject_banner(path: Path) -> None:
    """Prepend the replay-simulation honesty banner to a saved Evidently HTML page, so the
    standalone report (linked one click from the week page) can't be read without the label."""
    try:
        html = path.read_text(encoding="utf-8")
    except OSError:
        return
    if BANNER in html:
        return
    banner = (
        '<div style="background:#fff3cd;border:1px solid #ffca2c;border-radius:6px;'
        "padding:.5rem .8rem;margin:.5rem;font:14px -apple-system,BlinkMacSystemFont,"
        f'sans-serif;color:#4d3800">{BANNER}</div>'
    )
    if re.search(r"<body[^>]*>", html):
        html = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + banner, html, count=1)
    else:
        html = banner + html
    path.write_text(html, encoding="utf-8")


def _extract_dataset_drift(result: dict) -> bool:
    """Dataset drift = Evidently's DriftedColumnsCount share >= its drift_share threshold."""
    for metric in result.get("metrics", []):
        if str(metric.get("metric_name", "")).startswith("DriftedColumnsCount"):
            share = metric.get("value", {}).get("share")
            threshold = metric.get("config", {}).get("drift_share", 0.5)
            if isinstance(share, (int, float)):
                return share >= threshold
    return False


def _render_week_html(ctx, result, psi_table, ev) -> str:
    top = psi_table.head(20)
    rows = "\n".join(
        f"<tr class='{'breach' if r.breach else ''}'><td>{r.feature}</td>"
        f"<td>{r.kind}</td><td>{r.psi:.4f}</td></tr>"
        for r in top.itertuples()
    )
    breach_html = (
        "<div class='alert'>BREACH: " + "; ".join(result["breaches"]) + "</div>"
        if result["breaches"]
        else "<div class='ok'>No breaches this week.</div>"
    )
    return f"""<!doctype html><meta charset=utf-8><title>Monitoring — week {result["index"]:02d}</title>
{_STYLE}
<div class=banner>{BANNER}</div>
<h1>Simulated week {result["index"]:02d}</h1>
<p class=meta>provenance <b>{ctx.provenance}</b> · TransactionDT {result["dt_min"]:,}–{result["dt_max"]:,}
· {result["rows"]:,} transactions · fraud rate {result["fraud_rate"]:.3%}</p>
{breach_html}
<h2>Performance on labels <span class=caveat>(replay has labels immediately; production labels lag weeks)</span></h2>
<table><tr><th>PR-AUC</th><th>value capture</th><th>vs baseline</th><th>max PSI</th><th>Evidently drift</th></tr>
<tr><td>{result["pr_auc"]:.4f}</td><td>{result["value_capture"]:.1%}</td>
<td>{-result["vc_degradation"]:+.1%}</td><td>{result["max_psi"]:.4f}</td>
<td>{"yes" if result["evidently_drift"] else "no"}</td></tr></table>
<h2>PSI — top 20 features (threshold 0.2)</h2>
<table><tr><th>feature</th><th>kind</th><th>PSI</th></tr>{rows}</table>
<p><a href=evidently.html>Full Evidently drift report →</a></p>
<p class=meta><a href=../index.html>← monitoring index</a></p>
"""


def render_index_html(history: list[dict], provenance: str, frozen: bool) -> str:
    rows = "\n".join(
        f"<tr class='{'breach' if h['breaches'] else ''}'>"
        f"<td><a href='week-{h['index']:02d}/report.html'>week {h['index']:02d}</a></td>"
        f"<td>{h['rows']:,}</td><td>{h['fraud_rate']:.3%}</td><td>{h['pr_auc']:.4f}</td>"
        f"<td>{h['value_capture']:.1%}</td><td>{h['max_psi']:.4f}</td>"
        f"<td>{'⚠ ' + '; '.join(h['breaches']) if h['breaches'] else 'ok'}</td></tr>"
        for h in history
    )
    status = "FROZEN — replay stream exhausted; final report." if frozen else "Active replay."
    return f"""<!doctype html><meta charset=utf-8><title>Drift monitoring — replayed</title>
{_STYLE}
<div class=banner>{BANNER}</div>
<h1>Fraud model — replayed drift monitoring</h1>
<p class=meta>provenance <b>{provenance}</b> · cadence: one simulated week per scheduled run · {status}</p>
<p>Each run replays the next holdout slice, scores it with the deployed model, and reports
Evidently + PSI drift and label performance. Breach rule: PSI &gt; 0.2, Evidently dataset
drift, or value-capture down &gt;10% vs baseline → a GitHub Issue is opened.</p>
<table><tr><th>week</th><th>rows</th><th>fraud rate</th><th>PR-AUC</th>
<th>value capture</th><th>max PSI</th><th>status</th></tr>{rows}</table>
<h2>Methodology</h2>
<ul class=meta>
<li><b>PSI on dense inputs + the score distribution only.</b> Sparse block-NaN Vesta
<code>V*</code>/<code>id_*</code> features are excluded — their PSI is small-sample NaN noise,
not drift (D-008).</li>
<li><b>Structural-time features excluded from drift PSI.</b> Causal aggregates
(<code>ent_*</code>) grow as an entity accumulates history, and the calendar-position
transforms (<code>dt_day</code>, <code>dt_dow</code>) encode <i>when</i> a ~2-day batch
falls on the clock — both shift by construction under forward replay, not from data/concept
drift (D-008, D-013). Intra-day timing (<code>dt_hour</code>, <code>dt_is_night</code>) is
kept.</li>
<li><b>Value-capture baseline = the out-of-time norm</b> (the model's holdout value capture
at the frozen threshold), so the performance guard fires on <i>new</i> degradation rather
than the known validation→holdout gap (D-013).</li>
<li><b>Cadence:</b> one simulated week (of <code>TransactionDT</code>) per scheduled run;
the stream freezes with a final report when exhausted.</li>
<li><b>Replay + label lag:</b> a replayed simulation on a static dataset (banner above).
This replay has labels immediately; in production fraud labels arrive <i>weeks late</i>, so
live monitoring watches input + score drift, not just outcomes.</li>
</ul>
"""


_STYLE = """<style>
body{font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
.banner{background:#fff3cd;border:1px solid #ffca2c;border-radius:6px;padding:.5rem .8rem;margin-bottom:1rem;font-size:.9rem}
table{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem;font-size:.9rem}
th,td{border-bottom:1px solid #ddd;padding:.4rem .5rem;text-align:left}
th{color:#555}
tr.breach{background:#fdecea}
.alert{background:#fdecea;border:1px solid #e45756;color:#a11;border-radius:6px;padding:.5rem .8rem;margin:1rem 0;font-weight:600}
.ok{color:#3a7d34;margin:1rem 0}
.caveat,.meta{color:#666;font-size:.85rem;font-weight:normal}
a{color:#4c78a8}
</style>"""
