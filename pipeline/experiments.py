"""Experiment grid, leakage comparison, calibration, and MLflow registry (SPEC Section 3).

Runs LR / XGBoost / LightGBM under two protocols — the honest **temporal** split and
a 5-fold **random CV** (the leakage anti-pattern, labelled as such) — quantifying the
optimism the random protocol buys. Calibrates the XGBoost model (isotonic vs Platt) on
VALIDATION and reports Brier/reliability on HOLDOUT. Registers the calibrated XGBoost.

Every run is logged to a local MLflow tracking store; the grid is exported to
``docs/experiments.md``. On the synthetic fixture the numbers are illustrative and
labelled synthetic; the same command reproduces real results once data is fetched.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from pipeline import data as data_mod
from pipeline import encoders, metrics, modeling, schema

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DOC = REPO_ROOT / "docs" / "experiments.md"
# SQLite backend: the file store is deprecated in MLflow 3.x and never supported the
# model registry. Artifacts land under ./mlartifacts (both gitignored).
MLFLOW_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
EXPERIMENT_NAME = "fraud-detection"
REGISTERED_MODEL = "fraud-scoring"
CALIBRATION_METHODS = {"isotonic": "isotonic", "platt": "sigmoid"}


@dataclass
class GridResult:
    provenance: str
    profile: str
    cells: list[dict] = field(default_factory=list)  # per (model, protocol)
    calibration: list[dict] = field(default_factory=list)
    leakage: list[dict] = field(default_factory=list)
    registry: dict = field(default_factory=dict)
    split_summary: list[dict] = field(default_factory=list)
    feature_counts: dict = field(default_factory=dict)
    model: object = None  # fitted calibrated XGBoost (isotonic) — the serving model
    feature_columns: dict = field(default_factory=dict)  # {numeric: [...], categorical: [...]}


def _random_cv_scores(name, X_dev, y_dev, profile, seed=modeling.SEED, folds=5):
    """Out-of-fold probabilities under random StratifiedKFold — the leakage protocol."""
    model = modeling.make_model(name, y_train=y_dev, profile=profile, seed=seed)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(model, X_dev, y_dev, cv=skf, method="predict_proba", n_jobs=1)[:, 1]


def run_grid(
    source: str = "synthetic",
    profile: str = "full",
    register: bool = True,
    tracking_uri: str = MLFLOW_URI,
) -> GridResult:
    ds = data_mod.load_dataset(source)
    feat = encoders.feature_frame(ds.frame)
    sp = data_mod.temporal_split(feat, provenance=ds.provenance)

    y_tr = sp.train[schema.TARGET].to_numpy()
    y_val = sp.val[schema.TARGET].to_numpy()
    y_hold = sp.holdout[schema.TARGET].to_numpy()

    # Development set for the random-CV protocol = train + val (holdout untouched).
    dev = _concat(sp.train, sp.val)
    y_dev = dev[schema.TARGET].to_numpy()

    numeric_cols, categorical_cols = encoders.model_feature_columns(sp.train)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    result = GridResult(
        provenance=ds.provenance,
        profile=profile,
        split_summary=sp.summary().to_dict(orient="records"),
        feature_counts={"numeric": len(numeric_cols), "categorical": len(categorical_cols)},
        feature_columns={"numeric": numeric_cols, "categorical": categorical_cols},
    )

    fitted_temporal = {}
    for name in modeling.MODEL_NAMES:
        # --- Temporal protocol: fit TRAIN, evaluate VAL (+ HOLDOUT). ---
        model = modeling.make_model(name, y_train=y_tr, profile=profile)
        model.fit(sp.train, y_tr)
        fitted_temporal[name] = model
        val_eval = metrics.evaluate(y_val, modeling.positive_proba(model, sp.val))
        hold_eval = metrics.evaluate(y_hold, modeling.positive_proba(model, sp.holdout))
        _log_run(
            f"{name}-temporal",
            {"model": name, "protocol": "temporal", "profile": profile},
            {**_prefixed("val", val_eval), **_prefixed("holdout", hold_eval)},
        )
        result.cells.append(
            {
                "model": name,
                "protocol": "temporal",
                "val_pr_auc": val_eval.pr_auc,
                "val_roc_auc": val_eval.roc_auc,
                "holdout_pr_auc": hold_eval.pr_auc,
                "holdout_roc_auc": hold_eval.roc_auc,
                "holdout_brier": hold_eval.brier,
            }
        )

        # --- Random-CV protocol (leakage anti-pattern): 5-fold on dev. ---
        oof = _random_cv_scores(name, dev, y_dev, profile)
        cv_eval = metrics.evaluate(y_dev, oof)
        _log_run(
            f"{name}-random_cv",
            {"model": name, "protocol": "random_cv", "profile": profile, "note": "LEAKAGE-DEMO"},
            _prefixed("cv", cv_eval),
        )
        result.cells.append(
            {
                "model": name,
                "protocol": "random_cv",
                "val_pr_auc": cv_eval.pr_auc,
                "val_roc_auc": cv_eval.roc_auc,
                "holdout_pr_auc": None,
                "holdout_roc_auc": None,
                "holdout_brier": None,
            }
        )

        # Leakage delta: random-CV PR-AUC vs honest temporal-val PR-AUC.
        result.leakage.append(
            {
                "model": name,
                "temporal_val_pr_auc": val_eval.pr_auc,
                "random_cv_pr_auc": cv_eval.pr_auc,
                "inflation_abs": cv_eval.pr_auc - val_eval.pr_auc,
                "inflation_pct": _pct(cv_eval.pr_auc, val_eval.pr_auc),
            }
        )

    # --- Calibration of the XGBoost model on VAL, assessed on VAL (in-distribution)
    #     and HOLDOUT. Reporting both exposes calibration drift honestly; the
    #     registered method is the a-priori default (isotonic), never chosen by
    #     peeking at holdout (that would leak the test set into model selection). ---
    xgb = fitted_temporal["xgboost"]
    base_val = metrics.evaluate(y_val, modeling.positive_proba(xgb, sp.val))
    base_hold = metrics.evaluate(y_hold, modeling.positive_proba(xgb, sp.holdout))
    result.calibration.append(
        {
            "method": "none",
            "val_brier": base_val.brier,
            "holdout_brier": base_hold.brier,
            "holdout_pr_auc": base_hold.pr_auc,
        }
    )
    cal_models = {}
    for label, sk_method in CALIBRATION_METHODS.items():
        cal = modeling.calibrate(xgb, sp.val, y_val, method=sk_method)
        cal_models[label] = cal
        val_brier = metrics.evaluate(y_val, cal.predict_proba(sp.val)[:, 1]).brier
        cal_hold = metrics.evaluate(y_hold, cal.predict_proba(sp.holdout)[:, 1])
        _log_run(
            f"xgboost-calibrated-{label}",
            {"model": "xgboost", "protocol": "temporal", "calibration": label},
            {
                "val_brier": val_brier,
                "holdout_brier": cal_hold.brier,
                "holdout_pr_auc": cal_hold.pr_auc,
            },
        )
        result.calibration.append(
            {
                "method": label,
                "val_brier": val_brier,
                "holdout_brier": cal_hold.brier,
                "holdout_pr_auc": cal_hold.pr_auc,
            }
        )

    # --- Register the calibrated XGBoost (isotonic = SPEC default). ---
    result.model = cal_models["isotonic"]
    if register:
        result.registry = _register(cal_models["isotonic"], "isotonic", ds.provenance, profile)

    return result


def _concat(a, b):
    import pandas as pd

    return pd.concat([a, b], ignore_index=True)


def _prefixed(prefix: str, ev: metrics.EvalResult) -> dict:
    out = {
        f"{prefix}_pr_auc": ev.pr_auc,
        f"{prefix}_roc_auc": ev.roc_auc,
        f"{prefix}_brier": ev.brier,
    }
    for fpr, rec in ev.recall_at_fpr.items():
        out[f"{prefix}_recall_fpr_{fpr:g}"] = rec
    return out


def _pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b if b else float("nan")


def _log_run(run_name: str, tags: dict, metrics_dict: dict) -> str:
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_metrics({k: float(v) for k, v in metrics_dict.items() if v is not None})
        return run.info.run_id


def _register(model, method: str, provenance: str, profile: str) -> dict:
    with mlflow.start_run(run_name="register-calibrated-xgboost") as run:
        mlflow.set_tags(
            {
                "model": "xgboost",
                "calibration": method,
                "provenance": provenance,
                "profile": profile,
                "role": "registered-production-candidate",
            }
        )
        # cloudpickle (not the newer skops serializer) — our pipeline wraps a custom
        # transformer + XGBoost booster that skops will not trust by default.
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=REGISTERED_MODEL,
            serialization_format="cloudpickle",
        )
        run_id = run.info.run_id
    version = _latest_version(REGISTERED_MODEL)
    return {
        "registered_model": REGISTERED_MODEL,
        "version": version,
        "run_id": run_id,
        "calibration": method,
        "model_uri": info.model_uri,
    }


def render_experiments_md(r: GridResult) -> str:
    lines = ["# Experiment grid", ""]
    if r.provenance != "real":
        lines += [
            "> **SYNTHETIC FIXTURE — illustrative numbers, NOT results.** The grid ran "
            "on the schema-identical synthetic fixture (SPEC Section 10). The machinery "
            "(models, protocols, calibration, registry) is real; regenerate on the real "
            "IEEE-CIS data with `uv run python -m pipeline.train --full`.",
            "",
        ]
    lines += [
        f"- **Provenance:** {r.provenance}  ·  **profile:** {r.profile}",
        f"- **Feature counts:** {r.feature_counts.get('numeric', '?')} numeric + "
        f"{r.feature_counts.get('categorical', '?')} categorical",
        "- **Tracking:** MLflow (SQLite backend); registered model + version below.",
        "",
        "## Temporal segmentation",
        "",
        "| split | rows | frac | dt_min | dt_max | fraud_rate |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in r.split_summary:
        lines.append(
            f"| {s['split']} | {s['rows']:,} | {s['frac']} | {s['dt_min']} | "
            f"{s['dt_max']} | {s['fraud_rate']} |"
        )

    lines += [
        "",
        "## Model × protocol grid",
        "",
        "PR-AUC is primary (rare-positive). Temporal rows report VAL and HOLDOUT; the "
        "random-CV rows are the **leakage anti-pattern** (5-fold, time ignored) and "
        "report out-of-fold metrics on train+val only.",
        "",
        "| model | protocol | val/cv PR-AUC | val/cv ROC-AUC | holdout PR-AUC | holdout ROC-AUC | holdout Brier |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in r.cells:
        hp = "—" if c["holdout_pr_auc"] is None else f"{c['holdout_pr_auc']:.4f}"
        hr = "—" if c["holdout_roc_auc"] is None else f"{c['holdout_roc_auc']:.4f}"
        hb = "—" if c["holdout_brier"] is None else f"{c['holdout_brier']:.5f}"
        proto = "random_cv ⚠️" if c["protocol"] == "random_cv" else c["protocol"]
        lines.append(
            f"| {c['model']} | {proto} | {c['val_pr_auc']:.4f} | {c['val_roc_auc']:.4f} | "
            f"{hp} | {hr} | {hb} |"
        )

    lines += [
        "",
        "## Leakage experiment — random CV vs temporal split",
        "",
        "Same features, same model, two validation protocols. Random k-fold CV ignores "
        "time, so it trains on future-dated rows and reports an **optimistic** PR-AUC. "
        "The inflation is the cost of the anti-pattern — the reason this project validates "
        "temporally throughout.",
        "",
        "| model | temporal-val PR-AUC | random-CV PR-AUC | inflation (abs) | inflation (%) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for lk in r.leakage:
        lines.append(
            f"| {lk['model']} | {lk['temporal_val_pr_auc']:.4f} | {lk['random_cv_pr_auc']:.4f} | "
            f"{lk['inflation_abs']:+.4f} | {lk['inflation_pct']:+.1f}% |"
        )

    lines += [
        "",
        "## Calibration (XGBoost) — fit on VAL, assessed on VAL and HOLDOUT",
        "",
        "Isotonic (default) and Platt calibration maps are fit on VALIDATION. Under "
        "class imbalance the Brier score is dominated by the rare-positive base rate and "
        "is a weak calibration diagnostic; the reliability curves (decision report) are "
        "the real check. Where VAL improves but HOLDOUT does not, that gap is **calibration "
        "drift** — a monitoring/recalibration trigger (validation report Section 7).",
        "",
        "| method | val Brier | holdout Brier | holdout PR-AUC |",
        "| --- | --- | --- | --- |",
    ]
    for c in r.calibration:
        vb = c.get("val_brier")
        vb = "—" if vb is None else f"{vb:.5f}"
        lines.append(
            f"| {c['method']} | {vb} | {c['holdout_brier']:.5f} | {c['holdout_pr_auc']:.4f} |"
        )

    reg = r.registry
    lines += [
        "",
        "## Registered model",
        "",
        f"- **Name:** `{reg.get('registered_model', REGISTERED_MODEL)}`  ·  "
        f"**version:** `{reg.get('version', '—')}`",
        f"- **Run id:** `{reg.get('run_id', '—')}`  ·  **calibration:** "
        f"{reg.get('calibration', 'isotonic')}",
        f"- **URI:** `{reg.get('model_uri', '—')}`",
        "",
        "The registered model is the isotonic-calibrated XGBoost. Its version and run id "
        "travel with the serving artifact and are reported by `/healthz` (Phase D).",
        "",
    ]
    return "\n".join(lines)


def write_experiments_md(r: GridResult, path: Path = EXPERIMENTS_DOC) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_experiments_md(r), encoding="utf-8")
    return path


def _latest_version(name: str) -> str | None:
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{name}'")
        if not versions:
            return None
        return str(max(int(v.version) for v in versions))
    except Exception:  # pragma: no cover - registry backend dependent
        return None
