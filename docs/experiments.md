# Experiment grid

> **SYNTHETIC FIXTURE — illustrative numbers, NOT results.** The grid ran on the schema-identical synthetic fixture (SPEC Section 10). The machinery (models, protocols, calibration, registry) is real; regenerate on the real IEEE-CIS data with `uv run python -m pipeline.train --full`.

- **Provenance:** synthetic  ·  **profile:** full
- **Feature counts:** 409 numeric + 15 categorical
- **Tracking:** MLflow (SQLite backend); registered model + version below.

## Temporal segmentation

| split | rows | frac | dt_min | dt_max | fraud_rate |
| --- | --- | --- | --- | --- | --- |
| train | 3,000 | 0.6 | 86436 | 687754 | 0.03867 |
| val | 750 | 0.15 | 688065 | 839124 | 0.03467 |
| holdout | 1,250 | 0.25 | 839357 | 1083726 | 0.0264 |

## Model × protocol grid

PR-AUC is primary (rare-positive). Temporal rows report VAL and HOLDOUT; the random-CV rows are the **leakage anti-pattern** (5-fold, time ignored) and report out-of-fold metrics on train+val only.

| model | protocol | val/cv PR-AUC | val/cv ROC-AUC | holdout PR-AUC | holdout ROC-AUC | holdout Brier |
| --- | --- | --- | --- | --- | --- | --- |
| logreg | temporal | 0.1298 | 0.7674 | 0.1030 | 0.7751 | 0.06761 |
| logreg | random_cv ⚠️ | 0.1314 | 0.7786 | — | — | — |
| xgboost | temporal | 0.1670 | 0.8663 | 0.0955 | 0.8070 | 0.02611 |
| xgboost | random_cv ⚠️ | 0.2569 | 0.8803 | — | — | — |
| lightgbm | temporal | 0.1824 | 0.8800 | 0.1023 | 0.8432 | 0.02636 |
| lightgbm | random_cv ⚠️ | 0.2271 | 0.8740 | — | — | — |

## Leakage experiment — random CV vs temporal split

Same features, same model, two validation protocols. Random k-fold CV ignores time, so it trains on future-dated rows and reports an **optimistic** PR-AUC. The inflation is the cost of the anti-pattern — the reason this project validates temporally throughout.

| model | temporal-val PR-AUC | random-CV PR-AUC | inflation (abs) | inflation (%) |
| --- | --- | --- | --- | --- |
| logreg | 0.1298 | 0.1314 | +0.0017 | +1.3% |
| xgboost | 0.1670 | 0.2569 | +0.0899 | +53.9% |
| lightgbm | 0.1824 | 0.2271 | +0.0447 | +24.5% |

## Calibration (XGBoost) — fit on VAL, assessed on VAL and HOLDOUT

Isotonic (default) and Platt calibration maps are fit on VALIDATION. Under class imbalance the Brier score is dominated by the rare-positive base rate and is a weak calibration diagnostic; the reliability curves (decision report) are the real check. Where VAL improves but HOLDOUT does not, that gap is **calibration drift** — a monitoring/recalibration trigger (validation report Section 7).

| method | val Brier | holdout Brier | holdout PR-AUC |
| --- | --- | --- | --- |
| none | 0.03307 | 0.02611 | 0.0955 |
| isotonic | 0.02968 | 0.02653 | 0.0865 |
| platt | 0.03378 | 0.02724 | 0.0955 |

## Registered model

- **Name:** `fraud-scoring`  ·  **version:** `1`
- **Run id:** `508faebf96794b74b64ca3ae582051c3`  ·  **calibration:** isotonic
- **URI:** `models:/m-a566a2c5c69943d8b0f81f41971bf0a9`

The registered model is the isotonic-calibrated XGBoost. Its version and run id travel with the serving artifact and are reported by `/healthz` (Phase D).
