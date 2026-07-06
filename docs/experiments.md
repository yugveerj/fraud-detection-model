# Experiment grid

> **SYNTHETIC FIXTURE — illustrative numbers, NOT results.** The grid ran on the schema-identical synthetic fixture (SPEC Section 10). The machinery (models, protocols, calibration, registry) is real; regenerate on the real IEEE-CIS data with `uv run python -m pipeline.train --full`.

- **Provenance:** synthetic  ·  **profile:** full
- **Feature counts:** 409 numeric + 15 categorical
- **Tracking:** MLflow (SQLite backend); registered model + version below.

## Temporal segmentation

| split | rows | frac | dt_min | dt_max | fraud_rate |
| --- | --- | --- | --- | --- | --- |
| train | 3,000 | 0.6 | 86436 | 687754 | 0.04067 |
| val | 750 | 0.15 | 688065 | 839124 | 0.024 |
| holdout | 1,250 | 0.25 | 839357 | 1083726 | 0.028 |

## Model × protocol grid

PR-AUC is primary (rare-positive). Temporal rows report VAL and HOLDOUT; the random-CV rows are the **leakage anti-pattern** (5-fold, time ignored) and report out-of-fold metrics on train+val only.

| model | protocol | val/cv PR-AUC | val/cv ROC-AUC | holdout PR-AUC | holdout ROC-AUC | holdout Brier |
| --- | --- | --- | --- | --- | --- | --- |
| logreg | temporal | 0.2525 | 0.9260 | 0.2041 | 0.8459 | 0.05606 |
| logreg | random_cv ⚠️ | 0.3590 | 0.8990 | — | — | — |
| xgboost | temporal | 0.3770 | 0.9611 | 0.3194 | 0.9367 | 0.02332 |
| xgboost | random_cv ⚠️ | 0.4695 | 0.9479 | — | — | — |
| lightgbm | temporal | 0.4093 | 0.9481 | 0.3362 | 0.9205 | 0.02575 |
| lightgbm | random_cv ⚠️ | 0.4690 | 0.9442 | — | — | — |

## Leakage experiment — random CV vs temporal split

Same features, same model, two validation protocols. Random k-fold CV ignores time, so it trains on future-dated rows and reports an **optimistic** PR-AUC. The inflation is the cost of the anti-pattern — the reason this project validates temporally throughout.

| model | temporal-val PR-AUC | random-CV PR-AUC | inflation (abs) | inflation (%) |
| --- | --- | --- | --- | --- |
| logreg | 0.2525 | 0.3590 | +0.1065 | +42.2% |
| xgboost | 0.3770 | 0.4695 | +0.0925 | +24.5% |
| lightgbm | 0.4093 | 0.4690 | +0.0597 | +14.6% |

## Calibration (XGBoost) — fit on VAL, assessed on VAL and HOLDOUT

Isotonic (default) and Platt calibration maps are fit on VALIDATION. Under class imbalance the Brier score is dominated by the rare-positive base rate and is a weak calibration diagnostic; the reliability curves (decision report) are the real check. Where VAL improves but HOLDOUT does not, that gap is **calibration drift** — a monitoring/recalibration trigger (validation report Section 7).

| method | val Brier | holdout Brier | holdout PR-AUC |
| --- | --- | --- | --- |
| none | 0.01935 | 0.02332 | 0.3194 |
| isotonic | 0.01611 | 0.02283 | 0.3080 |
| platt | 0.01954 | 0.02481 | 0.3194 |

## Registered model

- **Name:** `fraud-scoring`  ·  **version:** `1`
- **Run id:** `1449d41574ac40afa668f09d2a6bf72c`  ·  **calibration:** isotonic
- **URI:** `models:/m-5d36901da7e34b4a89b4b4c7bc48ea3e`

The registered model is the isotonic-calibrated XGBoost. Its version and run id travel with the serving artifact and are reported by `/healthz` (Phase D).
