# Experiment grid

- **Provenance:** real  ·  **profile:** full
- **Feature counts:** 421 numeric + 15 categorical = **436 model features**.
- **Raw vs engineered:** 415 raw IEEE-CIS columns + 21 engineered (9 per-transaction base transforms + 12 strict-past causal aggregates). The engineered features are the ones the causality tests guard; raw `V*`/`id_*` fields are anonymized and used as-is.
- **Tracking:** MLflow (SQLite backend); registered model + version below.

## Temporal segmentation

| split | rows | frac | dt_min | dt_max | fraud_rate |
| --- | --- | --- | --- | --- | --- |
| train | 354,324 | 0.6 | 86400 | 8745772 | 0.03383 |
| val | 88,581 | 0.15 | 8745798 | 11246605 | 0.04036 |
| holdout | 147,635 | 0.25 | 11246665 | 15811131 | 0.03454 |

## Model × protocol grid

PR-AUC is primary (rare-positive). Temporal rows report VAL and HOLDOUT; the random-CV rows are the **leakage anti-pattern** (5-fold, time ignored) and report out-of-fold metrics on train+val only.

| model | protocol | val/cv PR-AUC | val/cv ROC-AUC | holdout PR-AUC | holdout ROC-AUC | holdout Brier |
| --- | --- | --- | --- | --- | --- | --- |
| logreg | temporal | 0.4962 | 0.8712 | 0.2090 | 0.8327 | 0.22830 |
| logreg | random_cv ⚠️ | 0.4669 | 0.8763 | — | — | — |
| xgboost | temporal | 0.6057 | 0.9240 | 0.4711 | 0.8849 | 0.07143 |
| xgboost | random_cv ⚠️ | 0.7105 | 0.9473 | — | — | — |
| lightgbm | temporal | 0.6207 | 0.9283 | 0.4789 | 0.8791 | 0.05236 |
| lightgbm | random_cv ⚠️ | 0.7646 | 0.9576 | — | — | — |

## Leakage experiment — random CV vs temporal split

Same features, same model, two validation protocols. Random k-fold CV ignores time, so it trains on future-dated rows and reports an **optimistic** PR-AUC. The inflation is the cost of the anti-pattern — the reason this project validates temporally throughout.

| model | temporal-val PR-AUC | random-CV PR-AUC | inflation (abs) | inflation (%) |
| --- | --- | --- | --- | --- |
| logreg | 0.4962 | 0.4669 | -0.0293 | -5.9% |
| xgboost | 0.6057 | 0.7105 | +0.1049 | +17.3% |
| lightgbm | 0.6207 | 0.7646 | +0.1439 | +23.2% |

## Calibration (lightgbm) — fit on VAL, assessed on VAL and HOLDOUT

Isotonic (default) and Platt calibration maps are fit on VALIDATION. Under class imbalance the Brier score is dominated by the rare-positive base rate and is a weak calibration diagnostic; the reliability curves (decision report) are the real check. Where VAL improves but HOLDOUT does not, that gap is **calibration drift** — a monitoring/recalibration trigger (validation report Section 7).

| method | val Brier | holdout Brier | holdout PR-AUC |
| --- | --- | --- | --- |
| none | 0.05220 | 0.05236 | 0.4789 |
| isotonic | 0.02249 | 0.02347 | 0.4642 |
| platt | 0.02330 | 0.02409 | 0.4789 |

## Registered model

- **Champion:** `fraud-scoring` v`2` — isotonic-calibrated **lightgbm** (run `79f0f88354444c0ba8a964c95344123c`).
- **Challenger:** `fraud-scoring-challenger` v`1` — Platt-calibrated lightgbm (run `f15083412f1a4ddf923a6e13a5003cac`), registered for the champion/challenger protocol (validation report Section 7); not served.

The champion's version + run id travel with the serving artifact and are reported by `/healthz`. LightGBM was promoted over XGBoost on VALIDATION (docs/decisions D-011).
