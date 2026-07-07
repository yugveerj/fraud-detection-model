# Model card — `fraud-scoring`

Standard model-card fields (Mitchell et al.). Figures are computed on the **real Kaggle
IEEE-CIS Fraud Detection data** (590,540 transactions).

## Model details
- **Name / version:** `fraud-scoring`, MLflow registry **v2** (run id embedded in the
  serving artifact and reported by `/healthz`). Champion selected over XGBoost on
  VALIDATION ([`docs/decisions.md`](decisions.md) D-011).
- **Type:** isotonic-calibrated **LightGBM** classifier in a scikit-learn `Pipeline`
  (leakage-safe preprocessing → gradient-boosted trees → isotonic calibration).
- **Challenger:** Platt-calibrated LightGBM, registered `fraud-scoring-challenger` v1.
- **Owner:** portfolio project #3. **Framework:** SR 11-7 / SR 26-2-style validation.
- **Objective:** calibrated P(fraud) per transaction, turned into a review/approve
  decision by a dollar-framed threshold.

## Intended use
- **In scope:** demonstration of model-risk-grade development, calibration, and
  temporal validation on a public research dataset.
- **Out of scope / not approved:** any real fraud decision. The API and demo carry the
  disclaimer "Demonstration system on a public research dataset — not a production fraud
  decision."

## Training data
- Kaggle **IEEE-CIS Fraud Detection** (`train_transaction` + `train_identity`),
  segmented **by `TransactionDT`**: TRAIN 60% / VALIDATION 15% / HOLDOUT 25%.
  Manifest: [`docs/data_manifest.md`](data_manifest.md). Class balance ≈ 3.5% fraud.

## Evaluation & metrics (HOLDOUT, 147,635 transactions)
| metric | value |
| --- | --- |
| PR-AUC (primary) | 0.464 |
| ROC-AUC | 0.879 |
| Brier (calibrated; 0.052 uncalibrated) | 0.0235 |
| operating point | 47.8% fraud value captured @ 4.03% FPR (recall 55.3%) |
| net value | ≈ $111k / 100k transactions (review cost $25) |

Threshold optimized on VALIDATION, frozen, reported on HOLDOUT. Independently reproduced
in R ([`docs/validation_r/`](validation_r/replication.Rmd)). Full grid + leakage
comparison: [`docs/experiments.md`](experiments.md). Isotonic is the champion calibrator
despite a small ranking cost (holdout PR-AUC 0.464 vs 0.479 for the Platt challenger) —
it wins on Brier and reliability, which the dollar thresholds depend on; the tradeoff is
documented in [`docs/validation_report.md`](validation_report.md) §4.2.

## Factors & interpretation
- SHAP global importance + case studies ([`docs/decision_report.md`](decision_report.md)).
- **Caveat:** `V*`/`id_*` features are anonymized — attributions show *which engineered
  inputs* move a score, not a business reason.

## Ethical considerations & limitations
- **Non-production**; no real decisions are made.
- **Label lag:** production fraud labels arrive weeks late; live metrics are unavailable
  in real time — drift is watched via input/score distributions.
- **Fairness:** the dataset lacks protected-attribute labels; no fairness assessment is
  claimed. A production deployment would require one.
- **Drift:** performance degrades on out-of-time data; monitored continuously with
  breach → GitHub Issue ([`docs/validation_report.md`](validation_report.md) §7).

## Maintenance
- Retrain triggers, champion/challenger, and monitoring thresholds:
  [`docs/validation_report.md`](validation_report.md) §7.
- Reproduce: `uv run python -m pipeline.train --full && uv run python -m pipeline.evaluate --report`.
