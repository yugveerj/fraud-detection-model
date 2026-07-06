# Model validation report — IEEE-CIS fraud detection

**Model:** `fraud-scoring` (MLflow registry v1) — isotonic-calibrated XGBoost
**Status:** **Non-production demonstration** on a public research dataset
**Framework:** structured after Federal Reserve **SR 11-7** *Supervisory Guidance on
Model Risk Management*. SR 11-7 was superseded on 17 Apr 2026 by **SR 26-2** (*Revised
Guidance on Model Risk Management*, which also replaced SR 21-8); SR 26-2 retains the
same core framework — conceptual soundness, ongoing monitoring, outcomes analysis, and
effective challenge — while emphasizing a **risk-based approach tailored to the
institution's model-risk profile**. This report mirrors those concerns, not
boilerplate, and cites this repository's real artifacts throughout.

> **Provenance caveat, applied to every number below.** No Kaggle credentials were
> available at build time, so the pipeline ran on a **schema-identical synthetic
> fixture** ([`pipeline/synthetic.py`](../pipeline/synthetic.py); manifest
> [`docs/data_manifest.md`](data_manifest.md)). Every figure is labelled *synthetic /
> illustrative*. The machinery is real and reproduces the **real** results with one
> command (`uv run python -m pipeline.train --full && uv run python -m pipeline.evaluate --report`)
> the moment credentials are supplied. This caveat travels with any number that leaves
> the repo.

---

## 0. Executive summary

A gradient-boosted fraud model is developed under **leakage-proof temporal validation**,
calibrated to produce trustworthy probabilities, and operated through a **dollar-framed**
decision threshold. The central finding of the validation is methodological: a
random-cross-validation protocol inflates the primary metric by **+15% to +42%** over the
honest temporal protocol (§4.1) — the reason this model is validated temporally
throughout. On the (synthetic) holdout the calibrated model achieves **PR-AUC 0.308,
ROC-AUC 0.909, Brier 0.0228**; at the frozen operating point it captures **35.9% of fraud
value at a 1.65% false-positive rate** for a positive net value. Performance degrades on
the out-of-time holdout relative to validation — an expected consequence of temporal
drift that the monitoring layer (§7) is built to detect. Results are independently
reproduced in R (§4.4, effective challenge).

---

## 1. Purpose and intended use (SR 11-7 §III — model definition & use)

**Purpose.** Assign each card transaction a calibrated probability of fraud and, via a
cost model, a review/approve decision.

**Intended use.** *Demonstration only.* This model is **not approved for production
use** and makes no real fraud decisions. It exists to demonstrate model-risk-grade
development and validation practice on a public dataset. The served demo and API carry
the disclaimer "Demonstration system on a public research dataset — not a production
fraud decision" ([`serving/models.py`](../serving/models.py), the web demo footer).

**Development vs usage risk (SR 11-7).** *Development risk* is controlled by temporal
integrity (§3.2), calibration (§3.4), and independent replication (§4.4). *Usage risk*
is bounded by the non-production status, the anonymized-feature interpretation caveat
(§6), and the label-lag caveat that accompanies every performance number.

---

## 2. Data (SR 11-7 §V — data quality & relevance)

- **Source.** Kaggle IEEE-CIS Fraud Detection — `train_transaction` + `train_identity`
  (~590k labelled transactions; `TransactionDT` is a relative timestamp). The unlabelled
  competition test set is not used. Integrity (row counts, class balance, checksums) is
  recorded at fetch time to [`docs/data_manifest.md`](data_manifest.md)
  ([`pipeline/manifest.py`](../pipeline/manifest.py)).
- **This run (synthetic fixture).** 5,000 transactions × 394 columns, **3.5% fraud**;
  identity present for 24% of transactions (left-joined). Splits below.
- **Temporal segmentation** — by `TransactionDT`, documented cut points
  ([`pipeline/data.py`](../pipeline/data.py)):

  | split | share | role |
  | --- | --- | --- |
  | TRAIN | first 60% | model fitting |
  | VALIDATION | next 15% | calibration + threshold + model selection |
  | HOLDOUT | final 25% | reported once, then replayed as the monitoring stream (§7) |

- **Known dataset limitations.** Features `V1–V339` (Vesta engineered) and `id_12–id_38`
  are **anonymized** with no published semantics; heavy block-structured missingness;
  a single fixed observation window (no multi-year seasonality); the competition labels
  are the ground truth used here.

---

## 3. Methodology and assumptions (SR 11-7 §IV — conceptual soundness)

### 3.1 Features
Base per-transaction transforms (amount log/decimal/round-number flags, time-of-day and
day-of-week from `TransactionDT`) plus **causal entity aggregates** (per card / email /
address: prior count, prior mean/std amount, recency)
([`pipeline/features.py`](../pipeline/features.py)). The feature set is deliberately
disciplined (~50–100 engineered columns) — the project's value is validation rigor, not
feature count.

### 3.2 Causality — the headline control
Every entity aggregate is computed over **strictly-earlier** transactions only (past-only
expanding windows; rows sharing a `TransactionDT` do not see one another). This is
enforced by **property-style tests** that shuffle, truncate, and append future rows and
assert past-row feature values are byte-for-byte unchanged, plus a brute-force O(n²)
ground-truth check on tie-heavy, NaN-laden data
([`tests/test_causality.py`](../tests/test_causality.py),
[`tests/test_strict_past_bruteforce.py`](../tests/test_strict_past_bruteforce.py)). An
adversarial multi-agent audit of this layer found and fixed three temporal-integrity
defects during development ([`docs/decisions.md`](decisions.md) D-004).

### 3.3 Class imbalance
Handled with **cost-sensitive learning** (`scale_pos_weight` / balanced class weights) and
threshold optimization (§5). SMOTE and resampling are **deliberately declined** — synthetic
minority oversampling distorts the calibration the decision layer depends on and fabricates
transactions that never occurred in a temporal stream ([`docs/decisions.md`](decisions.md)
D-001).

### 3.4 Calibration
Probabilities are calibrated with **isotonic regression** (default; Platt compared) fit on
VALIDATION only, using a frozen base estimator so the model is never refit
([`pipeline/modeling.py`](../pipeline/modeling.py)). No probability is displayed anywhere
unless it comes from the calibrated model.

### 3.5 Assumptions
(1) `TransactionDT` ordering is a faithful proxy for real time; (2) blocking a flagged
fraud recovers its full `TransactionAmt`; (3) review cost is a fixed per-alert parameter
(default $25); (4) in this replay, labels are available immediately — **in production they
lag weeks** (§6).

---

## 4. Performance testing (SR 11-7 §IV — outcomes analysis)

### 4.1 The leakage experiment — protocol matters more than the model
Same features, same model, two validation protocols
([`docs/experiments.md`](experiments.md)):

| model | temporal-val PR-AUC | random-CV PR-AUC | inflation |
| --- | --- | --- | --- |
| logistic regression | 0.2525 | 0.3590 | **+42.2%** |
| XGBoost | 0.3770 | 0.4695 | **+24.5%** |
| LightGBM | 0.4093 | 0.4690 | **+14.6%** |

Random k-fold CV trains on future-dated rows and reports an optimistic PR-AUC. The
inflation is the cost of the anti-pattern — quantified here, and the reason every split
in this project is temporal.

### 4.2 Holdout metrics (reported once)
Isotonic-calibrated XGBoost on the HOLDOUT (n=1,250; 35 frauds):

| metric | value |
| --- | --- |
| PR-AUC (primary) | **0.308** |
| ROC-AUC | 0.909 |
| Brier score | 0.0228 |
| recall @ frozen operating point | 34.3% |

Calibration (fit on VAL, assessed on VAL and HOLDOUT): isotonic improves the
**validation** Brier (0.0194 → 0.0161). On the holdout the Brier is comparable
(≈0.023); under class imbalance the Brier is dominated by the base rate and is a weak
calibration diagnostic — the **reliability curve** ([`docs/figures/reliability.png`](figures/reliability.png))
is the real check.

### 4.3 Stability across time (SR 11-7 — performance over time)
The HOLDOUT is replayed as ~10 simulated weeks ([§7](#7-ongoing-monitoring-plan); the
`/monitoring/` index). Weekly PR-AUC and value-capture **vary materially** across the
window and degrade relative to validation — the expected signature of the temporal drift
injected into the fixture (concept + covariate; [`docs/decisions.md`](decisions.md) D-006).
This instability is the justification for continuous monitoring and recalibration
triggers, not a defect to be hidden.

### 4.4 Independent replication — effective challenge (SR 11-7 §V; SPEC §7b)
The holdout metrics are recomputed **independently in R, with no imports from the Python
codebase**, from a single exported scores file
([`docs/validation_r/replication.Rmd`](validation_r/replication.Rmd) reading
`holdout_scores.csv`). R re-derives PR-AUC, ROC-AUC, Brier, the reliability curve, the
value-capture curve, and a score-distribution PSI, and **reconciles them against the
Python figures within tolerance**. This is the "effective challenge" SR 11-7 requires:
an independent implementation reproducing the developer's results.

---

## 5. Decision framework (SR 11-7 — model use & interpretation)

The model output is an input to a business decision, framed in dollars
([`pipeline/decisions.py`](../pipeline/decisions.py),
[`docs/decision_report.md`](decision_report.md)):

- **Cost model.** Net value = (fraud $ caught by alerts) − review_cost × (alerts).
  Reviewing costs $25 (parameter); catching a fraud saves its `TransactionAmt`.
- **Operating point.** Threshold **optimized on VALIDATION, frozen, reported on
  HOLDOUT** — never tuned on the reported data. At the frozen threshold (0.25) the model
  captures **35.9% of fraud value at 1.65% FPR** for **≈ $236k net value per 100k
  transactions** (synthetic). The tie-block realizability of the operating point was
  corrected after an adversarial review ([`docs/decisions.md`](decisions.md), Phase C).
- **Sensitivity.** The optimal operating point is reported across review costs $5–$75
  ([`docs/decision_report.md`](decision_report.md) §3): higher cost → higher threshold →
  fewer, higher-precision alerts.
- **Attribution.** SHAP global importance + fraud/legit/borderline case studies
  ([`pipeline/explain.py`](../pipeline/explain.py)), with the anonymized-feature caveat.

---

## 6. Limitations (SR 11-7 — assumptions & limitations)

- **Anonymized features.** `V*` / `id_*` have no published meaning; SHAP shows *which
  engineered inputs* move a score, not a mechanistic business reason. Interpretability is
  structurally capped.
- **Replay vs production.** Monitoring is a **replayed simulation on a static dataset**,
  not live traffic; every monitoring page says so.
- **Label lag.** This replay has labels immediately; in production fraud labels arrive
  **weeks late**, so live performance metrics are unavailable in real time — drift must
  be watched through *input* and *score* distributions (§7), not just outcomes.
- **Dataset vintage.** A single fixed window; no multi-year seasonality or regime change.
- **Synthetic fixture.** All numbers here are synthetic and illustrative until the
  pipeline is rerun on the real IEEE-CIS data.
- **Small-sample monitoring.** On the ~125-row synthetic weekly batches, value-capture is
  noisy; the breach logic guards on minimum fraud counts. Real weekly volumes remove this.

---

## 7. Ongoing monitoring plan (SR 11-7 §IV — ongoing monitoring)

Implemented in [`monitoring/`](../monitoring/); published to GitHub Pages `/monitoring/`
by a daily cron ([`.github/workflows/monitoring.yml`](../.github/workflows/monitoring.yml)).

- **What is monitored.** Per simulated week: Evidently data/prediction drift, explicit
  **PSI** over the top dense input features and the **score distribution**, and
  performance on labels (with the label-lag caveat). Causal aggregates are excluded from
  drift PSI (they grow structurally with history; [`docs/decisions.md`](decisions.md)
  D-008).
- **Breach thresholds → automated GitHub Issue.** (1) PSI **> 0.2** on any tracked
  feature; (2) Evidently flags dataset drift; (3) value-capture at the operating point
  **down > 10%** vs baseline. The alert path is verified by a forced-breach test
  ([`tests/test_monitoring.py`](../tests/test_monitoring.py); `--inject-drift`).
- **Escalation.** A drift Issue is triaged by severity; a sustained value-capture breach
  or PSI breach on a top-importance input escalates to retraining.
- **Retrain triggers.** (a) sustained value-capture degradation > 10% over two
  consecutive windows; (b) PSI > 0.2 on a top-5 importance feature; (c) calibration drift
  (holdout reliability departing from validation) — recalibrate first, retrain if
  unresolved.
- **Champion / challenger.** The registered model is champion. A challenger (e.g. the
  LightGBM configuration in the grid, or a retrained champion) is scored on the same
  replay stream; promotion requires beating the champion on temporal-holdout PR-AUC **and**
  net value at the operating point, never on random-CV.
- **Availability.** A scheduled uptime check pings the demo and `/healthz`; failure opens
  a GitHub Issue ([`.github/workflows/uptime.yml`](../.github/workflows/uptime.yml)) — a
  dead recruiter-facing link is itself a monitored failure state.

---

## 8. Governance & documentation (SR 11-7 §VI)

- **Model inventory.** The model is registered in MLflow (`fraud-scoring` v1) with its run
  id; the serving artifact and `/healthz` report the version + run id
  ([`serving/handler.py`](../serving/handler.py)).
- **Reproducibility.** One seeded command rebuilds the model, experiments, and reports;
  CI runs lint, unit + causality tests, and a seeded smoke-train on every push.
- **Change control.** Infrastructure is Terraform ([`infra/`](../infra/)); once applied,
  console-only changes are prohibited. `terraform apply` is owner/CI-run behind a gated
  environment; the agent authors but never applies.
- **Decisions log.** Consequential choices and their rationale are recorded in
  [`docs/decisions.md`](decisions.md).
- **Effective challenge.** Each development phase was subjected to an adversarial,
  reproduction-driven review; findings and fixes are logged (D-004, Phase C tie-block,
  Phase D serving hardening).

---

*Companion artifacts:* [model card](model_card.md) ·
[experiments](experiments.md) · [decision report](decision_report.md) ·
[data manifest](data_manifest.md) · [R replication](validation_r/replication.Rmd) ·
[decisions log](decisions.md).
