# Decision report — is this model worth operating?

This report is written for a reviewer deciding whether to **operate** the model, not how it was trained. Everything is framed in dollars.

## 1. Cost model

- Reviewing a transaction costs **$25** (business parameter).
- Catching a fraudulent transaction saves its `TransactionAmt` (loss avoided).
- A missed fraud loses its `TransactionAmt`.
- **Net value** of an operating point = fraud dollars caught − review cost × alerts.

The threshold is optimized on **VALIDATION** and reported on **HOLDOUT** — never tuned on the data it is reported on.

## 2. Operating point

**At the frozen operating point (threshold 0.1369, review cost $25), the model captures 47.5% of fraud value at a 3.44% false-positive rate, for approximately $125,228 net value per 100k transactions.**

| metric | VALIDATION (optimized) | HOLDOUT (reported) |
| --- | --- | --- |
| threshold | 0.1369 | 0.1369 (frozen) |
| alert rate | 6.13% | 5.18% |
| recall (count) | 65.0% | 53.8% |
| false-positive rate | 3.65% | 3.44% |
| fraud value captured | 58.2% | 47.5% |
| net value / 100k | $250,995 | $125,228 |

The holdout capture is lower than validation: an out-of-time slice under drift is genuinely harder, and the frozen threshold does not perfectly transfer. Net value remains positive, and the gap is exactly what ongoing monitoring watches (validation report Section 7).

![value capture](figures/value_capture.png)

![net value](figures/net_value.png)

## 3. Review-cost sensitivity

How the optimal operating point shifts as the review-cost assumption varies ($5–$75). Higher review cost → higher threshold → fewer, higher-precision alerts.

| review cost | opt threshold | alert rate | recall | fpr | value capture | net / 100k |
| --- | --- | --- | --- | --- | --- | --- |
| $5 | 0.0308 | 21.19% | 87.0% | 18.43% | 86.0% | $491,080 |
| $15 | 0.0909 | 10.67% | 75.8% | 7.93% | 73.8% | $352,385 |
| $25 | 0.1369 | 6.13% | 65.0% | 3.65% | 58.2% | $250,995 |
| $50 | 0.2358 | 3.39% | 52.8% | 1.32% | 42.2% | $123,015 |
| $75 | 0.4145 | 2.38% | 45.0% | 0.59% | 32.9% | $49,720 |

## 4. Calibration

Holdout Brier 0.0236. Under class imbalance the Brier score is dominated by the rare-positive base rate; the reliability curve is the real check. Isotonic calibration corrects the probability *level* the dollar thresholds depend on. Where the holdout curve departs from validation, that is calibration drift.

![reliability](figures/reliability.png)

## 5. Feature attribution (SHAP)

Global importance on the base XGBoost (TreeExplainer). **Caveat:** many inputs are anonymized Vesta `V*` / `id_*` features with no published meaning, so SHAP shows *which engineered inputs* move a score, not a mechanistic business reason.

![shap global](figures/shap_global_bar.png)

![shap beeswarm](figures/shap_beeswarm.png)

### Case studies (HOLDOUT)

**Fraud** — actual label 1, calibrated P(fraud) 0.942, amount $100.00 → **ALERT** at threshold 0.137.

| feature | value | SHAP |
| --- | --- | --- |
| V258 | 3.000 | +1.157 |
| C1 | 25.000 | +1.131 |
| C13 | 1.000 | +0.490 |
| V187 | 3.000 | +0.488 |
| V152 | 4.000 | +0.465 |
| C8 | 15.000 | +0.358 |

**Legit** — actual label 0, calibrated P(fraud) 0.000, amount $311.95 → **pass** at threshold 0.137.

| feature | value | SHAP |
| --- | --- | --- |
| C13 | 511.000 | -0.708 |
| C14 | 109.000 | -0.501 |
| C1 | 133.000 | +0.448 |
| TransactionAmt | 311.950 | +0.359 |
| ent_P_emaildomain__prior_amt_mean | 166.890 | -0.323 |
| V70 | 5.000 | -0.298 |

**Borderline** — actual label 0, calibrated P(fraud) 0.137, amount $34.00 → **ALERT** at threshold 0.137.

| feature | value | SHAP |
| --- | --- | --- |
| V294 | 1.000 | +0.357 |
| dt_day | 130.000 | +0.342 |
| dist1 | 925.000 | +0.322 |
| TransactionAmt | 34.000 | -0.244 |
| ent_addr1__prior_count | 29756.000 | -0.178 |
| M3 | 1.000 | -0.125 |

## 6. Limitations

- Anonymized features cap semantic interpretation (above).
- Holdout is a single out-of-time slice; live performance depends on drift, tracked by the monitoring layer (Section 6).
- Fraud labels arrive weeks late in production; this replay has them immediately — the label-lag caveat travels with every number that leaves the repo.

