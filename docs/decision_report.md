# Decision report — is this model worth operating?

This report is written for a reviewer deciding whether to **operate** the model, not how it was trained. Everything is framed in dollars.

## 1. Cost model

- Reviewing a transaction costs **$25** (business parameter).
- Catching a fraudulent transaction saves its `TransactionAmt` (loss avoided).
- A missed fraud loses its `TransactionAmt`.
- **Net value** of an operating point = fraud dollars caught − review cost × alerts.

The threshold is optimized on **VALIDATION** and reported on **HOLDOUT** — never tuned on the data it is reported on.

## 2. Operating point

**At the frozen operating point (threshold 0.1215, review cost $25), the model captures 47.8% of fraud value at a 4.03% false-positive rate, for approximately $111,346 net value per 100k transactions.**

| metric | VALIDATION (optimized) | HOLDOUT (reported) |
| --- | --- | --- |
| threshold | 0.1215 | 0.1215 (frozen) |
| alert rate | 6.92% | 5.80% |
| recall (count) | 67.8% | 55.3% |
| false-positive rate | 4.36% | 4.03% |
| fraud value captured | 61.1% | 47.8% |
| net value / 100k | $250,929 | $111,346 |

The holdout capture is lower than validation: an out-of-time slice under drift is genuinely harder, and the frozen threshold does not perfectly transfer. Net value remains positive, and the gap is exactly what ongoing monitoring watches (validation report Section 7).

![value capture](figures/value_capture.png)

![net value](figures/net_value.png)

## 3. Review-cost sensitivity

How the optimal operating point shifts as the review-cost assumption varies ($5–$75). Higher review cost → higher threshold → fewer, higher-precision alerts.

| review cost | opt threshold | alert rate | recall | fpr | value capture | net / 100k |
| --- | --- | --- | --- | --- | --- | --- |
| $5 | 0.0291 | 22.25% | 88.1% | 19.48% | 85.3% | $480,986 |
| $15 | 0.0704 | 10.61% | 75.5% | 7.88% | 70.1% | $327,499 |
| $25 | 0.1215 | 6.92% | 67.8% | 4.36% | 61.1% | $250,929 |
| $50 | 0.2479 | 3.59% | 55.5% | 1.41% | 43.4% | $121,934 |
| $75 | 0.4838 | 2.27% | 44.5% | 0.49% | 31.0% | $44,835 |

## 4. Calibration

Holdout Brier 0.0235. Under class imbalance the Brier score is dominated by the rare-positive base rate; the reliability curve is the real check. Isotonic calibration corrects the probability *level* the dollar thresholds depend on. Where the holdout curve departs from validation, that is calibration drift.

**Method tradeoff.** Isotonic is not free on ranking: it scores holdout PR-AUC **0.4642** versus **0.4789** for the rank-preserving alternatives (the raw model and Platt scaling), because its step function ties large blocks of transactions and PR-AUC penalises the ambiguous within-tie ordering. Isotonic is kept as champion anyway — it wins Brier and the reliability the dollar layer needs, and the operating point is chosen on *value*, not raw ranking — while the Platt-calibrated model is registered as the named challenger so the choice stays auditable (validation report Section 7).

![reliability](figures/reliability.png)

## 5. Feature attribution (SHAP)

Global importance on the base gradient-boosted model (TreeExplainer). **Caveat:** many inputs are anonymized Vesta `V*` / `id_*` features with no published meaning, so SHAP shows *which engineered inputs* move a score, not a mechanistic business reason.

![shap global](figures/shap_global_bar.png)

![shap beeswarm](figures/shap_beeswarm.png)

### Case studies (HOLDOUT)

**Fraud** — actual label 1, calibrated P(fraud) 1.000, amount $100.00 → **ALERT** at threshold 0.122. (an isotonic upper-bin value — the calibrator's top bin is saturated, so this is that bin's empirical fraud rate, not a claim of certainty)

| feature | value | SHAP |
| --- | --- | --- |
| V258 | 3.000 | +1.227 |
| C1 | 25.000 | +0.900 |
| C13 | 1.000 | +0.871 |
| V256 | 3.000 | +0.660 |
| V189 | 3.000 | +0.659 |
| C11 | 14.000 | +0.455 |

**Legit** — actual label 0, calibrated P(fraud) 0.000, amount $29.26 → **pass** at threshold 0.122.

| feature | value | SHAP |
| --- | --- | --- |
| V55 | 6.000 | -1.408 |
| card3 | 106.000 | -0.901 |
| id_17 | 159.000 | -0.724 |
| id_20 | 633.000 | -0.704 |
| ent_card1__prior_amt_mean | 32.541 | -0.457 |
| TransactionAmt | 29.264 | -0.359 |

**Borderline** — actual label 0, calibrated P(fraud) 0.122, amount $117.00 → **ALERT** at threshold 0.122.

| feature | value | SHAP |
| --- | --- | --- |
| card6 | 1.000 | +0.357 |
| D2 | 131.000 | -0.228 |
| D15 | nan | +0.222 |
| dt_hour | 9.000 | +0.221 |
| dt_hour_sin | 0.707 | +0.202 |
| C13 | 21.000 | -0.184 |

## 6. Limitations

- Anonymized features cap semantic interpretation (above).
- Holdout is a single out-of-time slice; live performance depends on drift, tracked by the monitoring layer (validation report Section 7).
- Fraud labels arrive weeks late in production; this replay has them immediately — the label-lag caveat travels with every number that leaves the repo.

