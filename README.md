# Fraud Detection Model + Validation Package

**The objective here is decision quality and validation rigor — not leaderboard AUC.**
This project is deliberately *not* tuned to, compared against, or benchmarked on the
Kaggle IEEE-CIS leaderboard. It treats the model as the easy part and the decisions
around it as the product: leakage-proof temporal validation, calibrated probabilities,
a dollar-framed operating point, a deployed scoring API, replayed drift monitoring, and
an SR 11-7-style validation document — built and documented the way a bank's model-risk
function would expect to review it.

> **Data.** The numbers below are computed on the **real Kaggle IEEE-CIS Fraud Detection
> data** (590,540 transactions). Everything reproduces with one command
> (`uv run python -m pipeline.train --full`); with no Kaggle credentials the pipeline
> falls back to a schema-identical synthetic fixture so tests and CI still run.

## Live

- **Demo:** <https://yugveerj.github.io/fraud-detection-model/> — score three preset
  transactions against the deployed model (calibrated probability, dollar-framed
  decision, live SHAP factors).
- **API:** `https://mrx6i8np6k.execute-api.us-east-2.amazonaws.com` — `GET /healthz`,
  `POST /score` (pydantic-validated). Deployed on AWS Lambda + API Gateway via Terraform
  ([`infra/`](infra/)): arm64/Graviton, throttled, billing-alarmed, scale-to-zero
  (≈ $0–3/mo).
- **Monitoring:** <https://yugveerj.github.io/fraud-detection-model/monitoring/> —
  replayed Evidently + PSI drift, refreshed by a scheduled GitHub Action; a scheduled
  uptime check pings the demo + `/healthz` and opens an Issue on failure.

## What it does

```mermaid
flowchart LR
  K[Kaggle IEEE-CIS] -->|fetch + manifest| D[Temporal split<br/>60/15/25 by TransactionDT]
  D --> F[Causal features<br/>strict-past aggregates]
  F --> M[LightGBM + isotonic calibration<br/>MLflow registry]
  M --> C[Decision layer<br/>dollar-framed threshold]
  C --> S[Lambda + API Gateway<br/>/score /healthz]
  S --> W[Static demo<br/>GitHub Pages]
  M --> V[Validation report<br/>SR 11-7 + R replication]
  D -->|holdout replay| MON[Drift monitoring<br/>Evidently + PSI → Pages]
  MON -->|breach| ISS[GitHub Issue]
```

## Headline results (real IEEE-CIS holdout)

**The leakage experiment** — same features, same model, two validation protocols
([full grid](docs/experiments.md)):

| model | temporal-val PR-AUC | random-CV PR-AUC | inflation |
| --- | --- | --- | --- |
| logistic regression | 0.496 | 0.467 | −6% |
| XGBoost | 0.606 | 0.711 | **+17%** |
| LightGBM | 0.621 | 0.765 | **+23%** |

Random cross-validation trains on future-dated rows and reports an optimistic number.
The tree models — the ones deployed — inflate **+17–23%**; the linear model, which
doesn't exploit temporal structure, doesn't. That inflation is why every split here is
temporal.

**Holdout** (isotonic-calibrated LightGBM, 147,635 transactions): PR-AUC **0.464**,
ROC-AUC **0.879**, Brier **0.024** (0.052 uncalibrated — calibration matters).
**Operating point:** captures **47.8% of fraud value at 4.03% FPR** (recall 55.3%) for
**≈ $111k net per 100k transactions** (review cost $25), optimized on validation and
reported on the temporal holdout. Independently reproduced in R
([`docs/validation_r/`](docs/validation_r/replication.Rmd)). LightGBM is the champion,
promoted over XGBoost on validation ([decisions D-011](docs/decisions.md)).

**Features:** the model sees **436 columns — 415 raw** IEEE-CIS fields (400 numeric + 15
categorical) plus **21 engineered** (9 per-transaction transforms + 12 **strict-past
causal aggregates** per card / email / address). The 21 engineered features are the ones
the causality tests guard; the raw `V*`/`id_*` fields are anonymized and used as-is.

## Read the work

| Artifact | What it is |
| --- | --- |
| [Validation report](docs/validation_report.md) | SR 11-7 / SR 26-2-style model validation, cites every artifact |
| [Model card](docs/model_card.md) | one-page standard fields |
| [Decision report](docs/decision_report.md) | dollar-framed cost model, operating point, SHAP |
| [Experiments](docs/experiments.md) | full grid, leakage table, calibration, registry |
| [R replication](docs/validation_r/replication.Rmd) | independent recompute (effective challenge) |
| [EDA notebook](docs/notebooks/eda.ipynb) | class balance, distributions, missingness, coverage |
| [Decisions log](docs/decisions.md) | consequential choices + rationale |
| [Infrastructure](infra/README.md) | Terraform + cost projection + deploy runbook |

## How to run

```bash
uv sync                                        # core env + dev tools
uv run python -m tooling.preflight             # environment + credential check
uv run ruff check . && uv run pytest           # lint + tests (incl. causality tests)

# With Kaggle credentials (real data); otherwise everything falls back to the fixture:
uv run python -m pipeline.fetch_data           # Kaggle pull + manifest
uv run python -m pipeline.train --full         # train, calibrate, register (MLflow)
uv run python -m pipeline.evaluate --report    # decision report + figures + R scores
uv run python -m monitoring.replay_week --all  # replay the holdout → drift reports
uv run python -m serving.artifact              # build the serving bundle
docker build -t fraud-scoring .                # Lambda image (serving)
terraform -chdir=infra validate                # infrastructure (apply is owner/CI)
```

## What's shipped

Reproducible pipeline · the leakage experiment · calibrated holdout metrics ·
dollar-framed thresholds + sensitivity · SHAP · MLflow registry · Terraform-defined
scoring API (Lambda + API Gateway, throttled, billing-alarmed) · static demo · replayed
Evidently + PSI monitoring with breach → Issue · uptime check · SR 11-7-style validation
document + model card + R replication. Serving/infra are **authored**; the public
`terraform apply` is the owner's gated action (never the agent's).

## Design constraints (non-negotiable)

- **Temporal integrity is the project.** Every split is by `TransactionDT`; every entity
  aggregate is strictly past-only; property + brute-force tests enforce it. Random CV
  appears in exactly one place — the leakage experiment — labelled as the anti-pattern.
- **Calibration is not optional.** No probability is displayed unless it is calibrated.
- **Honesty labels travel with the numbers.** Monitoring is a replayed simulation on a
  static dataset; every page says so. The validation document carries the label-lag
  caveat. Raw competition data is never committed.

## Limitations

Anonymized `V*`/`id_*` features cap interpretability; the holdout is a single out-of-time
slice; production fraud labels lag weeks (this replay has them immediately); no fairness
assessment (the dataset lacks protected attributes). Full treatment:
[validation report §6](docs/validation_report.md#6-limitations).
