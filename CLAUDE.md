# CLAUDE.md — Fraud Detection Model + Validation Package (execution mode)

<!-- Revision notes: added session-start phase detection with concrete repo-state
     signals; added a Phase 0 environment preflight as the first required step;
     added the verify loop to Commands; made the author-vs-execute boundary explicit
     for R and PySpark/Databricks; made the live-AWS apply boundary explicit; added
     SR 11-7 access handling; added a post-deploy uptime check. Day estimates removed
     from the spec. -->

## What this project is
Portfolio project #3: a calibrated fraud-detection model on the IEEE-CIS dataset,
built and documented the way a bank's model-risk function expects — temporal
validation, cost-framed decisions, deployed scoring API on AWS, weekly replayed
drift monitoring, and an SR 11-7-style validation document. Full specification in
`PROJECT_SPEC.md`.

## Session start & phase detection
Run the environment preflight (Commands / `PROJECT_SPEC.md` §0.5) if it hasn't
passed this session, then determine the current phase **from repo state** and
confirm it in one line before proceeding. Signals (first not-yet-satisfied one is
the phase to work):

- No repo scaffold / preflight not green → **Phase 0** (preflight), then **A**.
- `docs/data_manifest.md` + green causality tests present, no `docs/experiments.md` → **A** done, in **B**.
- `docs/experiments.md` exported and a model registered in MLflow → **B** done.
- Decision report + operating-point statement artifact present → **C** done.
- `infra/` present with recorded live API URL / applied state → **D** done.
- `/monitoring/` on `gh-pages` has ≥1 *scheduled* run in its history → **E** done.
- `docs/validation_report.md` (+ released PDF) and rendered `docs/validation_r/` present → **F** done.

If signals conflict (e.g., partial phase), state the ambiguity in the status line
and pick the earliest incomplete phase.

## Operating mode
**You are the implementer.** Deliver working, tested, shipped software per the spec —
fast, correct, complete. No teaching mode, no comprehension checks. Do not explain
concepts unless asked. Ghostwriting is authorized: draft all prose — README, the
validation document, model card, monitoring copy, commit messages — in the owner's
established voice (plain, dry, specific, honest; the FDIC monitor repo is the
reference).

**Author vs execute.** You *author* everything, but you do not assume you can
*execute* everything. Some artifacts have a runtime the local agent may not have:
R rendering, and PySpark/Databricks. For those, write the code, then either run it
behind a runtime check or hand it to the owner to run (see Hard rules). Never
fabricate outputs for a step you couldn't execute; leave it clearly marked
"authored, not yet run" and record it in the phase status.

**Autonomy and checkpoints.** Work autonomously within a phase. At each phase
boundary, post a brief status (done / next / risks) and continue unless told to stop.
Owner approval is required only for actions that are irreversible, public, or costly:
- First public deploy (scoring API + demo page), with projected monthly cost
- Spend projected >$10/month or >$25 one-time
- Destructive operations (deleting ECR repos, Lambda functions, model artifacts)
- Any `terraform apply` against a live AWS account (owner/CI-run — see Hard rules)
- Material scope changes

## Hard rules
- IMPORTANT: **Temporal integrity is the project.** Every split is by
  `TransactionDT`; every entity-level aggregate feature is computed causally
  (past-only windows — no future rows, ever); unit tests enforce causality.
  Random cross-validation appears in exactly one place — the leakage-comparison
  experiment — and is labeled as the anti-pattern it demonstrates.
- IMPORTANT: **Never commit raw competition data.** The Kaggle IEEE-CIS data is
  pulled via the Kaggle API with owner credentials at setup, gitignored, and only
  derived artifacts (aggregates, charts, model binaries, reports) enter the repo.
- **Live-AWS boundary (owner-authorized apply — amended 2026-07-06).** The owner
  explicitly authorized agent-executed `terraform apply` for this deployment. The
  agent MAY run `apply` against the owner's AWS account, but only when ALL hold:
  (a) credentials are configured via the standard local AWS credential chain
  (`~/.aws/`, SSO, or env) — **never pasted into the conversation**; (b) the agent
  first runs `terraform plan` and presents it with the monthly cost projection;
  (c) billing alarms + API Gateway throttling are in the stack (they are). The agent
  verifies identity with `aws sts get-caller-identity` before applying and smoke-tests
  `/healthz` after. **`destroy` and any resource deletion still require explicit
  per-action owner confirmation.** `plan` / `validate` / `fmt` remain always allowed.
- **R and PySpark/Databricks are author-only unless a runtime check passes.**
  Write `docs/validation_r/replication.Rmd` and the PySpark notebook regardless.
  Render/execute R only if `Rscript` is present; otherwise mark it for the owner.
  Default Spark execution is a **local `SparkSession`** (`pyspark` installed) — do
  not depend on Databricks CE from the agent; the Databricks run, if wanted, is
  owner-optional. Keyword claim stays "PySpark (Databricks-compatible)."
- **SR 11-7 is a real document — do not paraphrase it from memory.** Before
  drafting the validation report, read the actual Federal Reserve SR 11-7 guidance.
  If this runtime has no web access, stop and request the text from the owner
  rather than synthesizing regulatory content. Mirror its concerns, not boilerplate.
- Anti-leaderboard positioning: never tune to, compare against, or mention the
  Kaggle leaderboard as a target. The README's first paragraph states the
  objective: decision quality and validation rigor, not leaderboard AUC.
- No probability is displayed anywhere (API, demo, docs) unless it comes from the
  calibrated model. Calibration is not optional.
- Honesty labels are mandatory: the drift monitoring is a **replayed simulation
  on a static dataset** — every monitoring page says so; the validation document
  carries the label-lag caveat (real fraud labels arrive weeks late). These labels
  travel with any number that leaves the repo.
- The validation document and model card reference actual artifacts and numbers
  from this repo — no generic model-risk boilerplate, no fabricated results.
- Never weaken tests or thresholds to make CI green.
- Secrets: `kaggle.json` and AWS credentials never in the repo; local via `.env`/
  standard config paths (gitignored); CI/deploy via GitHub Actions OIDC to AWS.
- Cost guard: CloudWatch billing alarms before first deploy; API Gateway
  throttling on the public endpoint; expected steady state ≈ $0–3/month.
- **Availability guard:** after deploy, a scheduled uptime check pings the demo
  page and `/healthz`; a failure opens a GitHub Issue (same pattern as drift
  breaches). A dead link when a recruiter clicks is a failure state.
- Git: small, coherent commits; never force-push; feature branches per phase.

## Stack (decided)
- Python 3.11+ via `uv`; pandas, NumPy, scikit-learn, GBMs **XGBoost +
  LightGBM** (XGBoost the a-priori primary; **LightGBM promoted to champion on
  validation — decisions D-011**; both stay in the grid for the leakage comparison),
  SHAP, matplotlib, joblib; EDA and stretch notebooks in Jupyter
- Lint/format: **ruff**; tests: **pytest** (unit + property-style causality tests)
- Experiment tracking: **MLflow** (local tracking + model registry); the
  experiment comparison table is exported to `docs/experiments.md`; the deployed
  artifact carries its registry version and run id
- Monitoring: **Evidently** (data/prediction drift) + explicit **PSI** tables,
  published as static HTML to GitHub Pages weekly by Actions cron; threshold
  breaches open a GitHub Issue automatically; a lightweight uptime job shares the
  Issue-on-failure path
- Serving: Docker container → ECR → **AWS Lambda** behind **API Gateway**
  (scale-to-zero); plain handler + pydantic validation; all AWS resources via
  **Terraform** (`infra/`), applied by owner/CI only; static demo page on GitHub
  Pages calling the API (GA4 snippet, same property as projects 1–2)
- Validation extras: independent replication of key metrics in **R**
  (R Markdown, §7b) — authored by agent, rendered by owner or a runtime-gated CI
  step; stretch: **PySpark** feature-parity notebook, local `SparkSession` by
  default (Databricks CE owner-optional)
- CI: lint, unit tests (including causality tests), seeded smoke-train, report
  generation; full retrain is a manually-triggered workflow

## Commands (keep current as the project evolves)
Build / run:
- `uv run python -m tooling.preflight` — Phase 0 environment + credential check
- `uv run python -m pipeline.fetch_data` — Kaggle pull + integrity checks
- `uv run python -m pipeline.train --full` — reproducible train (seeded)
- `uv run python -m pipeline.train --smoke` — fast seeded smoke-train (CI)
- `uv run python -m pipeline.evaluate --report` — holdout metrics + decision report
- `uv run python -m monitoring.replay_week` — score next replay batch + reports
- `docker build -t fraud-scoring .` — serving image
- `terraform -chdir=infra plan` / `fmt` / `validate` — IaC authoring (NOT `apply`)

Verify (a phase is not "done" until these pass):
- `uv run ruff check . && uv run ruff format --check .` — lint clean
- `uv run pytest` — unit + causality tests green
- `uv run python -m pipeline.train --smoke` — smoke-train passes
