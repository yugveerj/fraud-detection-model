# PROJECT_SPEC.md — Fraud Detection Model + Validation Package
Companion to `CLAUDE.md` (the operating contract — read it first; it wins on any
conflict). Execution mode: build phase by phase, post a status line at each boundary,
stop only at the approval points defined in CLAUDE.md.

<!-- Revision notes: added §0.5 environment preflight; added the uptime check
     (item 12) to "shipped"; made the live-AWS apply boundary and the R/Spark
     author-vs-execute boundary explicit (§5, §7b, §8); added an SR 11-7 access note
     (§7); removed day-based estimates from the phase plan (§8); expanded
     pre-authorized fallbacks (§10). -->

---

## 0. Mission and definition of shipped

A fraud-detection system that treats the model as the easy part and the decisions
around it as the product: leakage-proof temporal validation, calibrated
probabilities, dollar-framed thresholds, a deployed scoring service, weekly
(replayed) drift monitoring, and a validation document written to the standard a
bank's model-risk reviewer would apply.

**Shipped means ALL of:**
1. One-command reproducible pipeline: Kaggle pull → features → trained, calibrated
   model artifact (seeded; run recorded in MLflow with registry version)
2. **The leakage experiment:** random-CV vs temporal-split comparison table — same
   features, same model, both protocols — quantifying the inflation, in the README
3. Holdout metrics reported: PR-AUC (primary), ROC-AUC, recall at fixed FPR
   points, Brier score, reliability curve (pre/post calibration)
4. **Dollar-framed threshold analysis:** cost model (fraud loss = transaction
   amount; review cost = parameter), tradeoff curve, chosen operating point
   stated as "captures X% of fraud value at Y% false-positive rate ≈ $Z net per
   100k transactions," plus a sensitivity table across review-cost assumptions
5. SHAP: global importance + 3 case studies (fraud / legit / borderline), with
   the honest note that anonymized V-features limit semantic interpretation
6. `docs/experiments.md`: MLflow experiment grid exported (LR baseline, XGBoost
   variants, LightGBM comparison, calibration on/off, both validation protocols)
7. Scoring API live: all AWS resources (ECR, Lambda, API Gateway, throttling,
   alarms, OIDC deploy role) provisioned via **Terraform** in `infra/`;
   pydantic-validated `/score`, billing-alarmed; static demo page on GitHub
   Pages with three preset transactions showing calibrated score, threshold
   decision, and top contributing features
8. Weekly replayed monitoring live: ≥1 green *scheduled* run publishing Evidently
   + PSI reports to GitHub Pages `/monitoring/` with a history index; breach →
   automated GitHub Issue; every page labeled as replayed simulation
9. **Validation document** (`docs/validation_report.md`, 8–10 pages, rendered PDF
   in releases): SR 11-7-inspired structure per §7 — read the actual Federal
   Reserve SR 11-7 guidance before drafting; document must cite this repo's real
   artifacts throughout. Plus a one-page model card and the **R replication**
   per §7b, cited in the document's performance-testing section.
10. README in house voice: anti-leaderboard positioning in paragraph one,
    architecture diagram, results tables, limitations, how to run
11. Raw data never committed; CI green including causality tests
12. **Uptime check live:** a scheduled availability job pings the demo page and
    `/healthz`; failure opens a GitHub Issue. The demo must survive being clicked
    months after it was built.

---

## 0.5 Environment preflight (Phase 0 — run before Phase A)

`tooling.preflight` verifies the toolchain and credentials up front and stops with
a checklist of what's missing, rather than discovering failures mid-phase. It
checks, and reports pass/fail/absent for each:

- **Python/uv:** Python 3.11+, `uv` present, virtualenv resolvable.
- **Lint/test:** `ruff`, `pytest` importable.
- **Kaggle:** `kaggle.json` present and readable; API reachable; IEEE-CIS
  competition rules accepted (a metadata call succeeds). *Blocker for Phase A.*
- **AWS:** credentials resolve; the intended region is set; the OIDC deploy role
  exists or is documented as owner-to-create. *Blocker for Phase D, not earlier.*
- **Terraform:** binary present; `terraform -chdir=infra validate` runs once
  `infra/` exists.
- **R (optional):** `Rscript` present → R replication can be rendered in-agent;
  absent → authored only, owner renders. Not a blocker.
- **Spark (optional):** `pyspark` importable for the local-SparkSession path;
  Databricks CE is owner-optional. Not a blocker.
- **Web access for SR 11-7:** can the runtime fetch the Federal Reserve guidance?
  If not, flag that the owner must supply the text before Phase F. Not a blocker
  until Phase F.
- **GitHub:** Pages enabled, Actions permitted, OIDC to AWS configured (or
  documented as owner-to-do).

Output: a short table (item · status · what to do if absent). Non-blocking absences
are recorded and the build proceeds; blockers stop the relevant phase with the
exact remediation step. Preflight result is written to `docs/preflight.md`.

---

## 1. Data and temporal design

- Source: Kaggle IEEE-CIS Fraud Detection — `train_transaction.csv` +
  `train_identity.csv` (~590k labeled transactions; `TransactionDT` is a
  relative timestamp). The unlabeled competition test set is not used.
  Owner console prerequisite: Kaggle account, competition rules accepted, API
  token in place (verified in Phase 0).
- Integrity: record row counts, class balance, and file checksums to
  `docs/data_manifest.md` at fetch time.
- **Temporal segmentation (by `TransactionDT`, documented cut points):**
  - TRAIN ≈ first 60% — model fitting
  - VALIDATION ≈ next 15% — calibration fitting, threshold selection, model
    selection
  - HOLDOUT ≈ final 25% — reported metrics once, then consumed as the replay
    stream in weekly monitoring batches (§6)
- Class imbalance: cost-sensitive learning (`scale_pos_weight` / class weights)
  plus threshold optimization. SMOTE and resampling are deliberately declined —
  record the rationale in `docs/decisions.md`.

## 2. Features (causality is the headline constraint)

- Base: transaction amount transforms, time-of-day/period decompositions from
  `TransactionDT`, card/address/email-domain categorical handling, identity joins
- Entity aggregates (card id, email domain, address): counts, means, recency
  deltas — computed with **past-only expanding windows** relative to each row's
  `TransactionDT`. No feature may see the future.
- Enforcement: property-style unit tests that shuffle/truncate future rows and
  assert feature values for past rows are unchanged; CI runs them always
- Keep the feature set disciplined (~50–100 engineered columns); the project's
  value is validation rigor, not feature-count bragging

## 3. Modeling and experiments (MLflow-tracked)

Experiment grid (each an MLflow run with params/metrics/artifacts):
1. Logistic regression baseline (scaled numerics, target-free encodings)
2. XGBoost — primary model; modest tuning (depth, learning rate, estimators,
   `scale_pos_weight`) on VALIDATION only
3. LightGBM — single comparison configuration
4. Each of the above under BOTH protocols: temporal split (real) and 5-fold
   random CV (leakage demonstration only)
5. Calibration: isotonic (default) vs Platt on VALIDATION; report Brier +
   reliability curves; the calibrated champion becomes the registered model
   (a priori XGBoost; **LightGBM was promoted on validation — decisions D-011**)
Registry: final model registered with version + run id; the serving artifact
embeds both and `/healthz` reports them.

## 4. Decision layer

- Cost model: loss(missed fraud) = `TransactionAmt`; cost(review) = parameter
  (default $25, sensitivity over $5–$75). Optimize expected net value on
  VALIDATION across thresholds; freeze the operating point; report on HOLDOUT.
- Outputs (figures produced with matplotlib): value-capture curve (fraud
  dollars caught vs review volume), the operating-point statement of §0.4,
  and the sensitivity table
- SHAP on the registered model: global bar + beeswarm; 3 case studies for the
  demo presets; interpretation caveats stated
- The decision report is written for a reviewer deciding whether to operate the
  model (dollar-framed), not how it was trained.

## 5. Serving

- Lambda container image (Docker → ECR; python base pinned; model + SHAP
  explainer loaded at init) behind an API Gateway HTTP API (region us-west-2,
  matching project 2's AWS footprint)
- `/score`: pydantic-validated single transaction → calibrated probability,
  threshold decision at the frozen operating point, top-5 SHAP contributions;
  `/healthz`: model version, registry run id
- API Gateway throttling (default 5 rps / burst 10) + usage plan; CloudWatch
  billing alarms at $10/$25; CORS restricted to the GitHub Pages origin
- **Infrastructure as code:** every AWS resource above is defined in Terraform
  under `infra/` (local state or Terraform Cloud free tier); once IaC lands,
  console-only changes are prohibited — changes go through `terraform plan`
  in a PR
- **Apply boundary:** the agent authors all Terraform and the deploy workflow and
  runs `plan` / `validate` / `fmt` only. `terraform apply` against the live AWS
  account is owner-run or executed by owner-triggered CI (GitHub Actions OIDC);
  the agent does not mutate a real cloud account. This is the mechanism of the
  deploy gate below.
- **Uptime:** once live, a scheduled Actions job pings the demo page and
  `/healthz` and opens a GitHub Issue on failure (shares the breach-Issue path).
- Demo page (static, GitHub Pages): three preset transactions (legit / fraud /
  borderline drawn from HOLDOUT patterns, values rounded — presets are
  representative, not raw dataset rows), score button, decision + factors
  display, GA4 snippet (same property), footer: "Demonstration system on a
  public research dataset — not a production fraud decision."
- **Deploy gate (owner approval):** projected monthly cost (expected ≈ $0–3)
  before the API and page go public; the actual `apply` is owner/CI, per the
  apply boundary above.

## 6. Monitoring (replayed, honestly labeled)

- Actions cron consumes the next HOLDOUT slice; each run = one simulated week
  of `TransactionDT`. Cadence is configurable: weekly by default, daily is
  acceptable to accrue monitoring history faster — state the cadence on the
  monitoring index either way. The stream covers ~20 simulated weeks, then
  freezes with a final report
- Each run: score the batch via the deployed artifact → Evidently data-drift and
  prediction-drift reports + **PSI table** (per top-20 features and for the score
  distribution) + performance-on-labels section carrying the label-lag caveat
  (real labels arrive weeks late; this replay has them immediately)
- Publish HTML to `gh-pages` `/monitoring/` with an index (latest + history);
  every page banner: "Replayed simulation on a static public dataset."
- Alerting: PSI > 0.2 on any tracked feature, dataset drift flagged, or
  value-capture at the operating point degrading >10% vs baseline → the cron
  opens a GitHub Issue with the offending table embedded. The uptime check (§5)
  uses the same Issue-on-failure path.
- Thresholds, escalation, retrain triggers, and a champion/challenger protocol
  are defined in the monitoring-plan section of the validation document

## 7. Validation document (`docs/validation_report.md`, 8–10 pages)

Read the Federal Reserve's SR 11-7 guidance first; mirror its concerns, not its
boilerplate. **If the runtime has no web access, request the guidance text from
the owner before drafting — do not synthesize regulatory content from memory**
(flagged in Phase 0). Sections: (1) purpose and intended use, incl. explicit
non-production status; (2) data — source, manifest, known dataset limitations;
(3) methodology and assumptions — features, causality controls, imbalance
handling, calibration; (4) performance testing — both validation protocols and
the leakage delta, holdout metrics, stability across time slices of HOLDOUT;
(5) decision framework — cost model, operating point, sensitivity; (6)
limitations — anonymized features, replay vs production gaps, label lag,
dataset vintage; (7) ongoing monitoring plan — §6 thresholds, alerting,
retrain and champion/challenger protocol. Every claim cites a repo artifact.
Model card: one page, standard fields, linked from README.

## 7b. Independent replication in R (`docs/validation_r/replication.Rmd`)

Model-risk practice replicates a developer's results independently ("effective
challenge") — this artifact does exactly that, in the language validation teams
actually use. From an exported holdout scores CSV (id, label, calibrated
probability, amount): independently recompute PR-AUC, the reliability curve and
Brier score, the PSI table for the score distribution, and the value-capture
figure at the frozen operating point — in R, with no imports from the Python
codebase. Render to HTML, commit with outputs, reconcile any discrepancies
beyond tolerance in a short findings note, and cite the replication in §7's
performance-testing section.

**Author vs execute:** the agent authors the `.Rmd` regardless. Rendering happens
in-agent only if `Rscript` is present (Phase 0), otherwise the owner renders it;
either way the numbers must match the Python results or the mismatch is explained.
Do not fabricate the rendered outputs — an authored-but-unrendered `.Rmd` is a
valid interim state, marked as such.

## 8. Phase plan
(Ordering, not a schedule — timeline is out of scope. Post a status line at each
boundary; stop only at the defined approval points.)

**Phase 0 — Environment preflight.** `tooling.preflight` per §0.5; write
`docs/preflight.md`; resolve or record blockers.
Checkpoint: preflight table; proceed if Phase A prerequisites are green.

**Phase A — Data + temporal framework.** Fetch pipeline, manifest, segmentation,
causal-aggregate framework + causality tests green in CI; an **EDA notebook**
(`docs/notebooks/eda.ipynb`) committed with outputs — class balance, amount
distributions, missingness, `TransactionDT` coverage.
Checkpoint: status + class-balance/segment table; proceed.

**Phase B — Experiments.** Full §3 grid in MLflow; calibration; registry;
`docs/experiments.md` exported; leakage table produced.
Stretch (optional, non-blocking, may land in any later phase): re-express the
causal aggregate features in **PySpark** — `docs/notebooks/feature_engineering_pyspark.ipynb`,
executed on a **local `SparkSession`** by default (Databricks CE owner-optional),
committed with outputs including a parity check against the pandas implementation
on a sample. If neither Spark runtime is available, the notebook is authored and
marked "authored, not yet run."
Checkpoint: status + experiment table; proceed.

**Phase C — Decision layer.** §4 complete; decision report generated by one
command.
Checkpoint: status + operating-point statement; proceed.

**Phase D — Serving.** §5 complete, all AWS resources authored in Terraform per
§5's IaC and apply-boundary bullets. **Deploy gate (owner approval);** the `apply`
is owner/CI. Uptime check authored.
Checkpoint: live URL + cost projection.

**Phase E — Monitoring.** §6 complete; first scheduled run green; alert path tested
with a forced breach on a scratch branch; uptime job scheduled.
Checkpoint: status + first monitoring report link; proceed.

**Phase F — Validation document + ship.** §7 and §7b complete (SR 11-7 read or
owner-supplied); README final; §0 checklist verified end to end.
Checkpoint: ship report.

## 9. Costs (steady state)

Lambda + API Gateway at demo traffic ≈ $0–2/month · ECR ≈ pennies · GitHub
Pages/Actions free tier (uptime + monitoring crons included) · MLflow local = $0 ·
no LLM costs in this project. Billing alarms mandatory before the deploy gate.

## 10. Pre-authorized fallbacks (announce, record in decisions.md, proceed)

- SHAP-in-Lambda cold start too heavy → precompute explanations for the demo
  presets and serve top-factors from the model's native importances for ad-hoc
  requests, labeled accordingly
- Lambda image size limits bite → strip the explainer to the serving path or
  step up to a slim App Runner service; re-present cost if >$10/month
- Kaggle API friction in CI → data fetch becomes a local-only step; CI uses a
  committed 5k-row synthetic schema-identical fixture (clearly labeled synthetic,
  used for tests only — never for reported metrics)
- Replay stream exhausts before season ends → freeze with a final report rather
  than looping data, and say so on the monitoring index
- **R runtime absent** → author the `.Rmd`, mark it "authored, not yet run,"
  leave the reconciliation for owner render; do not fabricate rendered outputs
- **No Spark runtime** (local or Databricks) → author the PySpark notebook only,
  mark it "authored, not yet run"; the keyword claim stays "PySpark
  (Databricks-compatible)," never "production Spark"
- **No web access for SR 11-7** → pause the validation-document draft and request
  the guidance text from the owner; proceed on the rest of the phase meanwhile
- **Live-cloud action required but blocked from the agent** (any `apply`/`destroy`)
  → prepare the `terraform plan` output and the exact command, hand to owner/CI,
  and continue with non-cloud work
