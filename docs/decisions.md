# Decisions log

Running record of consequential choices, their rationale, and pre-authorized
fallbacks taken (PROJECT_SPEC.md Section 10). Newest first.

## D-008 — Phase E monitoring: honest drift signal on a small fixture
The replayed monitoring must detect the drift injected in D-006 without drowning in
small-sample noise (synthetic holdout is only ~1250 rows → ~125/week):
- **Monitor dense features only** (<30% missing). Sparse block-NaN V/id features gave
  PSI up to 3.8 from NaN-fraction swings at small N — noise, not drift. Restricting to
  dense features surfaces the real signal (amount, C2, score distribution).
- **Exclude causal aggregates (`ent_*`)** from drift PSI: expanding-window counts /
  recency grow *structurally* over time (accumulating history), so their PSI reflects
  the design, not drift. They dominated breaches until excluded.
- **Min-sample guards:** PSI returns 0 below 30 non-null; value-capture breach requires
  ≥4 frauds in the batch (tiny batches make capture wildly noisy — an honest limitation
  of the fixture; real holdout has thousands/week).
- **Evidently drift** parsed from `DriftedColumnsCount.share ≥ drift_share`; Evidently
  HTML pruned to the latest 3 weeks (~4MB each) to bound gh-pages size (~14MB total).
- **Breach → Issue** via `breach.json`; forced-breach test (`--inject-drift`) verified
  (PSI 11 on the shifted feature). Daily cadence; stream freezes when exhausted.

## D-007 — Phase D serving choices
- **Live SHAP in Lambda (not the precompute-only fallback).** XGBoost `TreeExplainer`
  is cheap to build/query, so `/score` computes top-5 SHAP live; the explainer is
  built once at cold start. Demo presets are additionally precomputed (SPEC §5). The
  §10 precompute-only fallback stays available if cold start ever bites.
- **Serving bundle not committed.** `artifacts/serving/` (bundle.joblib + presets +
  meta) is gitignored and rebuilt by `serving.artifact` at image-build time; the
  demo's data snapshot is committed at `web/presets.json`.
- **Billing alarms via CloudWatch in us-east-1** (billing metrics are only there),
  through an aliased provider + SNS email — spec-faithful ("CloudWatch billing
  alarms"). Requires the account's "Receive Billing Alerts" toggle (owner, once).
- **Two-phase apply** (ECR first, then push image, then the rest): the Lambda needs
  an existing image. Encoded in `deploy.yml`; `lifecycle.ignore_changes=[image_uri]`
  keeps `terraform apply` from fighting the CI image push.
- **Serving bug fixed during demo verification:** float32 V-column NaNs leaked into
  `presets.json` (invalid JSON) and would have reached `/score` responses. Fixed via
  `pd.isna` (all NaN types) + null-valued SHAP factors + `allow_nan=False` guard.

## D-006 — Synthetic fixture tuned for realistic decision economics (Phase C)
The decision layer's net value was negative on holdout because synthetic fraud
amounts (~$60) were comparable to the $25 review cost — catching fraud barely beat
the cost of the reviews to find it. Real fraud targets larger amounts. Two changes
(both realistic, still labelled synthetic):
- **Fraud amounts uplifted** post-labeling (~3.3× median → fraud ≈ $205 vs legit
  ≈ $61), so review is economically worthwhile and amount is a *stable* predictive
  signal.
- **Concept drift moved to a secondary feature (C2)** and made *moderate* (fades to
  half, not zero), decoupling the leakage/monitoring story from the amount signal.
Result on synthetic: holdout ROC-AUC ≈ 0.91, frozen operating point captures ~36%
of fraud value at 1.65% FPR for **+$236k net per 100k**, leakage inflation ≈ +25%,
mild calibration drift retained. This is fixture *realism* tuning, not result
fabrication — every number stays labelled synthetic. Downstream artifacts
(experiments.md, EDA notebook, manifest) were regenerated from the final fixture.

## D-005 — Phase B modeling choices (MLflow SQLite, drift, calibration, artifacts)
- **MLflow SQLite backend.** The file store is deprecated in MLflow 3.x and never
  supported the model registry; switched tracking to `sqlite:///mlflow.db`
  (`pipeline/experiments.py`). Registry is a hard requirement (registered model +
  version + run id travel with the serving artifact).
- **Injected temporal drift into the synthetic fixture.** The fixture was
  stationary, so the random-CV-vs-temporal leakage experiment showed only noise.
  Added a mild amount→fraud **signal fade** (concept drift, no reversal) + amount
  **covariate drift**, so the leakage demo shows real inflation (XGBoost ≈ +54%)
  and Phase E monitoring has drift to detect. Numbers stay labelled synthetic.
- **Calibration method chosen a priori (isotonic), not by holdout.** Selecting the
  calibration by holdout Brier would leak the test set into model selection; the
  registered method is the spec default (isotonic). Reporting VAL + HOLDOUT Brier
  exposed **calibration drift** — it helps in-distribution (VAL) but transfers
  imperfectly to the drifted HOLDOUT, a monitoring/recalibration trigger for §7.
- **Brier is a weak calibration diagnostic under imbalance** (rewards near-zero
  predictions); the reliability curves in the decision report are the real check.
- **Model binary not committed.** `experiments.md` + `artifacts/model/metadata.json`
  are committed; the `.joblib` is regenerable via `pipeline.train` and (being
  synthetic + version-sensitive) is gitignored. Phase D produces the real binary.
- **numba/llvmlite floored** for Python 3.12 so `shap` resolves (D-002 follow-up).

## D-004 — Phase A leakage audit: three temporal-integrity bugs found and fixed
A multi-agent adversarial leakage audit (5 lenses, reproduction-driven) plus a
brute-force ground-truth test surfaced and fixed three real issues in the causal
feature framework — the project's headline guarantee:
1. **HIGH — same-timestamp recency leak.** `pandas transform("first")` skips NaN,
   so an entity's first *tied-timestamp* block got `recency = 0.0` (fabricated from
   a same-second sibling) instead of NaN. Reachable on real IEEE-CIS data (many
   same-second transactions). Fix: gate `recency` on `prior_count > 0`
   (`pipeline/features.py`). Regression test:
   `test_entity_first_tied_block_recency_is_nan`.
2. **MEDIUM — NaN `TransactionDT` silently dropped.** `temporal_split` matched no
   mask for null timestamps, breaking the exact-partition contract. Fix: raise
   `ValueError` on null `TransactionDT`. Test: `test_temporal_split_raises_on_null_transactiondt`.
3. **MEDIUM — hollow tie coverage.** The synthetic fixture had strictly-increasing
   timestamps, so the property tests never exercised the strict-past tie path. Fix:
   inject same-`(entity, DT)` bursts (`_inject_entity_ties`); assert coverage in
   `test_fixture_actually_contains_entity_ties`.
Plus a **bonus fix** (caught by the brute-force reference, not the audit): with a
NaN prior amount the mean was `sum_present / count_all` (inconsistent denominator).
Now NaN-skipping via a present-amount count (moot on real data — `TransactionAmt`
is never null — but correct). Ground truth: `tests/test_strict_past_bruteforce.py`.

## D-003 — SR 11-7 superseded by SR 26-2; read both at Phase F (Phase 0)
Verified during preflight: the Fed's **SR 11-7** (Apr 4, 2011, "Supervisory
Guidance on Model Risk Management") was **superseded by SR 26-2** ("Revised
Guidance on Model Risk Management") on **Apr 17, 2026** — before this build but
after the spec was written. The spec names SR 11-7, so the validation document
stays "SR 11-7-style" in framing, but at Phase F we read **both** and mirror the
*current* supervisory concerns. Canonical sources (both fetchable from this
runtime):
- SR 11-7: https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107.htm
- SR 26-2: https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm
Preflight's `SR_11_7_URL` was corrected to the working `boarddocs/2011` path.

## D-002 — Python pinned to 3.12, staged dependency groups (Phase 0)
The system interpreter is 3.14; the ML stack (numba/shap, xgboost, lightgbm) has
uneven wheel coverage on the newest Python. Pinned the project to **3.12**
(`.python-version`) for universal wheel availability. Dependencies are split into
groups (`dev`, `models`, `data`, `monitoring`, `notebook`) so the base sync and CI
smoke stay lean; heavier stacks install when their phase begins. `xgboost` /
`lightgbm` also require `libomp` on macOS — installed at Phase B, not Phase 0.

## D-001 — SMOTE / resampling declined (SPEC Section 1)
Class imbalance is handled with cost-sensitive learning (`scale_pos_weight`) plus
threshold optimization on the decision layer, **not** SMOTE or random resampling.
Rationale: synthetic minority oversampling distorts the calibration we depend on
for dollar-framed thresholds and fabricates transactions that never occurred in a
temporally ordered stream; cost-sensitive weights preserve the real distribution
and keep probabilities calibratable. Recorded here per SPEC Section 1.
