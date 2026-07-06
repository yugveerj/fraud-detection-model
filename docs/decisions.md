# Decisions log

Running record of consequential choices, their rationale, and pre-authorized
fallbacks taken (PROJECT_SPEC.md Section 10). Newest first.

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
