# Decisions log

Running record of consequential choices, their rationale, and pre-authorized
fallbacks taken (PROJECT_SPEC.md Section 10). Newest first.

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
