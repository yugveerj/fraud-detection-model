# Consistency sweep — champion-model references

A final repo-wide check that no pre-promotion (XGBoost-era) figure survives as a claim
about the **current, served model**. The champion is the isotonic-calibrated **LightGBM**,
MLflow registry **v2**, run `79f0f883…` (promoted over XGBoost on validation, decisions
D-011). Comparison and historical references to XGBoost are correct and are kept.

Source of truth: `artifacts/operating_point.json`, `artifacts/model/metadata.json`,
`docs/experiments.md`. Served state confirmed live by `/healthz`.

## What was checked

| # | Pattern searched | Result |
| --- | --- | --- |
| 1 | `XGBoost` as the current/served model | **No stale claims.** All shipped results prose (README, validation report, model card, decision report, experiments.md) names LightGBM. Remaining hits are the leakage-comparison table (both models, by design), the a-priori spec, and D-010/D-011 history — all legitimate. |
| 2 | old value capture `47.5%` (now **47.8%**) | Only in `decisions.md` D-010, explicitly labelled superseded-XGBoost history. `validation_report.pdf` binary match is a compressed-stream coincidence — the `.md` source has 47.8% only. `uv.lock` hits are package hashes. |
| 3 | old FPR `3.44%` (now **4.03%**) | None in prose. `uv.lock` hits are package hashes. |
| 4 | old net `$125k` (now **≈$111k**) | Only in `decisions.md` D-010 (labelled history). PDF/`uv.lock` matches are compression/hash coincidences; the `.md` source is clean. |
| 5 | old XGBoost threshold `0.1369` (now **0.1215 / 0.122**) | **None anywhere.** |
| 6 | phrase `isotonic-calibrated XGBoost` | **None.** The only `calibrated XGBoost` was the a-priori line in `PROJECT_SPEC.md`, now annotated (see below). |

## Corrections made this pass

- `PROJECT_SPEC.md` §"Experiment grid" and `CLAUDE.md` §Stack named XGBoost as the
  primary / registered model. These are the *a-priori specification*; the decisions log
  (D-011) records the promotion to LightGBM. Both lines now carry an inline pointer
  ("a-priori XGBoost; **LightGBM promoted on validation — D-011**") so a reader who does
  not reach the decisions log is not misled, while the "chose a priori, then promoted on
  the evidence" narrative is preserved.

## Confirmed-legitimate (kept as-is)

- **Leakage table** (`experiments.md`, `validation_report.md` §4.1): the XGBoost row and
  the "+17% (XGBoost) / +23% (LightGBM)" phrasing are the anti-pattern comparison — both
  models are supposed to appear.
- **`decisions.md` D-010**: the XGBoost-era headline (0.463 / 0.885 / 47.5% / $125k) is
  the running record of the real-data run when XGBoost was still champion, explicitly
  labelled superseded by D-011. Deliberate history, not a stale current claim.
- **`decisions.md` D-011**: the calibrated-VAL comparison table (XGBoost 0.6012/0.4631 vs
  LightGBM) is the promotion evidence.
- **Production label-lag caveats** ("in production, labels arrive weeks late") describe the
  real-world deployment context, not this demonstration — accurate and kept.

## Result

No stale current-model reference remains in shipped prose. The demo results card reads its
four headline numbers from `web/metrics.json`, which `tests/test_web_metrics.py` asserts
against the committed sources, so those figures cannot silently drift either.
