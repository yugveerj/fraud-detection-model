# Data manifest

> **SYNTHETIC FIXTURE — NOT REAL DATA.** These numbers describe the schema-identical synthetic fixture used for tests/CI (SPEC Section 10). They are regenerated from the real IEEE-CIS competition data at fetch time (`uv run python -m pipeline.fetch_data`) and must never be cited as results.

- **Provenance:** synthetic
- **Generated (UTC):** 2026-07-06T04:04:08+00:00

## Transactions

| metric | value |
| --- | --- |
| rows | 5,000 |
| columns | 394 |
| fraud count | 175 |
| fraud rate | 3.5000% |
| TransactionDT range | 86,436 – 1,083,726 |
| TransactionAmt range | 1.55 – 2,042.56 |

## Identity (left-joined subset)

| metric | value |
| --- | --- |
| rows | 1,208 |
| columns | 41 |
| coverage of transactions | 24.16% |

## Temporal segmentation (by TransactionDT)

Cut points: train|val = `687878.4`, val|holdout = `839182.25`.

| split | rows | frac | dt_min | dt_max | fraud_rate |
| --- | --- | --- | --- | --- | --- |
| train | 3,000 | 0.6 | 86436 | 687754 | 0.03867 |
| val | 750 | 0.15 | 688065 | 839124 | 0.03467 |
| holdout | 1,250 | 0.25 | 839357 | 1083726 | 0.0264 |

## Integrity

No raw file checksums (synthetic fixture). Generator parameters:

- module: `pipeline.synthetic`  ·  n = 5000  ·  seed = 42
