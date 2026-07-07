# Data manifest

- **Provenance:** real
- **Generated (UTC):** 2026-07-06T23:33:11+00:00

## Transactions

| metric | value |
| --- | --- |
| rows | 590,540 |
| columns | 394 |
| fraud count | 20,663 |
| fraud rate | 3.4990% |
| TransactionDT range | 86,400 – 15,811,131 |
| TransactionAmt range | 0.25 – 31,937.39 |

## Identity (left-joined subset)

| metric | value |
| --- | --- |
| rows | 144,233 |
| columns | 41 |
| coverage of transactions | 24.42% |

## Temporal segmentation (by TransactionDT)

Cut points: train|val = `8745782.399999999`, val|holdout = `11246620.0`.

| split | rows | frac | dt_min | dt_max | fraud_rate |
| --- | --- | --- | --- | --- | --- |
| train | 354,324 | 0.6 | 86400 | 8745772 | 0.03383 |
| val | 88,581 | 0.15 | 8745798 | 11246605 | 0.04036 |
| holdout | 147,635 | 0.25 | 11246665 | 15811131 | 0.03454 |

## Integrity

| file | size (bytes) | sha256 |
| --- | --- | --- |
| train_transaction.csv | 683,351,067 | `3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642` |
| train_identity.csv | 26,529,680 | `b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c` |
