# Fraud Detection Model + Validation Package

> The objective here is **decision quality and validation rigor** — leakage-proof
> temporal validation, calibrated probabilities, and dollar-framed operating
> decisions — not leaderboard AUC. This project is deliberately not tuned to,
> compared against, or benchmarked on the Kaggle IEEE-CIS leaderboard. It is built
> and documented the way a bank's model-risk function would expect to review it.

A calibrated fraud-detection model on the IEEE-CIS dataset with an SR 11-7-style
validation package: temporal validation, cost-framed thresholds, a deployed scoring
API on AWS, weekly replayed drift monitoring, and an independent R replication.

**Status:** under construction — see `PROJECT_SPEC.md` for the full specification and
`docs/preflight.md` for the current environment status. This README is finalized in
Phase F with the architecture diagram, results tables, and run instructions.

## How to run (bootstrap)

```bash
uv sync                                   # core env + dev tools
uv run python -m tooling.preflight        # environment + credential check
uv run ruff check . && uv run pytest      # lint + tests
```

See `CLAUDE.md` for the operating contract and `docs/decisions.md` for the
decisions log.
