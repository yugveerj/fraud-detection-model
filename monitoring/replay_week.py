"""Replay the next monitoring week (SPEC Section 6).

Reads the cursor from ``<out>/state.json``, scores + assesses the next holdout slice,
writes its report, refreshes the index, and records any breach to ``<out>/breach.json``
(the cron reads that to open a GitHub Issue). The stream freezes with a final report
when exhausted.

    uv run python -m monitoring.replay_week                         # advance one week
    uv run python -m monitoring.replay_week --all                    # backfill every week
    uv run python -m monitoring.replay_week --reset                  # start the stream over
    uv run python -m monitoring.replay_week --inject-drift           # forced-breach test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from monitoring import replay

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "site" / "monitoring"
KEEP_EVIDENTLY = 3  # full Evidently HTML (~4MB each) kept only for the latest weeks


def _prune_evidently(out_dir: Path, keep: int = KEEP_EVIDENTLY) -> None:
    """Bound gh-pages size: replace older weeks' Evidently HTML with a pointer."""
    weeks = sorted(out_dir.glob("week-*"))
    for week_dir in weeks[:-keep] if keep else weeks:
        ev = week_dir / "evidently.html"
        if ev.exists() and ev.stat().st_size > 5000:
            ev.write_text(
                "<p>Evidently report pruned to bound site size. The PSI summary for this "
                "week is in report.html; full Evidently reports are kept for recent weeks.</p>",
                encoding="utf-8",
            )


def _load_state(out_dir: Path) -> dict | None:
    path = out_dir / "state.json"
    return json.loads(path.read_text()) if path.exists() else None


def _init_state(ctx: replay.MonitorContext) -> dict:
    return {
        "provenance": ctx.provenance,
        "n_batches": ctx.n_batches,
        "cursor": 0,
        "baseline_value_capture": ctx.baseline_value_capture,
        "history": [],
        "frozen": False,
    }


def _write_site(out_dir: Path, ctx: replay.MonitorContext, state: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(
        replay.render_index_html(state["history"], ctx.provenance, state["frozen"]),
        encoding="utf-8",
    )
    (out_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def _run_once(out_dir: Path, ctx: replay.MonitorContext, state: dict) -> dict:
    idx = state["cursor"]
    if idx >= ctx.n_batches:
        state["frozen"] = True
        return state
    result = replay.run_batch(ctx, idx, out_dir)
    state["history"].append(result)
    state["cursor"] = idx + 1
    state["frozen"] = state["cursor"] >= ctx.n_batches
    if result["breaches"]:
        (out_dir / "breach.json").write_text(
            json.dumps({"week": idx, "breaches": result["breaches"]}, indent=2), encoding="utf-8"
        )
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay the next monitoring week")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--source", default="auto", choices=["auto", "real", "synthetic"])
    parser.add_argument("--profile", default="full", choices=["full", "smoke"])
    parser.add_argument("--n-batches", type=int, default=replay.DEFAULT_BATCHES)
    parser.add_argument("--all", action="store_true", help="run every remaining week now")
    parser.add_argument("--reset", action="store_true", help="restart the replay stream")
    parser.add_argument(
        "--inject-drift", action="store_true", help="perturb the stream to force a breach (test)"
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    ctx = replay.prepare(source=args.source, profile=args.profile, n_batches=args.n_batches)

    if args.inject_drift and ctx.numeric_cols:
        col = ctx.numeric_cols[0]  # most important feature → guaranteed PSI breach
        ctx.holdout[col] = ctx.holdout[col].astype(float) + 6.0 * ctx.holdout[col].std()

    # Stale breach flag from a prior run must not linger.
    (out_dir / "breach.json").unlink(missing_ok=True)

    state = None if args.reset else _load_state(out_dir)
    if state is None:
        state = _init_state(ctx)

    if args.all:
        while state["cursor"] < ctx.n_batches:
            state = _run_once(out_dir, ctx, state)
    else:
        state = _run_once(out_dir, ctx, state)

    _write_site(out_dir, ctx, state)
    _prune_evidently(out_dir)

    done = state["cursor"]
    breach = (out_dir / "breach.json").exists()
    print(
        f"provenance={ctx.provenance} weeks_done={done}/{ctx.n_batches} "
        f"frozen={state['frozen']} breach={breach}"
    )
    print(f"Wrote {out_dir}/ (index.html + week reports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
