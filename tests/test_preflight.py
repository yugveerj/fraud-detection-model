"""Structural tests for the Phase 0 preflight (tooling.preflight)."""

from __future__ import annotations

from tooling import preflight


def test_run_checks_returns_all_items():
    checks = preflight.run_checks()
    items = {c.item for c in checks}
    # Every declared essential must be probed.
    assert preflight.PHASE0_ESSENTIALS <= items
    # Each check carries a valid status.
    assert all(c.status in {preflight.PASS, preflight.ABSENT, preflight.FAIL} for c in checks)


def test_python_essential_passes_in_this_runtime():
    # We are, by definition, running under a supported interpreter here.
    py = next(c for c in preflight.run_checks() if c.item == "Python >= 3.11")
    assert py.ok


def test_markdown_renders_table_and_summary():
    md = preflight.render_markdown(preflight.run_checks())
    assert "| Item | Status |" in md
    assert "## Summary" in md
    # Every check appears as a row.
    for c in preflight.run_checks():
        assert c.item in md


def test_blocks_field_uses_known_phase_tags():
    valid = {"-", "0/A", "A", "D", "F", "D/E"}
    assert all(c.blocks in valid for c in preflight.run_checks())
