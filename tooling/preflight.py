"""Phase 0 environment preflight (PROJECT_SPEC.md Section 0.5).

Verifies the toolchain and credentials up front and reports pass / absent / fail
for each item, with the remediation step and the phase each item blocks. Writes a
markdown table to ``docs/preflight.md``.

Deliberately stdlib-only so it runs before the heavy stack is installed. Exit code
reflects *Phase 0 essentials* only (Python/uv/ruff/pytest/git); everything else is
reported with its blocking phase but does not fail the command, because later-phase
blockers have documented fallbacks (PROJECT_SPEC.md Section 10) and are handled when
their phase begins.

    uv run python -m tooling.preflight            # report + write docs/preflight.md
    uv run python -m tooling.preflight --strict   # non-zero if ANY blocker is unmet
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREFLIGHT_DOC = REPO_ROOT / "docs" / "preflight.md"
SR_11_7_URL = "https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107.htm"

PASS = "PASS"
ABSENT = "ABSENT"
FAIL = "FAIL"

_STATUS_ICON = {PASS: "✅", ABSENT: "⚠️", FAIL: "❌"}


@dataclass
class Check:
    item: str
    status: str
    detail: str
    remediation: str
    blocks: str  # which phase this blocks, or "-" for none

    @property
    def ok(self) -> bool:
        return self.status == PASS


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def _run(args: list[str], timeout: int = 15) -> tuple[int, str]:
    """Run a command, returning (exit_code, combined_output). -1 on OSError."""
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env dependent
        return -1, str(exc)


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# --------------------------------------------------------------------------- checks


def check_python() -> Check:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 11)
    return Check(
        "Python >= 3.11",
        PASS if ok else FAIL,
        f"running {v.major}.{v.minor}.{v.micro}",
        "install Python 3.11+ (project pins 3.12 via .python-version)",
        "-" if ok else "0/A",
    )


def check_uv() -> Check:
    path = _which("uv")
    if not path:
        return Check("uv", ABSENT, "not on PATH", "brew install uv (or astral.sh/uv)", "0/A")
    code, out = _run(["uv", "--version"])
    return Check(
        "uv", PASS if code == 0 else FAIL, out or path, "reinstall uv", "-" if code == 0 else "0/A"
    )


def check_ruff() -> Check:
    if _importable("ruff") or _which("ruff"):
        code, out = _run(["uv", "run", "ruff", "--version"]) if _which("uv") else (0, "importable")
        return Check("ruff", PASS, out or "available", "-", "-")
    return Check("ruff", ABSENT, "not importable", "uv sync (dev group)", "0/A")


def check_pytest() -> Check:
    if _importable("pytest"):
        return Check("pytest", PASS, "importable", "-", "-")
    return Check("pytest", ABSENT, "not importable", "uv sync (dev group)", "0/A")


def check_git() -> Check:
    path = _which("git")
    if not path:
        return Check("git", ABSENT, "not on PATH", "install git", "0/A")
    code, out = _run(["git", "--version"])
    return Check("git", PASS if code == 0 else FAIL, out, "-", "-")


def check_kaggle() -> Check:
    pkg = _importable("kaggle") or _which("kaggle")
    cred = (Path.home() / ".kaggle" / "kaggle.json").exists() or bool(os.environ.get("KAGGLE_KEY"))
    if pkg and cred:
        return Check(
            "Kaggle (data pull)",
            PASS,
            "package + credentials present",
            "-",
            "-",
        )
    if pkg and not cred:
        return Check(
            "Kaggle (data pull)",
            ABSENT,
            "package present, credentials MISSING",
            "place kaggle.json at ~/.kaggle/ (chmod 600) and accept IEEE-CIS rules; "
            "else CI uses the synthetic fixture (SPEC Section 10)",
            "A",
        )
    return Check(
        "Kaggle (data pull)",
        ABSENT,
        "package not installed" + ("" if cred else ", credentials missing"),
        "uv sync --group data; supply kaggle.json; accept competition rules",
        "A",
    )


def check_aws() -> Check:
    if not _which("aws"):
        return Check(
            "AWS CLI + creds",
            ABSENT,
            "aws CLI not on PATH",
            "install AWS CLI v2; configure creds/region (owner)",
            "D",
        )
    code, out = _run(["aws", "sts", "get-caller-identity"], timeout=20)
    if code == 0:
        return Check("AWS CLI + creds", PASS, "credentials resolve", "-", "-")
    return Check(
        "AWS CLI + creds",
        ABSENT,
        "CLI present, credentials do not resolve",
        "configure AWS creds + region us-west-2; create OIDC deploy role (owner)",
        "D",
    )


def check_terraform() -> Check:
    if not _which("terraform"):
        return Check(
            "Terraform",
            ABSENT,
            "not on PATH",
            "install Terraform (needed for `terraform validate` once infra/ exists)",
            "D",
        )
    code, out = _run(["terraform", "--version"])
    first = out.splitlines()[0] if out else ""
    return Check("Terraform", PASS if code == 0 else FAIL, first, "-", "-" if code == 0 else "D")


def check_r() -> Check:
    if not _which("Rscript"):
        return Check(
            "R (Rscript)",
            ABSENT,
            "not present — R replication authored only, owner renders",
            "install R to render docs/validation_r/ in-agent (optional)",
            "-",
        )
    code, out = _run(["Rscript", "--version"])
    return Check(
        "R (Rscript)",
        PASS if code == 0 else FAIL,
        (out.splitlines()[0] if out else "").strip() or "present",
        "-",
        "-",
    )


def check_spark() -> Check:
    pyspark = _importable("pyspark")
    java = _which("java") is not None
    if pyspark and java:
        return Check("PySpark (local)", PASS, "pyspark + Java present", "-", "-")
    missing = []
    if not pyspark:
        missing.append("pyspark not installed")
    if not java:
        missing.append("no Java runtime")
    return Check(
        "PySpark (local)",
        ABSENT,
        "; ".join(missing) + " — notebook authored, not executed",
        "install a JRE + `uv add pyspark` for local SparkSession (optional; SPEC Section 10)",
        "-",
    )


def check_web_sr117() -> Check:
    req = urllib.request.Request(SR_11_7_URL, headers={"User-Agent": "preflight/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 - fixed trusted URL
            code = resp.getcode()
        if code == 200:
            return Check(
                "Web access (SR 11-7)",
                PASS,
                "federalreserve.gov reachable",
                "-",
                "-",
            )
        return Check(
            "Web access (SR 11-7)",
            ABSENT,
            f"HTTP {code}",
            "owner supplies SR 11-7 text before Phase F (SPEC Section 10)",
            "F",
        )
    except OSError as exc:
        return Check(
            "Web access (SR 11-7)",
            ABSENT,
            f"unreachable ({type(exc).__name__})",
            "owner supplies SR 11-7 guidance text before Phase F drafting",
            "F",
        )


def check_github() -> Check:
    if not _which("gh"):
        return Check(
            "GitHub (gh + Pages/Actions/OIDC)",
            ABSENT,
            "gh CLI not on PATH",
            "install gh; enable Pages + Actions; configure OIDC-to-AWS (owner)",
            "D/E",
        )
    code, out = _run(["gh", "auth", "status"], timeout=15)
    if code == 0:
        return Check(
            "GitHub (gh + Pages/Actions/OIDC)",
            PASS,
            "gh authenticated (Pages/Actions/OIDC are owner-configured)",
            "confirm Pages + Actions enabled; OIDC deploy role wired (owner)",
            "-",
        )
    return Check(
        "GitHub (gh + Pages/Actions/OIDC)",
        ABSENT,
        "gh present, not authenticated",
        "gh auth login; enable Pages + Actions; configure OIDC (owner)",
        "D/E",
    )


def check_docker() -> Check:
    if not _which("docker"):
        return Check(
            "Docker",
            ABSENT,
            "not on PATH — serving image built in CI",
            "install Docker to build/verify the Lambda image locally (optional)",
            "-",
        )
    code, out = _run(["docker", "--version"])
    return Check(
        "Docker", PASS if code == 0 else ABSENT, out, "start Docker daemon to build image", "-"
    )


CHECKS = [
    check_python,
    check_uv,
    check_ruff,
    check_pytest,
    check_git,
    check_kaggle,
    check_aws,
    check_terraform,
    check_r,
    check_spark,
    check_web_sr117,
    check_github,
    check_docker,
]

# Items that must be green to proceed out of Phase 0 into Phase A.
PHASE0_ESSENTIALS = {"Python >= 3.11", "uv", "ruff", "pytest", "git"}


def run_checks() -> list[Check]:
    return [fn() for fn in CHECKS]


def render_markdown(checks: list[Check]) -> str:
    lines = [
        "# Environment preflight",
        "",
        "Generated by `uv run python -m tooling.preflight` (PROJECT_SPEC.md Section 0.5).",
        "Regenerate after changing the toolchain; do not hand-edit.",
        "",
        "| Item | Status | Detail | Blocks | What to do if absent |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in checks:
        icon = _STATUS_ICON.get(c.status, c.status)
        remediation = "—" if c.remediation == "-" else c.remediation
        blocks = "—" if c.blocks == "-" else f"Phase {c.blocks}"
        lines.append(f"| {c.item} | {icon} {c.status} | {c.detail} | {blocks} | {remediation} |")
    essentials_ok = all(c.ok for c in checks if c.item in PHASE0_ESSENTIALS)
    blockers = [c for c in checks if c.status != PASS and c.blocks != "-"]
    lines += [
        "",
        "## Summary",
        "",
        f"- Phase 0 essentials (Python/uv/ruff/pytest/git): "
        f"**{'GREEN' if essentials_ok else 'NOT GREEN'}**",
    ]
    if blockers:
        lines.append(
            "- Outstanding phase blockers (handled at their phase, fallbacks in SPEC Section 10):"
        )
        for c in blockers:
            lines.append(f"  - Phase {c.blocks}: **{c.item}** — {c.detail}")
    else:
        lines.append("- No outstanding phase blockers.")
    lines.append("")
    return "\n".join(lines)


def print_console(checks: list[Check]) -> None:
    width = max(len(c.item) for c in checks)
    print("\nEnvironment preflight (PROJECT_SPEC.md Section 0.5)")
    print("=" * (width + 40))
    for c in checks:
        icon = _STATUS_ICON.get(c.status, "?")
        blocks = "" if c.blocks == "-" else f"  [blocks Phase {c.blocks}]"
        print(f"  {icon} {c.item.ljust(width)}  {c.status:<7} {c.detail}{blocks}")
    print("=" * (width + 40))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0 environment preflight")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if ANY blocker (any phase) is unmet, not just Phase 0 essentials",
    )
    args = parser.parse_args(argv)

    checks = run_checks()
    print_console(checks)

    PREFLIGHT_DOC.parent.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_DOC.write_text(render_markdown(checks), encoding="utf-8")
    print(f"\nWrote {PREFLIGHT_DOC.relative_to(REPO_ROOT)}")

    essentials_ok = all(c.ok for c in checks if c.item in PHASE0_ESSENTIALS)
    any_blocker = any(c.status != PASS and c.blocks != "-" for c in checks)

    if not essentials_ok:
        print("\nPhase 0 essentials NOT green — resolve before Phase A.", file=sys.stderr)
        return 1
    if args.strict and any_blocker:
        print("\n--strict: outstanding phase blockers present.", file=sys.stderr)
        return 2
    print("\nPhase 0 essentials green. Later-phase blockers (if any) are handled at their phase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
