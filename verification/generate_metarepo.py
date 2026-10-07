#!/usr/bin/env python3
"""Generate meta repo artifacts from the verification registry and test results.

Reads the registry (``registry.yaml``), the pytest report
(``output/verification_report.json``), and, when available, CI job results
(``output/ci_jobs.json``) to produce under ``output/metarepo/``:

  - requirements/<id>-<slug>.md   one per requirement, rendered from the registry
  - test-cases/test-case-<id>.md  one per test case, with each step's result
  - vcrm.md                       requirement to test case matrix with verification row

A test case is a list of steps. Each step names the pytest tests (``alias::test``
or a full node id), CI jobs (``CI: <job>``), or planned tests (``NEW: <what>``)
that give its evidence. A step passes when every automated piece passes. A step
with a planned test, or with CI evidence that is not available, is pending.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import yaml

VERIFICATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VERIFICATION_DIR.parent
REGISTRY_PATH = VERIFICATION_DIR / "registry.yaml"
REPORT_PATH = PROJECT_ROOT / "output" / "verification_report.json"
CI_JOBS_PATH = PROJECT_ROOT / "output" / "ci_jobs.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "metarepo"

DR_15 = "https://jatic.pages.jatic.net/internal-docs/standards/product/documentation/program-doc-requirements/#dr-15-product-requirements-definitions"

# Step and test case outcomes.
PASS, FAIL, SKIP, PENDING = "passed", "failed", "skipped", "pending"


def load_registry() -> dict:
    """Load the verification registry."""
    return yaml.safe_load(REGISTRY_PATH.read_text())


def load_json(path: Path) -> dict | None:
    """Load a JSON file if it exists."""
    return json.loads(path.read_text()) if path.exists() else None


def slug(text: str) -> str:
    """File-name slug for a requirement name."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def requirement_filename(req: dict) -> str:
    """Requirement file name: ``FR-<n>-<slug>.md`` or ``NFR-<n>-<slug>.md`` (DR-1.5-H-2)."""
    return f"{req['id']}-{slug(req['name'])}.md"


# ---------------------------------------------------------------------------
# Evidence resolution
# ---------------------------------------------------------------------------


def expand(ref: str, aliases: dict[str, str]) -> str:
    """Expand ``alias::test`` to a full pytest node id."""
    head, sep, rest = ref.partition("::")
    return f"{aliases[head]}::{rest}" if sep and head in aliases else ref


def evidence_status(ref: str, nodes: dict[str, str] | None, ci_jobs: dict[str, str] | None) -> str:
    """Outcome of one piece of evidence."""
    if ref.startswith("NEW:"):
        return PENDING
    if ref.startswith("CI:"):
        job = ref[3:].strip()
        if not ci_jobs or job not in ci_jobs:
            return PENDING
        return {"success": PASS, "failed": FAIL, "skipped": SKIP}.get(ci_jobs[job], PENDING)
    if nodes is None or ref not in nodes:
        return PENDING
    return {"passed": PASS, "skipped": SKIP}.get(nodes[ref], FAIL)


def combine(statuses: list[str]) -> str:
    """Overall outcome: any failure fails; any pending is pending; all skipped is skipped."""
    if FAIL in statuses:
        return FAIL
    if PENDING in statuses:
        return PENDING
    if statuses and all(s == SKIP for s in statuses):
        return SKIP
    return PASS


def step_statuses(
    tc: dict, aliases: dict[str, str], nodes: dict | None, ci_jobs: dict | None
) -> list[tuple[str, list[tuple[str, str]]]]:
    """For each step: (outcome, [(evidence, outcome), ...])."""
    out = []
    for step in tc["steps"]:
        ev = [(expand(r, aliases), evidence_status(expand(r, aliases), nodes, ci_jobs)) for r in step["tests"]]
        out.append((combine([s for _, s in ev]), ev))
    return out


def tc_outcome(tc: dict, aliases: dict, nodes: dict | None, ci_jobs: dict | None) -> str:
    """Outcome of a whole test case."""
    return combine([s for s, _ in step_statuses(tc, aliases, nodes, ci_jobs)])


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

_MARK = {PASS: "P", FAIL: "F", SKIP: "S", PENDING: "—"}
_LABEL = {PASS: "passed", FAIL: "failed", SKIP: "skipped", PENDING: "pending"}


def requirement_md(req: dict) -> str:
    """Render one requirement in the DR-1.5-R-1 shape."""
    kind = "Non-Functional" if req["id"].startswith("N") else "Functional"
    out = [
        f"# {req['id']}: {req['name']}",
        "",
        f"## {kind} Requirements",
        "",
        f"- Requirement ID: {req['id']}",
        f"  - Name: {req['name']}",
        f"  - Description: {req['description']}",
        "  - Acceptance Criteria",
    ]
    out += [f"    - {c}" for c in req["criteria"]]
    if req.get("notes"):
        out += [f"  - {req.get('notes_title', 'Reference Measurements (informative; not acceptance criteria)')}"]
        out += [f"    - {n}" for n in req["notes"]]
    return "\n".join(out) + "\n"


def test_case_md(tc: dict, aliases: dict, nodes: dict | None, ci_jobs: dict | None, today: str) -> str:
    """Render one test case in the DR-1.6-H-4 template."""
    steps = step_statuses(tc, aliases, nodes, ci_jobs)
    n = len(steps)
    overall = combine([s for s, _ in steps])
    lines = [
        f"# {tc['name']}",
        "",
        "## Description",
        "",
        f"- Test Type: {tc['type']}",
        f"- Business Case: {tc['business']}",
        "",
        "**Initial Conditions:**",
        "",
    ]
    lines += [f"{i}. {c}" for i, c in enumerate(tc["conditions"], 1)]
    lines += ["", "## Test Steps", ""] + [f"{i}. {s['do']}" for i, s in enumerate(tc["steps"], 1)]
    lines += [f"{n + 1}. Confirm the Expected Results by validating all steps pass.", "", "**Expected Results**", ""]
    lines += [f"{i}. {s['expect']}" for i, s in enumerate(tc["steps"], 1)]
    lines += ["", "## Test Results", "", "| Test Step |  Result | Notes |", "|:----------|:-------:|:------|"]
    for i, (status, _) in enumerate(steps, 1):
        lines.append(f"|{i:<10}|    {_MARK[status]}    |  [^{i}] |")
    lines += [f"|{n + 1:<10}|    {_MARK[overall]}    |  [^{n + 1}] |", ""]
    for i, (_, ev) in enumerate(steps, 1):
        parts = []
        for ref, status in ev:
            if ref.startswith("NEW:"):
                parts.append(f"{ref[4:].strip()}: not yet automated")
            elif ref.startswith("CI:"):
                parts.append(
                    f"CI job `{ref[3:].strip()}`: {'result not available' if status == PENDING else _LABEL[status]}"
                )
            else:
                parts.append(f"`{ref}`: {'not run' if status == PENDING else _LABEL[status]}")
        lines.append(f"[^{i}]: " + "; ".join(parts))
    lines += [f"[^{n + 1}]: Overall verification: {_LABEL[overall]}", "", f"**Last Updated Date:** {today}"]
    return "\n".join(lines) + "\n"


def tc_sort_key(tc_id: str) -> list[int]:
    """Sort test case ids like ``1-1`` and ``21-1`` numerically."""
    return [int(p) for p in tc_id.split("-")]


def vcrm_md(registry: dict, nodes: dict | None, ci_jobs: dict | None, today: str) -> str:
    """Render the VCRM."""
    aliases = registry.get("aliases", {})
    tcs = sorted(registry["test_cases"], key=lambda t: tc_sort_key(t["id"]))
    ids = [t["id"] for t in tcs]
    header = (
        "| Requirement ID | Requirement Origin | Coverage | "
        + " | ".join(f"[TC-{i.replace('-', '.')}][{i}]" for i in ids)
        + " |"
    )
    sep = (
        "| "
        + " | ".join([":--------------", ":-------------------", ":--------:"] + [":-------------:"] * len(ids))
        + " |"
    )
    rows, origin_links, req_links = [], {}, []
    for req in registry["requirements"]:
        mine = {t["id"] for t in tcs if t["req"] == req["id"]}
        ref = req["id"].lower().replace("-", "")
        origin = req.get("origin", "DR-1.5")
        oref = origin.lower().replace("-", "").replace(".", "")
        origin_links[oref] = f"[{oref}]:{req.get('origin_link', DR_15)}"
        req_links.append(f"[{ref}]:requirements/{requirement_filename(req)}")
        cells = ["X" if i in mine else " " for i in ids]
        rows.append(
            "| " + " | ".join([f"[{req['id']}][{ref}]", f"[{origin}][{oref}]", "Yes" if mine else "No", *cells]) + " |"
        )
    label = {PASS: "Pass", FAIL: "Fail", SKIP: "Skipped", PENDING: "Pending"}
    verification = [label[tc_outcome(t, aliases, nodes, ci_jobs)] for t in tcs]
    parts = [
        f"# {registry['product']} Verification Cross-Reference Matrix (VCRM)",
        "",
        header,
        sep,
        *rows,
        "| **Verification** | | | " + " | ".join(verification) + " |",
        "",
        f"**Last Updated:** {today}",
        "",
        "<!-- Links for Test Cases -->",
        "",
        *[f"[{i}]:test-cases/test-case-{i}.md" for i in ids],
        "",
        "<!-- Links for Requirement IDs -->",
        "",
        *req_links,
        "",
        "<!-- Links for Requirement Origins -->",
        "",
        *sorted(origin_links.values()),
    ]
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def check_registry(registry: dict) -> None:
    """Fail loudly on a registry that cannot render a consistent matrix."""
    req_ids = {r["id"] for r in registry["requirements"]}
    tc_ids = [t["id"] for t in registry["test_cases"]]
    if len(set(tc_ids)) != len(tc_ids):
        raise SystemExit("registry error: duplicate test case id")
    for t in registry["test_cases"]:
        if t["req"] not in req_ids:
            raise SystemExit(f"registry error: test case {t['id']} names unknown requirement {t['req']}")
        if not t["steps"]:
            raise SystemExit(f"registry error: test case {t['id']} has no steps")
    uncovered = req_ids - {t["req"] for t in registry["test_cases"]}
    if uncovered:
        raise SystemExit(f"registry error: requirements without a test case: {sorted(uncovered)}")


def main() -> None:
    """Write requirements, test cases, and the VCRM to ``output/metarepo``."""
    registry = load_registry()
    check_registry(registry)
    aliases = registry.get("aliases", {})
    report = load_json(REPORT_PATH)
    nodes = report.get("nodes") if report else None
    ci_jobs = load_json(CI_JOBS_PATH)
    today = datetime.now(tz=UTC).strftime("%m/%d/%Y")

    if nodes is None:
        print("No verification report with node results found: every pytest step will show as pending")
    for sub in ("requirements", "test-cases"):
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)
        for old in (OUTPUT_DIR / sub).glob("*.md"):
            old.unlink()
    for req in registry["requirements"]:
        (OUTPUT_DIR / "requirements" / requirement_filename(req)).write_text(requirement_md(req))
    for tc in registry["test_cases"]:
        out = OUTPUT_DIR / "test-cases" / f"test-case-{tc['id']}.md"
        out.write_text(test_case_md(tc, aliases, nodes, ci_jobs, today))
    (OUTPUT_DIR / "vcrm.md").write_text(vcrm_md(registry, nodes, ci_jobs, today))

    results = [tc_outcome(t, aliases, nodes, ci_jobs) for t in registry["test_cases"]]
    counts = {k: results.count(k) for k in (PASS, FAIL, SKIP, PENDING)}
    print(f"{len(registry['requirements'])} requirements, {len(results)} test cases: {counts}")
    print(f"Artifacts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
