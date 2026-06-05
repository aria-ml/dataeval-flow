#!/usr/bin/env python3
"""Generate meta repo artifacts from verification test results."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

VERIFICATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VERIFICATION_DIR.parent
REGISTRY_PATH = VERIFICATION_DIR / "registry.yaml"
REPORT_PATH = PROJECT_ROOT / "output" / "verification_report.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "metarepo"


def load_registry() -> dict:
    """Load the verification registry YAML."""
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def load_report() -> dict | None:
    """Load the verification JSON report if present, else return None."""
    if REPORT_PATH.exists():
        with open(REPORT_PATH) as f:
            return json.load(f)
    return None


def _human_readable_step(nodeid: str) -> str:
    """Convert a pytest nodeid's test name into a human-readable step description."""
    test_name = nodeid.rsplit("::", 1)[-1]
    # Strip pytest parametrize suffix, e.g. "test_foo[bar-baz]" -> "test_foo"
    if "[" in test_name:
        test_name = test_name.split("[", 1)[0]
    return re.sub(r"^test_", "", test_name).replace("_", " ").capitalize()


def _result_char(status: str) -> str:
    if status == "passed":
        return "P"
    if status == "skipped":
        return "S"
    return "F"


def generate_test_case_md(tc_id: str, tc_meta: dict, report: dict | None) -> str:
    """Render a single test-case markdown document from registry + optional report."""
    tc_key = f"test-case-{tc_id}"
    today = datetime.now(tz=timezone.utc).strftime("%m/%d/%Y")
    tc_report = None
    if report and tc_key in report.get("test_cases", {}):
        tc_report = report["test_cases"][tc_key]

    lines: list[str] = []
    lines.append(f"# {tc_meta['name']}")
    lines.append("")
    lines.append("## Description")
    lines.append("")
    lines.append(f"- Test Type: {tc_meta['test_type']}")
    lines.append(f"- Business Case: {tc_meta['business_case'].strip()}")
    lines.append("")
    lines.append("**Initial Conditions:**")
    lines.append("")
    for i, cond in enumerate(tc_meta["initial_conditions"], 1):
        lines.append(f"{i}. {cond}")
    lines.append("")
    lines.append("## Test Steps")
    lines.append("")

    if tc_report:
        tests = tc_report["tests"]
        for i, test in enumerate(tests, 1):
            lines.append(f"{i}. {_human_readable_step(test['test'])}")
        lines.append(f"{len(tests) + 1}. Confirm the Expected Results by validating all steps pass")
    else:
        for i, er in enumerate(tc_meta["expected_results"], 1):
            lines.append(f"{i}. Verify: {er}")
        lines.append(
            f"{len(tc_meta['expected_results']) + 1}. Confirm the Expected Results by validating all steps pass",
        )
    lines.append("")
    lines.append("**Expected Results**")
    lines.append("")
    for i, result in enumerate(tc_meta["expected_results"], 1):
        lines.append(f"{i}. {result}")
    lines.append("")
    lines.append("## Test Results")
    lines.append("")
    lines.append("| Test Step |  Result | Notes |")
    lines.append("|:----------|:-------:|:------|")

    if tc_report:
        tests = tc_report["tests"]
        for i, test in enumerate(tests, 1):
            r = _result_char(test["status"])
            lines.append(f"|{i:<10}|    {r}    |  [^{i}] |")
        overall = tc_report["status"]
        confirm = "P" if overall == "passed" else "F"
        n = len(tests) + 1
        lines.append(f"|{n:<10}|    {confirm}    |  [^{n}] |")
        lines.append("")
        for i, test in enumerate(tests, 1):
            lines.append(f"[^{i}]: `{test['test']}` — {test['status']}")
        lines.append(f"[^{n}]: Overall verification — {overall}")
    else:
        lines.append("|1         |   P/F   |  [^1] |")
        lines.append("")
        lines.append("[^1]: Awaiting automated test results")

    lines.append("")
    lines.append(f"**Last Updated Date:** {today}")
    return "\n".join(lines) + "\n"


def _tc_sort_key(tc_id: str) -> list[int]:
    return [int(p) for p in tc_id.split("-")]


def generate_vcrm(registry: dict, report: dict | None) -> str:
    """Render the VCRM markdown table from registry + optional report."""
    requirements = registry["requirements"]
    test_cases = registry["test_cases"]
    today = datetime.now(tz=timezone.utc).strftime("%m/%d/%Y")
    all_tc_ids = sorted(test_cases.keys(), key=_tc_sort_key)

    tc_headers = [f"[TC-{tc_id.replace('-', '.')}][{tc_id}]" for tc_id in all_tc_ids]
    header = "| Requirement ID | Requirement Origin | Coverage | " + " | ".join(tc_headers) + " |"

    sep_parts = [":--------------", ":-------------------", ":--------:"] + [":-------------:"] * len(all_tc_ids)
    separator = "| " + " | ".join(sep_parts) + " |"

    rows: list[str] = []
    for req_id, req_data in requirements.items():
        req_tcs = set(req_data.get("test_cases", []))
        coverage = "Yes" if req_tcs else "No"
        ref_key = req_id.lower().replace("-", "")
        req_cell = f"[{req_id}][{ref_key}]"
        origin = req_data.get("origin", "")
        origin_ref = origin.lower().replace("-", "").replace(".", "")
        origin_cell = f"[{origin}][{origin_ref}]" if origin else ""
        tc_cells = ["X" if tc_id in req_tcs else " " for tc_id in all_tc_ids]
        row_parts = [req_cell, origin_cell, coverage] + tc_cells
        rows.append("| " + " | ".join(row_parts) + " |")

    verification_cells: list[str] = []
    for tc_id in all_tc_ids:
        tc_key = f"test-case-{tc_id}"
        if report and tc_key in report.get("test_cases", {}):
            status = report["test_cases"][tc_key]["status"]
            verification_cells.append("Pass" if status == "passed" else "Fail")
        else:
            verification_cells.append("Pending")
    verification_row = "| **Verification** | | | " + " | ".join(verification_cells) + " |"

    tc_links = [f"[{tc_id}]:test-cases/test-case-{tc_id}.md" for tc_id in all_tc_ids]
    req_links = []
    for req_id, req_data in requirements.items():
        ref_key = req_id.lower().replace("-", "")
        filename = req_data.get("file", "#")
        req_links.append(f"[{ref_key}]:requirements/{filename}")
    origin_links: dict[str, str] = {}
    for req_data in requirements.values():
        origin = req_data.get("origin", "")
        origin_link = req_data.get("origin_link", "#")
        if origin:
            origin_ref = origin.lower().replace("-", "").replace(".", "")
            origin_links[origin_ref] = f"[{origin_ref}]:{origin_link}"

    parts = [
        "# DataEval Flow Verification Cross-Reference Matrix (VCRM)",
        "",
        header,
        separator,
        *rows,
        verification_row,
        "",
        f"**Last Updated:** {today}",
        "",
        "<!-- Links for Test Cases -->",
        "",
        *tc_links,
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


def main() -> None:
    """Generate test-case markdown stubs and VCRM under output/metarepo/."""
    registry = load_registry()
    report = load_report()

    if report:
        s = report["summary"]
        print(
            f"Loaded verification report: {s['total_test_cases']} test cases "
            f"({s['passed']} passed, {s['failed']} failed, {s['skipped']} skipped)",
        )
    else:
        print("No verification report found — generating templates only")

    tc_dir = OUTPUT_DIR / "test-cases"
    tc_dir.mkdir(parents=True, exist_ok=True)

    for tc_id, tc_meta in registry["test_cases"].items():
        content = generate_test_case_md(tc_id, tc_meta, report)
        out_path = tc_dir / f"test-case-{tc_id}.md"
        out_path.write_text(content)
        print(f"  Generated {out_path.name}")

    vcrm_content = generate_vcrm(registry, report)
    vcrm_path = OUTPUT_DIR / "vcrm.md"
    vcrm_path.write_text(vcrm_content)
    print("  Generated vcrm.md")
    print(f"\nAll artifacts written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
