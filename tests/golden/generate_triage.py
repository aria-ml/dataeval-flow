"""Records `triage_findings.json` from the legacy metadata-triage workflow, before its port to a preset.

Run it once, on the legacy workflow: `.venv/bin/python -m tests.golden.generate_triage`. For each case it records the
findings as severity, title and brief, in order; the suggested policy stanza; and the binning record the result's
envelope carries. The preset must agree with all three (spec §10.10).
"""

import json
from pathlib import Path
from typing import Any

from dataeval_flow import run_tasks
from tests.golden.triage import CASES, pipeline


def record(name: str) -> dict[str, Any]:
    """Case `name`'s findings, suggested stanza and binning record, read off the legacy result."""
    result = run_tasks(pipeline(name))["t"]
    assert result.success, result.errors
    return {
        "findings": [[f.severity, f.title, f.brief] for f in result.output.report.findings],  # type: ignore[attr-defined]
        "suggested_policy_yaml": result.output.raw.suggested_policy_yaml,  # type: ignore[attr-defined]
        "metadata_binning": result.metadata.metadata_binning,
    }


if __name__ == "__main__":
    golden = {name: record(name) for name in sorted(CASES)}
    path = Path(__file__).parent / "triage_findings.json"
    path.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}")
