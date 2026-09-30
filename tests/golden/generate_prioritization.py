"""Records `prioritization_rankings.json` from the legacy data-prioritization workflow, before its port to a preset.

Run it once, on the legacy workflow: `.venv/bin/python -m tests.golden.generate_prioritization`. For each case it
records each pool's ranking, as indices into the pool, and how many items cleaning removed from each source. The
preset must agree with them (spec §10.9).
"""

import json
from pathlib import Path
from typing import Any

from dataeval_flow import run_tasks
from tests.golden.prioritization import CASES, pipeline


def record(name: str) -> dict[str, dict[str, Any]]:
    """Case `name`'s rankings and removals, read off the legacy result."""
    result = run_tasks(pipeline(name))["t"]
    assert result.success, result.errors
    raw = result.output.raw  # type: ignore[attr-defined]
    rankings = {p["source_name"]: [int(i) for i in p["prioritized_indices"]] for p in raw.prioritizations}
    pools = {p["source_name"]: p["original_size"] - p["cleaned_size"] for p in raw.prioritizations}
    total = raw.cleaning_summary["total_removed"] if raw.cleaning_summary is not None else 0
    return {"rankings": rankings, "removed": {"ref": total - sum(pools.values()), **pools}}


if __name__ == "__main__":
    golden = {name: record(name) for name in sorted(CASES)}
    path = Path(__file__).parent / "prioritization_rankings.json"
    path.write_text(json.dumps(golden, indent=2) + "\n")
    print(f"wrote {path}")
