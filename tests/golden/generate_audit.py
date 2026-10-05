"""Record `audit.json` from the data-analysis workflow, before audit replaces it.

Run once, before the replacement: `.venv/bin/python -m tests.golden.generate_audit`.
"""

import json
from pathlib import Path
from typing import Any

from dataeval_flow import run_tasks
from tests.golden.audit import CASES, pipeline


def _record(name: str) -> dict[str, Any]:
    result = run_tasks(pipeline(name, legacy=True))["t"]
    assert result.success, result.errors
    raw = result.output.raw
    record: dict[str, Any] = {
        "splits": {
            split: {
                "outlier_count": s.image_quality.outlier_count,
                "outlier_rate": s.image_quality.outlier_rate,
                "exact_duplicates_count": s.redundancy.exact_duplicates_count,
                "near_duplicates_count": s.redundancy.near_duplicates_count,
                "class_distribution": s.label_health.class_distribution,
                "empty_images": len(s.label_health.empty_images),
            }
            for split, s in raw.splits.items()
        },
        "cross": {
            pair: {
                "exact_groups": c.redundancy.duplicate_leakage["exact_groups"],
                "near_groups": c.redundancy.duplicate_leakage["near_groups"],
                "divergence": c.distribution_shift.divergence,
            }
            for pair, c in raw.cross_split.items()
        },
    }
    summary = raw.splits["train"].bias.metadata_summary
    if summary:
        record["train_factors"] = json.loads(json.dumps(summary))
    return record


if __name__ == "__main__":
    golden = {name: _record(name) for name in CASES}
    path = Path(__file__).parent / "audit.json"
    path.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")
