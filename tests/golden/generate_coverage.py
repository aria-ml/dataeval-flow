"""Record `coverage.json` from the legacy data-coverage workflow, before its port to a preset.

Run once, on the legacy workflow: `.venv/bin/python -m tests.golden.generate_coverage`. It records each case's findings
and what they were computed from: coverage, completeness, the label distribution, the metadata summary, the gaps and
the class worklist (coverage spec §8.2).
"""

import json
from pathlib import Path
from typing import Any

from dataeval_flow import run_tasks
from tests.golden.coverage import CASES, pipeline


def _record(name: str) -> dict[str, Any]:
    result = run_tasks(pipeline(name, legacy=True))["t"]
    assert result.success, result.errors
    raw = result.output.raw
    coverage, completeness, gaps, ontology = raw.coverage, raw.completeness, raw.metadata_gaps, raw.ontology
    return {
        "findings": [[f.severity, f.title, f.brief, f.description] for f in result.output.report.findings],
        "coverage": None
        if coverage is None
        else {
            "uncovered": [[item.index, item.target, item.class_name, item.radius] for item in coverage.uncovered],
            "per_class": [row.model_dump() for row in coverage.per_class],
            "radius": coverage.coverage_radius,
            "observations": coverage.observation_count,
            "dropped": coverage.dropped_detections,
        },
        "completeness": None
        if completeness is None
        else {"score": completeness.completeness_score, "pairs": len(completeness.nearest_neighbor_pairs)},
        "labels": {
            "counts": dict(raw.label_distribution.class_distribution),
            "empty_images": list(raw.label_distribution.empty_images),
            "missing": list(raw.label_distribution.missing_classes),
        },
        "summary": raw.metadata_distribution.metadata_summary,
        "gaps": None
        if gaps is None
        else {
            "mutual_information": gaps.mutual_info_class_to_factor,
            "gaps": [gap.model_dump() for gap in gaps.gaps],
        },
        "worklist": None if ontology is None else [row.model_dump() for row in ontology.representation.worklist],
    }


if __name__ == "__main__":
    golden = {name: _record(name) for name in CASES}
    path = Path(__file__).parent / "coverage.json"
    path.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")
