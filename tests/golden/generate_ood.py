"""Records `ood.json` from the legacy ood-detection workflow, before its port to a preset.

Run it once, on the legacy workflow: `.venv/bin/python -m tests.golden.generate_ood`. For each case it records:
- the findings, each one's severity, title and description;
- each detector's flags, scores and threshold;
- the union, mutual and unique sets;
- the normalized scores;
- the factor predictors and the factor deviations.

The preset must agree (ood-detection spec §10).
"""

import json
from pathlib import Path
from typing import Any

from dataeval_flow import run_tasks, set_device
from dataeval_flow.config import TaskConfig
from dataeval_flow.workflows.ood_detection import OODDetectionConfig, OODDetectionResult
from dataeval_flow.workflows.ood_detection._report import _compute_normalized_scores
from tests.golden.ood import CASES, pipeline


def record(name: str) -> dict[str, Any]:
    """Case `name`, run through the legacy workflow."""
    case = CASES[name]
    ood = OODDetectionConfig.model_validate({"name": "ood", "type": "ood-detection", **case.legacy})
    task = TaskConfig(name="t", workflow="ood", sources=list(case.datasets()), extractor="flat")
    config = pipeline(name).model_copy(update={"workflows": [ood], "tasks": [task]})
    result = run_tasks(config)["t"]
    assert isinstance(result, OODDetectionResult), result
    assert result.success, result.errors
    raw = result.output.raw
    normalized, mutual, unique = _compute_normalized_scores(raw.detectors)
    return {
        "findings": [
            {"severity": finding.severity, "title": finding.title, "description": finding.description}
            for finding in result.output.report.findings
        ],
        "detectors": {
            key: {
                "is_ood": [sample["is_ood"] for sample in detector.get("samples", [])],
                "scores": [sample["score"] for sample in detector.get("samples", [])],
                "threshold": detector["threshold_score"],
            }
            for key, detector in raw.detectors.items()
        },
        "test_size": raw.test_size,
        "union": raw.ood_indices,
        "mutual": sorted(mutual),
        "unique": {key: sorted(indices) for key, indices in unique.items()},
        "normalized": [normalized.get(index) for index in range(raw.test_size)],
        "predictors": raw.factor_predictors,
        "deviations": {str(entry["index"]): entry["deviations"] for entry in raw.factor_deviations or []},
    }


if __name__ == "__main__":
    set_device("cpu")  # as the tests compute, so the golden holds on a CPU-only runner
    golden = {name: record(name) for name in sorted(CASES)}
    path = Path(__file__).parent / "ood.json"
    path.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}")
