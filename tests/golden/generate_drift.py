"""Records `drift.json` from the legacy drift-monitoring workflow, before its port to a preset.

Run it once, on the legacy workflow: `.venv/bin/python -m tests.golden.generate_drift`. For each case it records the
finding severities, each detector's verdict, threshold and chunks, each class row stored under its detector's key, and
which detectors were classwise or chunked. The preset must agree with all of it (spec §10.11).
"""

import json
from pathlib import Path
from typing import Any

from dataeval_flow import run_tasks
from dataeval_flow.config import TaskConfig
from dataeval_flow.workflows.drift_monitoring import DriftMonitoringConfig
from dataeval_flow.workflows.drift_monitoring._workflow import _detector_display_name, _unique_method_keys
from tests.golden.drift import CASES, pipeline, produced_legacy


def record(name: str) -> dict[str, Any]:
    """Case `name`, run through the legacy workflow."""
    drift = DriftMonitoringConfig.model_validate({"name": "drift", "type": "drift-monitoring", **CASES[name].legacy})
    task = TaskConfig(name="t", workflow="drift", sources=list(CASES[name].datasets()), extractor="flat")
    config = pipeline(name).model_copy(update={"workflows": [drift], "tasks": [task]})
    result = run_tasks(config)["t"]
    assert result.success, result.errors
    produced = produced_legacy(result)
    detectors = list(drift.detectors)
    keys = _unique_method_keys(detectors)
    display = {_detector_display_name(det): key for det, key in zip(detectors, keys, strict=True)}
    produced["classwise"] = {display[shown]: rows for shown, rows in produced["classwise"].items()}
    produced["classwise_detectors"] = [key for det, key in zip(detectors, keys, strict=True) if det.classwise]
    produced["chunked_detectors"] = [key for det, key in zip(detectors, keys, strict=True) if det.chunking]
    return produced


if __name__ == "__main__":
    golden = {name: record(name) for name in sorted(CASES)}
    path = Path(__file__).parent / "drift.json"
    path.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}")
