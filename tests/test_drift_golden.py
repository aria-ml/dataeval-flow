"""drift-monitoring's recorded verdicts, chunks, class rows and finding severities (spec §10.11).

For now the legacy workflow replays the golden, which pins it as stable across runs; the preset is compared with it
once the legacy workflow is gone.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config import TaskConfig
from dataeval_flow.workflows.drift_monitoring import DriftMonitoringConfig
from dataeval_flow.workflows.drift_monitoring._workflow import _detector_display_name, _unique_method_keys
from tests.golden.drift import CASES, pipeline, produced_legacy
from tests.golden.rerouting import approximately

_GOLDEN = json.loads((Path(__file__).parent / "golden" / "drift.json").read_text())
_FLAGS = ("classwise_detectors", "chunked_detectors")


def _legacy(name: str) -> dict[str, Any]:
    drift = DriftMonitoringConfig.model_validate({"name": "drift", "type": "drift-monitoring", **CASES[name].legacy})
    task = TaskConfig(name="t", workflow="drift", sources=list(CASES[name].datasets()), extractor="flat")
    result = run_tasks(pipeline(name).model_copy(update={"workflows": [drift], "tasks": [task]}))["t"]
    assert result.success, result.errors
    produced = produced_legacy(result)
    detectors = list(drift.detectors)
    keys = _unique_method_keys(detectors)
    display = {_detector_display_name(det): key for det, key in zip(detectors, keys, strict=True)}
    produced["classwise"] = {display[shown]: rows for shown, rows in produced["classwise"].items()}
    return produced


def test_every_case_is_recorded() -> None:
    assert sorted(_GOLDEN) == sorted(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_legacy_reproduces_the_golden(name: str) -> None:
    expected = {key: value for key, value in _GOLDEN[name].items() if key not in _FLAGS}
    assert _legacy(name) == approximately(expected)
