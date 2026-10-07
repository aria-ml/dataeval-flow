"""TC-8-1 — the shift preset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators.shift import DriftKNeighborsConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.shift import ShiftConfig

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow.config import PipelineConfig


@pytest.mark.test_case("8-1")
class TestShiftWorkflow:
    def test_drift_workflow_runs(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            sources=(("ref", 0), ("test", 99)),
            workflows=[
                ShiftConfig(
                    name="drift_main",
                    type="shift",
                    detectors=[DriftKNeighborsConfig(k=3)],
                ),
            ],
            tasks=[
                TaskConfig(
                    name="drift_task",
                    workflow="drift_main",
                    sources=["ref", "test"],
                    extractor="flat",
                ),
            ],
        )
        result = run_tasks(cfg, data_dir=data_dir)["drift_task"]
        assert isinstance(result, ChainResult)
        assert result.success
        text = result.report()
        assert isinstance(text, str)
        assert text.strip()
        # The detector's `drift` check judges the test source against the reference
        (finding,) = (result.steps["drift-kneighbors-check"].elements or {})["test"].output
        assert finding.title == "Drift (K-Neighbors)"
