"""TC-8-1 — drift monitoring workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import (
    DriftMonitoringTaskConfig,
    DriftMonitoringWorkflowConfig,
    run_tasks,
)
from dataeval_flow.workflows.drift.params import DriftDetectorKNeighbors

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow import PipelineConfig


@pytest.mark.test_case("8-1")
class TestDriftMonitoringWorkflow:
    def test_drift_workflow_runs(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            sources=(("ref", 0), ("test", 99)),
            workflows=[
                DriftMonitoringWorkflowConfig(
                    name="drift_main",
                    type="drift-monitoring",
                    detectors=[DriftDetectorKNeighbors(method="kneighbors", k=3)],
                ),
            ],
            tasks=[
                DriftMonitoringTaskConfig(
                    name="drift_task",
                    workflow="drift_main",
                    sources=["ref", "test"],
                    extractor="flat",
                ),
            ],
        )
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.success
        text = result.report()
        assert isinstance(text, str)
        assert text.strip()
        # Typed output check: exposes drift findings on the data payload
        assert len(result.data.raw.detectors) > 0
        assert len(result.data.report.findings) > 0
