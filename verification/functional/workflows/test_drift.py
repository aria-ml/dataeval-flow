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
from verification.fixtures import write_image_folder

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow import PipelineConfig


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

    def test_same_distribution_target_is_not_flagged_as_drift(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        """A target drawn from the reference distribution (different samples) shows no drift."""
        cfg, data_dir = image_folder_pipeline_builder(
            sources=(("ref", 0), ("test", 99)),
            n_per_class=16,
            workflows=[
                DriftMonitoringWorkflowConfig(
                    name="drift_same",
                    type="drift-monitoring",
                    detectors=[DriftDetectorKNeighbors(method="kneighbors", k=3)],
                ),
            ],
            tasks=[
                DriftMonitoringTaskConfig(
                    name="drift_same_task", workflow="drift_same", sources=["ref", "test"], extractor="flat"
                ),
            ],
        )
        # Reference and target are independent draws from the same dark-pixel distribution.
        write_image_folder(data_dir / "ref", n_per_class=16, seed=0, high=100)
        write_image_folder(data_dir / "test", n_per_class=16, seed=5, high=100)

        result = run_tasks(cfg, data_dir=data_dir)[0]

        assert result.success
        (detector,) = result.data.raw.detectors.values()
        assert detector["drifted"] is False
        assert result.warning_count == 0
