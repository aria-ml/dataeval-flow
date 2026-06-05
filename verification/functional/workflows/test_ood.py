"""TC-9-1 — OOD detection workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config import OODDetectionTaskConfig, OODDetectionWorkflowConfig
from dataeval_flow.workflows.ood.params import OODDetectorKNeighbors

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow import PipelineConfig


@pytest.mark.test_case("9-1")
class TestOODWorkflow:
    def test_ood_workflow_runs(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            sources=(("ref", 0), ("test", 99)),
            workflows=[
                OODDetectionWorkflowConfig(
                    name="ood_main",
                    type="ood-detection",
                    detectors=[OODDetectorKNeighbors(method="kneighbors", k=3)],
                    metadata_insights=False,
                ),
            ],
            tasks=[
                OODDetectionTaskConfig(
                    name="ood_task",
                    workflow="ood_main",
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
        # Typed output check: exposes OOD per-sample scores for the test dataset
        detectors = result.data.raw.detectors
        assert len(detectors) > 0
        (detector_result,) = detectors.values()
        # ``samples`` carries the per-sample (instance_score, is_ood) array; length
        # must equal the test dataset size (n_per_class=4 * n_classes=2 = 8).
        assert detector_result["total_count"] == 8
        assert len(detector_result["samples"]) == 8
