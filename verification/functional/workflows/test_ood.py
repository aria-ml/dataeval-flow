"""TC-9-1 — OOD detection workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config import OODDetectionTaskConfig, OODDetectionWorkflowConfig
from dataeval_flow.workflows.ood.params import OODDetectorKNeighbors
from verification.fixtures import write_image_folder

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow import PipelineConfig


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

    def test_clearly_different_target_is_flagged_out_of_distribution(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            sources=(("ref", 0), ("test", 99)),
            n_per_class=16,
            workflows=[
                OODDetectionWorkflowConfig(
                    name="ood_different",
                    type="ood-detection",
                    # Cosine distance cannot tell dark noise from bright noise (both point the same way).
                    detectors=[OODDetectorKNeighbors(method="kneighbors", k=3, distance_metric="euclidean")],
                    metadata_insights=False,
                ),
            ],
            tasks=[
                OODDetectionTaskConfig(
                    name="ood_different_task", workflow="ood_different", sources=["ref", "test"], extractor="flat"
                ),
            ],
        )
        # Reference is dark noise; the target is bright noise.
        write_image_folder(data_dir / "ref", n_per_class=16, seed=0, high=100)
        write_image_folder(data_dir / "test", n_per_class=16, seed=5, low=156)

        result = run_tasks(cfg, data_dir=data_dir)[0]

        assert result.success
        (detector,) = result.data.raw.detectors.values()
        assert detector["total_count"] == 32
        assert all(sample["is_ood"] for sample in detector["samples"])
        assert detector["ood_count"] == 32
