"""TC-9-1 — OOD detection preset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config import TaskConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.shift import ShiftConfig

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow.config import PipelineConfig


@pytest.mark.test_case("9-1")
class TestOODWorkflow:
    def test_ood_workflow_runs(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        preset = {
            "name": "ood_main",
            "detectors": [{"type": "ood-kneighbors", "k": 3}],
            "factor-predictors": False,
            "factor-deviation": False,
        }
        cfg, data_dir = image_folder_pipeline_builder(
            sources=(("ref", 0), ("test", 99)),
            workflows=[ShiftConfig.model_validate(preset)],
            tasks=[TaskConfig(name="ood_task", workflow="ood_main", sources=["ref", "test"], extractor="flat")],
        )
        result = run_tasks(cfg, data_dir=data_dir)["ood_task"]
        assert isinstance(result, ChainResult)
        assert result.success
        assert result.report().strip()
        # Per-image scores and OOD flags for the test source: n_per_class=4 * n_classes=2 = 8 images.
        (output,) = [element.output for element in (result.steps["ood-kneighbors"].elements or {}).values()]
        assert len(output.is_ood) == 8
        assert len(output.instance_score) == 8
