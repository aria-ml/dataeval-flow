"""TC-12-1 — prioritization workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config import TaskConfig
from dataeval_flow.workflows.data_prioritization import DataPrioritizationConfig

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow.config import PipelineConfig


@pytest.mark.test_case("12-1")
class TestDataPrioritizationWorkflow:
    def test_prioritization_workflow_runs(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            sources=(("ref", 0), ("pool", 11)),
            n_per_class=8,
            workflows=[
                DataPrioritizationConfig(
                    name="prio_main",
                    type="data-prioritization",
                    prioritization={"method": "knn", "k": 3, "order": "hard_first", "policy": "difficulty"},
                ),
            ],
            tasks=[
                TaskConfig(
                    name="prio_task",
                    workflow="prio_main",
                    sources=["ref", "pool"],
                    extractor="flat",
                ),
            ],
        )
        result = run_tasks(cfg, data_dir=data_dir)["prio_task"]
        assert result.success
        text = result.report()
        assert isinstance(text, str)
        assert text.strip()
        # The preset's `selected` step holds each pool in ranked order: all of it, with neither `n` nor `fraction` set
        selected = result.steps["selected"].elements
        assert selected is not None
        assert list(selected) == ["pool"]
        assert len(selected["pool"].output) > 0
