"""TC-12-1 — prioritization workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config.schemas import (
    DataPrioritizationTaskConfig,
    DataPrioritizationWorkflowConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow import PipelineConfig


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
                DataPrioritizationWorkflowConfig(
                    name="prio_main",
                    type="data-prioritization",
                    method="knn",
                    k=3,
                    order="hard_first",
                    policy="difficulty",
                ),
            ],
            tasks=[
                DataPrioritizationTaskConfig(
                    name="prio_task",
                    workflow="prio_main",
                    sources=["ref", "pool"],
                    extractor="flat",
                ),
            ],
        )
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.success
        text = result.report()
        assert isinstance(text, str)
        assert text.strip()
        # Typed output check: ranked indices exposed for the non-reference source
        # (ref is the reference; pool is the source ranked against it).
        prioritizations = result.data.raw.prioritizations
        assert len(prioritizations) == 1
        (pool_result,) = prioritizations
        assert pool_result["source_name"] == "pool"
        assert len(pool_result["prioritized_indices"]) == pool_result["cleaned_size"]
