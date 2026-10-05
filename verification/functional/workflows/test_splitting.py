"""TC-11-1 — splitting workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config import TaskConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.data_splitting import DataSplittingConfig

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow.config import PipelineConfig


@pytest.mark.test_case("11-1")
class TestDataSplittingWorkflow:
    def test_splitting_workflow_runs(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            n_per_class=8,
            include_extractor=False,
            workflows=[
                DataSplittingConfig(name="split_main", test_frac=0.25, val_frac=0.25, stratify=False),
            ],
            tasks=[
                TaskConfig(
                    name="split_task",
                    workflow="split_main",
                    sources="main",
                ),
            ],
        )
        result = run_tasks(cfg, data_dir=data_dir)["split_task"]
        assert result.success
        text = result.report()
        assert isinstance(text, str)
        assert text.strip()
        # The split step records each part's indices in its details
        assert isinstance(result, ChainResult)
        indices = result.steps["split"].details["indices"]
        assert all(len(indices[part]) > 0 for part in ("train", "val", "test"))
