"""TC-11-1 — splitting workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config.schemas import DataSplittingTaskConfig, DataSplittingWorkflowConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow import PipelineConfig


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
                DataSplittingWorkflowConfig(
                    name="split_main",
                    type="data-splitting",
                    test_frac=0.25,
                    val_frac=0.25,
                    num_folds=1,
                    stratify=False,
                ),
            ],
            tasks=[
                DataSplittingTaskConfig(
                    name="split_task",
                    workflow="split_main",
                    sources="main",
                ),
            ],
        )
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.success
        text = result.report()
        assert isinstance(text, str)
        assert text.strip()
        # Typed output check: splits (folds + test set) exposed on the data payload
        raw = result.data.raw
        assert len(raw.folds) > 0
        fold = raw.folds[0]
        assert len(fold.train_indices) > 0
        assert len(fold.val_indices) > 0
        assert len(raw.test_indices) > 0
