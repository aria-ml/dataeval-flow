"""TC-7-1 — data cleaning workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import (
    DataCleaningWorkflowConfig,
    TaskConfig,
    run_tasks,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow import PipelineConfig


@pytest.mark.test_case("7-1")
class TestDataCleaningWorkflow:
    def test_cleaning_workflow_runs(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            extractor_batch_size=None,
            workflows=[
                DataCleaningWorkflowConfig(
                    name="clean_main",
                    type="data-cleaning",
                    outlier_method="zscore",
                    outlier_flags=["dimension", "pixel"],
                ),
            ],
            tasks=[
                TaskConfig(
                    name="clean_task",
                    workflow="clean_main",
                    sources="main",
                    extractor="flat",
                ),
            ],
        )
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.success
        text = result.report()
        assert isinstance(text, str)
        assert text.strip()
        # Typed output check: exposes outlier and duplicate findings on the data payload
        assert len(result.data.report.findings) > 0
