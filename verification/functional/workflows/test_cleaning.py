"""TC-7-1 — data cleaning workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import (
    DataCleaningWorkflowConfig,
    TaskConfig,
    run_tasks,
)
from verification.fixtures import plant_duplicate_and_outlier

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow import PipelineConfig


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

    def test_planted_duplicate_and_outlier_are_found(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            n_per_class=10,
            workflows=[
                DataCleaningWorkflowConfig(
                    name="clean_main",
                    type="data-cleaning",
                    outlier_method="zscore",
                    outlier_flags=["dimension", "pixel"],
                ),
            ],
            tasks=[TaskConfig(name="clean_task", workflow="clean_main", sources="main", extractor="flat")],
        )
        duplicate_pair, outlier_index = plant_duplicate_and_outlier(data_dir / "main")

        result = run_tasks(cfg, data_dir=data_dir)[0]

        assert result.success
        raw = result.data.raw
        assert raw.dataset_size == 22
        assert raw.duplicates["items"]["exact"] == [duplicate_pair]
        assert {issue["item_index"] for issue in raw.img_outliers["issues"]} == {outlier_index}
        titles = {f.title for f in result.data.report.findings}
        assert {"Duplicates", "Image Outliers"} <= titles
        assert result.report().strip()
