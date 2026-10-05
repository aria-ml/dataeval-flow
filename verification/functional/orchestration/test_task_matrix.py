"""TC-13-1 — task matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import MatrixResult, run_tasks
from dataeval_flow.config import TaskConfig
from dataeval_flow.workflows.data_cleaning import DataCleaningConfig

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow.config import PipelineConfig


@pytest.mark.test_case("13-1")
class TestTaskMatrix:
    def test_a_matrix_task_runs_each_combination(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            workflows=[
                DataCleaningConfig(
                    name="clean_main", outliers={"flags": ["dimension", "pixel"], "outlier_threshold": "zscore"}
                )
            ],
            tasks=[
                TaskConfig.model_validate(
                    {
                        "name": "matrix_task",
                        "workflow": "clean_main",
                        "sources": "main",
                        "extractor": "flat",
                        "matrix": {"outliers.outlier_threshold": ["zscore", "modzscore"]},
                    }
                ),
            ],
        )
        result = run_tasks(cfg, data_dir=data_dir)["matrix_task"]
        assert isinstance(result, MatrixResult)
        assert result.success
        assert [run.label for run in result.runs] == [
            "outliers.outlier_threshold=zscore",
            "outliers.outlier_threshold=modzscore",
        ]
        assert all(run.result.success for run in result.runs)
        assert result.report().strip()
