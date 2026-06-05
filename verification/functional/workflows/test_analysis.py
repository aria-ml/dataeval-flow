"""TC-10-1 — data analysis workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config import DataAnalysisTaskConfig, DataAnalysisWorkflowConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow import PipelineConfig


@pytest.mark.test_case("10-1")
class TestDataAnalysisWorkflow:
    def test_analysis_workflow_runs(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            workflows=[
                DataAnalysisWorkflowConfig(
                    name="analyze_main",
                    type="data-analysis",
                    outlier_method="zscore",
                    outlier_flags=["dimension", "pixel"],
                ),
            ],
            tasks=[
                DataAnalysisTaskConfig(
                    name="analyze_task",
                    workflow="analyze_main",
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
        # Typed output check: exposes analysis sub-outputs (image_quality, redundancy,
        # label_health, bias) per split on the data payload
        splits = result.data.raw.splits
        assert len(splits) > 0
        (split_result,) = splits.values()
        assert split_result.bias is not None
        assert split_result.image_quality is not None
        assert split_result.label_health is not None
        assert split_result.redundancy is not None
