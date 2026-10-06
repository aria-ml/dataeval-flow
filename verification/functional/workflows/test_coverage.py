"""TC-20-1 — data coverage workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config import TaskConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.data_coverage import DataCoverageConfig

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow.config import PipelineConfig


@pytest.mark.test_case("20-1")
class TestDataCoverageWorkflow:
    def test_coverage_workflow_runs(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            workflows=[DataCoverageConfig(name="coverage_main")],
            tasks=[TaskConfig(name="coverage_task", workflow="coverage_main", sources="main", extractor="flat")],
        )
        result = run_tasks(cfg, data_dir=data_dir)["coverage_task"]
        assert result.success
        assert isinstance(result, ChainResult)
        assert result.report().strip()
        titles = [finding.title for finding in result.findings]
        assert {"Class Coverage", "Dimensional Completeness", "Class Shortfall"} <= set(titles)
        assert result.steps["representation"].output is not None

    def test_coverage_runs_without_extractor(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        """Embedding analyses are not assessed, not fatal, when no extractor is configured."""
        cfg, data_dir = image_folder_pipeline_builder(
            include_extractor=False,
            workflows=[DataCoverageConfig(name="coverage_meta")],
            tasks=[TaskConfig(name="coverage_meta_task", workflow="coverage_meta", sources="main")],
        )
        result = run_tasks(cfg, data_dir=data_dir)["coverage_meta_task"]
        assert result.success
        assert isinstance(result, ChainResult)
        assert result.steps["coverage"].status == "skipped"
        by_title = {finding.title: finding for finding in result.findings}
        assert "Class Shortfall" in by_title
        assert by_title["Class Coverage"].brief == "not assessed"
        assert by_title["Dimensional Completeness"].brief == "not assessed"
