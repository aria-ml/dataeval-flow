"""TC-20-1 — data coverage workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config import TaskConfig
from dataeval_flow.workflows.data_coverage import DataCoverageConfig

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow import PipelineConfig


@pytest.mark.test_case("20-1")
class TestDataCoverageWorkflow:
    def test_coverage_workflow_runs(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            workflows=[
                DataCoverageConfig(
                    name="coverage_main",
                    type="data-coverage",
                ),
            ],
            tasks=[
                TaskConfig(
                    name="coverage_task",
                    workflow="coverage_main",
                    sources="main",
                    extractor="flat",
                ),
            ],
        )
        result = run_tasks(cfg, data_dir=data_dir)["coverage_task"]
        assert result.success
        text = result.report()
        assert isinstance(text, str)
        assert text.strip()
        raw = result.output.raw
        # Metadata- and label-based analyses always run.
        assert raw.label_distribution is not None
        assert raw.metadata_distribution is not None

    def test_coverage_runs_without_extractor(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        """Embedding analyses are skipped, not fatal, when no extractor is configured."""
        cfg, data_dir = image_folder_pipeline_builder(
            include_extractor=False,
            workflows=[
                DataCoverageConfig(
                    name="coverage_no_ext",
                    type="data-coverage",
                ),
            ],
            tasks=[
                TaskConfig(
                    name="coverage_no_ext_task",
                    workflow="coverage_no_ext",
                    sources="main",
                ),
            ],
        )
        result = run_tasks(cfg, data_dir=data_dir)["coverage_no_ext_task"]
        assert result.success
        raw = result.output.raw
        assert raw.coverage is None
        assert raw.completeness is None
        assert raw.label_distribution is not None

    def test_coverage_with_ontology(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        """A declared ontology drives label-space analysis instead of a synthesized one."""
        cfg, data_dir = image_folder_pipeline_builder(
            n_classes=2,
            workflows=[
                DataCoverageConfig(
                    name="coverage_onto",
                    type="data-coverage",
                    ontology={"root": {"class_0": [], "class_1": [], "class_2": []}},
                ),
            ],
            tasks=[
                TaskConfig(
                    name="coverage_onto_task",
                    workflow="coverage_onto",
                    sources="main",
                    extractor="flat",
                ),
            ],
        )
        result = run_tasks(cfg, data_dir=data_dir)["coverage_onto_task"]
        assert result.success
        raw = result.output.raw
        # The ontology either produced an assessment or recorded why it could not.
        assert raw.ontology is not None or raw.ontology_skipped_reason is not None
