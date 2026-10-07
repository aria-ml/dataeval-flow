"""TC-20-1 — data coverage workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config import DataCoverageTaskConfig, DataCoverageWorkflowConfig
from dataeval_flow.workflows.coverage.params import DataCoverageHealthThresholds

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow import PipelineConfig


class TestDataCoverageWorkflow:
    def test_coverage_workflow_runs(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            workflows=[
                DataCoverageWorkflowConfig(
                    name="coverage_main",
                    type="data-coverage",
                ),
            ],
            tasks=[
                DataCoverageTaskConfig(
                    name="coverage_task",
                    workflow="coverage_main",
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
        raw = result.data.raw
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
                DataCoverageWorkflowConfig(
                    name="coverage_no_ext",
                    type="data-coverage",
                ),
            ],
            tasks=[
                DataCoverageTaskConfig(
                    name="coverage_no_ext_task",
                    workflow="coverage_no_ext",
                    sources="main",
                ),
            ],
        )
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.success
        raw = result.data.raw
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
                DataCoverageWorkflowConfig(
                    name="coverage_onto",
                    type="data-coverage",
                    ontology={"root": {"class_0": [], "class_1": [], "class_2": []}},
                ),
            ],
            tasks=[
                DataCoverageTaskConfig(
                    name="coverage_onto_task",
                    workflow="coverage_onto",
                    sources="main",
                    extractor="flat",
                ),
            ],
        )
        result = run_tasks(cfg, data_dir=data_dir)[0]
        assert result.success
        raw = result.data.raw
        # The ontology either produced an assessment or recorded why it could not.
        assert raw.ontology is not None or raw.ontology_skipped_reason is not None

    def test_unreadable_ontology_is_recorded_as_a_skip_reason(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        """An ontology file that cannot be parsed skips ontology analysis; the rest of the run is unaffected."""
        cfg, data_dir = image_folder_pipeline_builder(
            workflows=[
                DataCoverageWorkflowConfig(name="coverage_bad_onto", type="data-coverage", ontology="broken.ttl"),
            ],
            tasks=[
                DataCoverageTaskConfig(
                    name="coverage_bad_onto_task", workflow="coverage_bad_onto", sources="main", extractor="flat"
                ),
            ],
        )
        (data_dir / "broken.ttl").write_text("this is @@ not turtle <<<")

        result = run_tasks(cfg, data_dir=data_dir)[0]

        assert result.success
        raw = result.data.raw
        assert raw.ontology is None
        assert raw.ontology_skipped_reason
        assert raw.label_distribution is not None

    def test_threshold_breach_raises_a_warning_finding(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        """An ontology class with no examples breaches a leaf-coverage threshold the data fails."""

        def label_space_severity(thresholds: DataCoverageHealthThresholds) -> tuple[str, int]:
            cfg, data_dir = image_folder_pipeline_builder(
                n_classes=2,
                workflows=[
                    DataCoverageWorkflowConfig(
                        name="coverage_thresholds",
                        type="data-coverage",
                        ontology={"root": {"class_0": [], "class_1": [], "class_2": []}},
                        health_thresholds=thresholds,
                    ),
                ],
                tasks=[
                    DataCoverageTaskConfig(
                        name="coverage_thresholds_task",
                        workflow="coverage_thresholds",
                        sources="main",
                        extractor="flat",
                    ),
                ],
            )
            result = run_tasks(cfg, data_dir=data_dir)[0]
            assert result.success
            (finding,) = [f for f in result.data.report.findings if f.title == "Label Space Coverage"]
            return finding.severity, result.warning_count

        # Two of the three sanctioned classes have examples: leaf coverage is 2/3.
        severity, warning_count = label_space_severity(
            DataCoverageHealthThresholds(leaf_coverage=0.9, dark_branch_count=0)
        )
        assert severity == "warning"
        assert warning_count >= 1

        severity, _ = label_space_severity(DataCoverageHealthThresholds(leaf_coverage=0.5, dark_branch_count=5))
        assert severity != "warning"
