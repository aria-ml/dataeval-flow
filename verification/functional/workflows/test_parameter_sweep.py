"""TC-13-1 — parameter sweep workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import DataCleaningWorkflowConfig, TaskConfig, run_tasks
from dataeval_flow.config.schemas import (
    ParameterSweepTaskConfig,
    ParameterSweepWorkflowConfig,
)

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow import PipelineConfig


class TestParameterSweepWorkflow:
    def test_parameter_sweep_runs(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            workflows=[
                ParameterSweepWorkflowConfig(
                    name="sweep_main",
                    type="parameter-sweep",
                    outlier_flags=["dimension", "pixel"],
                    outlier_method=["zscore", "modzscore"],
                ),
            ],
            tasks=[
                ParameterSweepTaskConfig(
                    name="sweep_task",
                    workflow="sweep_main",
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
        # Typed output check: one result entry per swept parameter combination
        # (two outlier_method values × one outlier_flags combo = 2 sweep cells)
        assert len(result.data.raw.results) == 2

    def test_sweep_returns_one_inner_result_per_combination(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        """A 2 x 2 grid yields four inner results, aggregated into the one WorkflowResult."""
        cfg, data_dir = image_folder_pipeline_builder(
            workflows=[
                ParameterSweepWorkflowConfig(
                    name="sweep_grid",
                    type="parameter-sweep",
                    outlier_flags=["dimension", "pixel"],
                    outlier_method=["zscore", "modzscore"],
                    outlier_threshold=[2.0, 3.5],
                ),
            ],
            tasks=[
                ParameterSweepTaskConfig(
                    name="sweep_grid_task", workflow="sweep_grid", sources="main", extractor="flat"
                ),
            ],
        )

        results = run_tasks(cfg, data_dir=data_dir)

        assert len(results) == 1
        result = results[0]
        assert result.success
        combos = {(r.params["outlier_method"], r.params["outlier_threshold"]) for r in result.data.raw.results}
        assert combos == {("zscore", 2.0), ("zscore", 3.5), ("modzscore", 2.0), ("modzscore", 3.5)}
        assert result.metadata.sweep_parameters == ["outlier_method", "outlier_threshold"]

    def test_failing_combination_is_reported_and_later_tasks_still_run(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A combination that raises fails the sweep task with the cause recorded; the next task still runs."""
        from dataeval_flow.workflows.parameter_sweep import workflow as sweep_workflow

        real_outliers = sweep_workflow.Outliers

        def outliers(*, flags: object, outlier_threshold: tuple[str, float | None]) -> object:
            if outlier_threshold[0] == "modzscore":
                raise RuntimeError("modzscore combination exploded")
            return real_outliers(flags=flags, outlier_threshold=outlier_threshold)  # type: ignore[arg-type]

        monkeypatch.setattr(sweep_workflow, "Outliers", outliers)
        cfg, data_dir = image_folder_pipeline_builder(
            workflows=[
                ParameterSweepWorkflowConfig(
                    name="sweep_broken",
                    type="parameter-sweep",
                    outlier_flags=["dimension", "pixel"],
                    outlier_method=["zscore", "modzscore"],
                ),
                DataCleaningWorkflowConfig(
                    name="clean_main", type="data-cleaning", outlier_method="zscore", outlier_flags=["pixel"]
                ),
            ],
            tasks=[
                ParameterSweepTaskConfig(
                    name="sweep_broken_task", workflow="sweep_broken", sources="main", extractor="flat"
                ),
                TaskConfig(name="clean_task", workflow="clean_main", sources="main", extractor="flat"),
            ],
        )

        sweep, cleaning = run_tasks(cfg, data_dir=data_dir)

        assert not sweep.success
        assert any("modzscore combination exploded" in error for error in sweep.errors)
        assert cleaning.success
