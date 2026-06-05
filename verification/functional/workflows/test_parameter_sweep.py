"""TC-13-1 — parameter sweep workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config.schemas import (
    ParameterSweepTaskConfig,
    ParameterSweepWorkflowConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow import PipelineConfig


@pytest.mark.test_case("13-1")
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
