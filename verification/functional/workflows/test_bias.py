"""TC-20-3 — data bias workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.config import TaskConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.data_bias import DataBiasConfig

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow.config import PipelineConfig


@pytest.mark.test_case("20-3")
class TestDataBiasWorkflow:
    def test_bias_workflow_runs_without_extractor(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        cfg, data_dir = image_folder_pipeline_builder(
            include_extractor=False,
            workflows=[DataBiasConfig(name="bias_main")],
            tasks=[TaskConfig(name="bias_task", workflow="bias_main", sources="main")],
        )
        result = run_tasks(cfg, data_dir=data_dir)["bias_task"]
        assert result.success
        assert isinstance(result, ChainResult)
        assert result.report().strip()
        titles = [finding.title for finding in result.findings]
        assert titles == ["Class Imbalance", "Shortcut Risk", "Factor Parity", "Factor Coverage Gaps"]
        assert result.steps["factor-summary"].output is not None
