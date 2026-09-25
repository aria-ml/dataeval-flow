"""TC-6-1 — workflow orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.workflows import Workflow, WorkflowResult, get_workflow, list_workflows

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from dataeval_flow import PipelineConfig


@pytest.mark.test_case("6-1")
class TestOrchestration:
    def test_list_workflows_returns_the_registered_set(self) -> None:
        wfs = list_workflows()
        names = {w.name for w in wfs}
        assert names == {
            "data-analysis",
            "data-cleaning",
            "data-coverage",
            "drift-monitoring",
            "ood-detection",
            "parameter-sweep",
            "data-prioritization",
            "data-splitting",
            "metadata-triage",
        }

    def test_get_workflow_returns_a_workflow_class(self) -> None:
        wf = get_workflow("data-cleaning")
        assert issubclass(wf, Workflow)

    def test_get_workflow_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            get_workflow("does-not-exist")

    def test_run_tasks_returns_results(self, synthetic_pipeline_config: tuple[PipelineConfig, Path]) -> None:
        cfg, data_dir = synthetic_pipeline_config
        results = run_tasks(cfg, data_dir=data_dir)
        assert list(results) == ["clean_task"]
        assert isinstance(results["clean_task"], WorkflowResult)
        assert results["clean_task"].metadata.tool == "dataeval-flow"
