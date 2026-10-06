"""`groups:` on a custom workflow: report headings over the findings of the checks they name (follow-ons spec §7)."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import load_config, run_tasks
from dataeval_flow._app._model._state import ConfigState
from dataeval_flow._blocks import Fields
from dataeval_flow._cache import DatasetCache
from dataeval_flow.config import ImageFolderDatasetConfig, PipelineConfig, SourceConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig, LabelHealthConfig
from dataeval_flow.steps import ChainResult, CustomWorkflowConfig
from tests.chain_toys import chain_pipeline
from tests.test_chain_verdict_report import _outline, _section, _top

_STEPS = [
    {"name": "dupes", "evaluator": "dupes", "input": "data"},
    {"name": "labels", "evaluator": "labels", "input": "data"},
    {"name": "duplicates", "check": "image-duplicates", "input": "dupes"},
    {"name": "imbalance", "check": "class-imbalance", "input": "labels"},
]
_CLEAN = {"heading": "Is the data clean?", "checks": ["image-duplicates"]}


@pytest.fixture(autouse=True)
def _fresh_caches():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _workflow(groups: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"name": "w", "inputs": ["data"], "steps": _STEPS, **({"groups": groups} if groups is not None else {})}


def _run(groups: list[dict[str, Any]] | None = None) -> ChainResult:
    config = chain_pipeline(
        workflows=[_workflow(groups)],
        evaluators=[DuplicatesConfig(name="dupes"), LabelHealthConfig(name="labels")],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    return result


def test_a_group_heads_the_findings_of_the_checks_it_names() -> None:
    result = _run([_CLEAN])
    assert "Is the data clean?" in _outline(result)
    assert _section(_top(result), "Is the data clean?").brief


def test_the_short_report_lists_each_group_with_its_status() -> None:
    questions = _section(_top(_run([_CLEAN]), detailed=False), "Questions")
    (fields,) = [block for block in questions.blocks if isinstance(block, Fields)]
    assert [label for label, _ in fields.items] == ["Is the data clean?"]


def test_groups_give_no_verdict() -> None:
    assert _run([_CLEAN]).verdict is None


def test_without_groups_the_report_is_laid_out_as_before() -> None:
    result = _run()
    assert result.preset_chain is None
    assert "Is the data clean?" not in _outline(result)


def test_a_group_naming_a_check_no_step_runs_is_refused() -> None:
    with pytest.raises(ValidationError, match=r"names check `leakage`, which no step runs\. Its steps' checks: "):
        CustomWorkflowConfig.model_validate(_workflow([{"heading": "Splits", "checks": ["leakage"]}]))


def test_groups_round_trip_through_to_yaml() -> None:
    import yaml

    workflow = CustomWorkflowConfig.model_validate(_workflow([_CLEAN]))
    (written,) = yaml.safe_load(workflow.to_yaml())["workflows"]
    assert written["groups"] == [_CLEAN]
    assert CustomWorkflowConfig.model_validate(written) == workflow


def test_groups_survive_the_builder_s_save(tmp_path: Path) -> None:
    config = PipelineConfig.model_validate(
        {
            "datasets": [ImageFolderDatasetConfig(name="src_data", path="images")],
            "sources": [SourceConfig(name="src", dataset="src_data")],
            "evaluators": [DuplicatesConfig(name="dupes"), LabelHealthConfig(name="labels")],
            "workflows": [_workflow([_CLEAN])],
        }
    )
    state = ConfigState()
    state.load_dict(config)
    state.save_file(tmp_path / "pipeline.yaml")
    (workflow,) = load_config(tmp_path / "pipeline.yaml").workflows or []
    assert isinstance(workflow, CustomWorkflowConfig)
    assert [group.heading for group in workflow.groups] == ["Is the data clean?"]
