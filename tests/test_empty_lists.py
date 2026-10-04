"""A list input that may bind no source, and the one record a step run over it leaves (audit spec §9.1)."""

from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow import PipelineConfig
from dataeval_flow._app._model._state import ConfigState
from dataeval_flow._cache import DatasetCache
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import ChainResult, CustomWorkflowConfig, InputSlot
from tests.chain_toys import chain_pipeline, register_toys, run_chain_task
from tests.evaluator_toys import ToyImages

_REST = {"name": "rest", "list": True, "empty": "no other source given"}
_STEPS = [
    {"name": "dupes", "evaluator": "dupes", "input": ["a", "rest"]},
    {"name": "count", "combine": "toy-count-groups", "input": "dupes"},
    {"name": "judge", "check": "toy-at-most", "input": "count"},
]


@pytest.fixture(autouse=True)
def _toys(plugins):
    register_toys(plugins)
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _result(sources: list[str]) -> ChainResult:
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["a", _REST], "steps": _STEPS}],
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[{"name": "t", "workflow": "w", "sources": sources}],
        datasets={name: ToyImages(seed=index) for index, name in enumerate(sources)},
    )
    result = run_chain_task(config)
    assert isinstance(result, ChainResult)
    return result


def test_a_list_input_with_an_empty_reason_binds_no_source() -> None:
    workflow = CustomWorkflowConfig.model_validate({"name": "w", "inputs": ["a", _REST], "steps": _STEPS})
    assert workflow.binding_problem(1) is None
    assert workflow.binding_problem(0) == "takes one source for 'a' and any number for 'rest', but the task names 0."


def test_without_one_a_list_input_still_takes_a_source() -> None:
    workflow = CustomWorkflowConfig.model_validate(
        {"name": "w", "inputs": ["a", {"name": "rest", "list": True}], "steps": _STEPS}
    )
    assert workflow.binding_problem(1) == "takes one source for 'a' and at least one for 'rest', but the task names 1."


def test_a_single_input_takes_no_empty_reason() -> None:
    with pytest.raises(ValidationError, match="takes no `empty:`"):
        InputSlot.model_validate({"name": "a", "empty": "none"})


def test_a_step_run_over_an_empty_list_leaves_one_record_with_the_reason() -> None:
    result = _result(["a"])
    dupes, count, judge = result.steps["dupes"], result.steps["count"], result.steps["judge"]
    assert (dupes.status, dupes.reason, dupes.not_assessed, dupes.elements) == (
        "skipped",
        "no other source given",
        "no other source given",
        None,
    )
    assert (count.status, count.not_assessed) == ("skipped", "no other source given")
    assert judge.status == "ok"
    assert judge.not_assessed == "no other source given"
    (finding,) = judge.output
    assert (finding.severity, finding.brief, finding.description) == (
        "info",
        "not assessed",
        "Not assessed: no other source given.",
    )
    assert result.health["warnings"] == 0
    steps = cast(dict[str, Any], result.to_dict()["steps"])
    assert steps["judge"]["not_assessed"] == "no other source given"


def test_with_a_source_the_list_runs_as_before() -> None:
    result = _result(["a", "b"])
    elements = result.steps["dupes"].elements
    assert elements is not None
    assert list(elements) == ["b"]
    assert result.steps["judge"].not_assessed is None


def test_an_empty_reason_round_trips_through_a_dump_and_the_tui(tmp_path: Path) -> None:
    written: dict[str, Any] = {"name": "w", "inputs": ["a", _REST], "steps": _STEPS}
    assert CustomWorkflowConfig.model_validate(written).model_dump()["inputs"] == ["a", _REST]
    config = PipelineConfig.model_validate(
        {"evaluators": [{"name": "dupes", "type": "duplicates"}], "workflows": [written]}
    )
    state = ConfigState()
    state.load_dict(config)
    path = tmp_path / "pipeline.yaml"
    state.save_file(path)
    assert yaml.safe_load(path.read_text())["workflows"][0]["inputs"] == ["a", _REST]
