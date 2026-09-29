"""Combines and checks: their bases, their registries, and step entries naming them (spec §5.1, §5.2, §9.1)."""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from dataeval_flow.steps import (
    Check,
    CheckConfig,
    CheckContext,
    Combine,
    CombineConfig,
    CombineContext,
    CustomWorkflowConfig,
    DataType,
    Port,
    get_check,
    get_combine,
    list_checks,
)
from dataeval_flow.steps._registry import CHECKS
from dataeval_flow.workflows import Finding
from tests.chain_toys import AtMostConfig, register_toys


@pytest.fixture
def toys(plugins):
    register_toys(plugins)
    return plugins


def _workflow(*steps: dict[str, Any]) -> CustomWorkflowConfig:
    return CustomWorkflowConfig.model_validate({"name": "w", "inputs": ["a"], "steps": list(steps)})


def test_a_check_s_one_output_is_its_findings(toys) -> None:
    assert get_check("toy-at-most").output_ports() == (Port("findings", DataType.FINDINGS),)


def test_plugin_combines_and_checks_load_from_their_own_groups(toys) -> None:
    assert get_combine("toy-count-groups").kind == "combine"
    assert get_check("toy-at-most").kind == "check"
    assert {"toy-at-most", "toy-worst"} <= {cls.name for cls in list_checks()}
    assert CHECKS.group == "dataeval_flow.checks"


def test_a_check_step_s_thresholds_are_validated_by_its_config(toys) -> None:
    workflow = _workflow(
        {"name": "dupes", "evaluator": "dupes", "input": "a"},
        {"name": "count", "combine": "toy-count-groups", "input": "dupes"},
        {"name": "judge", "check": "toy-at-most", "input": "count", "most": 2},
    )
    config = workflow.steps[2].config
    assert isinstance(config, AtMostConfig)
    assert (config.input, config.most) == ("count", 2.0)


def test_a_misspelled_threshold_fails_the_load(toys) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _workflow({"name": "judge", "check": "toy-at-most", "input": "count", "mots": 2})


def test_an_unknown_check_names_the_ones_installed(toys) -> None:
    with pytest.raises(ValidationError, match=r"Unknown check: 'nope'\. Installed: \[.*'toy-at-most'"):
        _workflow({"name": "judge", "check": "nope", "input": "a"})


def test_combine_and_check_steps_are_written_back_as_written(toys) -> None:
    steps = [
        {"name": "count", "combine": "toy-count-groups", "input": "dupes"},
        {"name": "judge", "check": "toy-at-most", "input": "count", "most": 2},
        {"name": "lax", "check": "toy-at-most", "input": "count", "optional": True},
    ]
    written = _workflow({"name": "dupes", "evaluator": "dupes", "input": "a"}, *steps).model_dump(mode="json")
    assert written["steps"][1:] == steps


def test_a_check_that_reads_a_dataset_is_refused_when_defined() -> None:
    with pytest.raises(TypeError, match=r"Peek's input `input` carries dataset: a check reads Outputs"):

        class Peek(Check[CheckConfig]):
            name: ClassVar[str] = "peek"
            description: ClassVar[str] = "Reads a Dataset."
            title: ClassVar[str] = "Peek"
            inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)

            def run(self, config: CheckConfig, inputs: Mapping[str, Any], context: CheckContext) -> Sequence[Finding]:
                return []


def test_a_check_without_a_title_is_refused_when_defined() -> None:
    with pytest.raises(TypeError, match="must declare title"):

        class Untitled(Check[CheckConfig]):
            name: ClassVar[str] = "untitled"
            description: ClassVar[str] = "No title."
            inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT),)

            def run(self, config: CheckConfig, inputs: Mapping[str, Any], context: CheckContext) -> Sequence[Finding]:
                return []


def test_a_combine_that_makes_a_dataset_is_refused_when_defined() -> None:
    with pytest.raises(TypeError, match=r"Maker's output `output` carries dataset: a combine makes Outputs"):

        class Maker(Combine[CombineConfig]):
            name: ClassVar[str] = "maker"
            description: ClassVar[str] = "Makes a Dataset."
            inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT),)
            outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)

            def run(
                self, config: CombineConfig, inputs: Mapping[str, Any], context: CombineContext
            ) -> Mapping[str, Any]:
                return {}
