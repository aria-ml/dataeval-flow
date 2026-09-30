"""Combines and checks: their bases, their registries, and step entries naming them (spec §5.1, §5.2, §9.1)."""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from dataeval_flow._chain._graph import build_graph
from dataeval_flow.config._json_schema import registry_twin
from dataeval_flow.evaluators.quality import DuplicatesConfig
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
    list_steps,
)
from dataeval_flow.steps._registry import CHECKS
from dataeval_flow.workflows import Finding
from tests.chain_toys import GroupLimitConfig, chain_pipeline, register_toys
from tests.evaluator_toys import ToyImages

pytestmark = pytest.mark.usefixtures("toys")


@pytest.fixture
def toys(plugins):
    register_toys(plugins)
    return plugins


def _workflow(*steps: dict[str, Any]) -> CustomWorkflowConfig:
    return CustomWorkflowConfig.model_validate({"name": "w", "inputs": ["a"], "steps": list(steps)})


def test_a_check_s_one_output_is_its_findings() -> None:
    assert get_check("toy-at-most").output_ports() == (Port("findings", DataType.FINDINGS),)


def test_plugin_combines_and_checks_load_from_their_own_groups() -> None:
    assert get_combine("toy-count-groups").kind == "combine"
    assert get_check("toy-at-most").kind == "check"
    assert {"toy-at-most", "toy-worst"} <= {cls.name for cls in list_checks()}
    assert CHECKS.group == "dataeval_flow.checks"


def test_a_check_step_s_thresholds_are_validated_by_its_config() -> None:
    workflow = _workflow(
        {"name": "dupes", "evaluator": "dupes", "input": "a"},
        {"name": "count", "combine": "toy-count-groups", "input": "dupes"},
        {"name": "judge", "check": "toy-at-most", "input": "count", "most": 2},
    )
    config = workflow.steps[2].config
    assert isinstance(config, GroupLimitConfig)
    assert (config.input, config.most) == ("count", 2.0)


def test_a_misspelled_threshold_fails_the_load() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _workflow({"name": "judge", "check": "toy-at-most", "input": "count", "mots": 2})


def test_an_unknown_check_names_the_ones_installed() -> None:
    with pytest.raises(ValidationError, match=r"Unknown check: 'nope'\. Installed: \[.*'toy-at-most'"):
        _workflow({"name": "judge", "check": "nope", "input": "a"})


def test_combine_and_check_steps_are_written_back_as_written() -> None:
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


_DUPES = {"name": "dupes", "evaluator": "dupes", "input": "a"}
_COUNT = {"name": "count", "combine": "toy-count-groups", "input": "dupes"}


def _pipeline(*steps: dict[str, Any], inputs: list[Any] | None = None):
    return chain_pipeline(
        workflows=[{"name": "w", "inputs": inputs or ["a"], "steps": list(steps)}],
        evaluators=[DuplicatesConfig(name="dupes")],
        datasets={"src": ToyImages()},
    )


def test_an_evaluator_a_combine_and_a_check_chain_at_load() -> None:
    config = _pipeline(_DUPES, _COUNT, {"name": "judge", "check": "toy-at-most", "input": "count", "most": 1})
    graph = build_graph(config.workflows[0], config)  # type: ignore[index,arg-type]
    assert [(spec.kind, spec.type) for spec in graph.steps] == [
        ("evaluator", "quality.duplicates"),
        ("combine", "toy-count-groups"),
        ("check", "toy-at-most"),
    ]


def test_a_check_that_reads_a_dataset_fails_the_load() -> None:
    wanted = "Step 'judge' reads `a` on `input`, which takes an Output, but `a` is a Dataset"
    with pytest.raises(ValidationError, match=wanted):
        _pipeline({"name": "judge", "check": "toy-at-most", "input": "a"})


def test_a_check_that_reads_an_output_of_another_class_fails_the_load() -> None:
    with pytest.raises(ValidationError, match="which takes GroupCount, not DuplicatesOutput"):
        _pipeline(_DUPES, {"name": "judge", "check": "toy-at-most", "input": "dupes"})


def test_no_port_takes_a_check_s_findings() -> None:
    with pytest.raises(ValidationError, match="which takes an Output, but `judge` is findings"):
        _pipeline(
            _DUPES,
            _COUNT,
            {"name": "judge", "check": "toy-at-most", "input": "count"},
            {"name": "again", "combine": "toy-count-groups", "input": "judge"},
        )


def test_a_check_that_judges_a_whole_list_refuses_one_item() -> None:
    with pytest.raises(ValidationError, match="which takes a whole list, but `count` is one item"):
        _pipeline(_DUPES, _COUNT, {"name": "worst", "check": "toy-worst", "input": "count"})


def test_a_check_handed_a_list_on_a_one_item_port_runs_once_per_element() -> None:
    config = _pipeline(
        {"name": "dupes", "evaluator": "dupes", "input": "cams"},
        {"name": "count", "combine": "toy-count-groups", "input": "dupes"},
        {"name": "judge", "check": "toy-at-most", "input": "count"},
        inputs=[{"name": "cams", "list": True}],
    )
    graph = build_graph(config.workflows[0], config)  # type: ignore[index,arg-type]
    assert [spec.broadcast for spec in graph.steps] == [True, True, True]


def test_a_combine_or_check_step_takes_no_extractor() -> None:
    with pytest.raises(ValidationError, match="runs check 'toy-at-most', which embeds nothing"):
        _pipeline(_DUPES, _COUNT, {"name": "judge", "check": "toy-at-most", "input": "count", "extractor": "flat"})


def test_the_catalog_describes_a_check_s_ports() -> None:
    entry = next(e for e in list_steps().steps if e.type == "toy-at-most")
    assert entry.kind == "check"
    assert [(port.port, port.type, port.classes) for port in entry.inputs] == [
        ("input", "output", ["tests.chain_toys.GroupCount"])
    ]
    assert [(port.port, port.type, port.kinds) for port in entry.outputs] == [("findings", "findings", None)]
    assert entry.origin == "an unknown package"  # served by the fixture, from no distribution
    assert "findings" in list_steps().data_types


def test_the_schema_holds_one_branch_per_combine_and_check() -> None:
    definitions = registry_twin(plugins=True).model_json_schema()["$defs"]
    check = definitions["CheckStep_toy-at-most"]
    assert check["properties"]["check"]["const"] == "toy-at-most"
    assert {"name", "check", "input", "most", "optional"} <= set(check["properties"])
    assert "extractor" not in check["properties"]
    assert definitions["CombineStep_toy-count-groups"]["properties"]["combine"]["const"] == "toy-count-groups"
