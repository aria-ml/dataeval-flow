"""Ports, addresses and the Step base: what the engine reads to connect steps without running them."""

from collections.abc import Mapping
from typing import Any, ClassVar

import pytest
from dataeval.quality import DuplicatesOutput

from dataeval_flow import InputKind, SourceCount
from dataeval_flow.evaluators.quality import DuplicatesEvaluator
from dataeval_flow.steps import (
    Address,
    DataType,
    Port,
    Transform,
    TransformConfig,
    TransformContext,
    get_transform,
    list_transforms,
    parse_address,
)
from dataeval_flow.steps._registry import TRANSFORMS
from tests.workflow_toys import ToyCountResult, ToyCountWorkflow


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("clean", Address("clean")),
        ("split.train", Address("split", "train")),
        ("kfold.train[0]", Address("kfold", "train", "0")),
        ("tests[cam1]", Address("tests", None, "cam1")),
        ("a-b_c", Address("a-b_c")),
    ],
)
def test_an_address_names_a_step_its_output_and_an_element(text: str, expected: Address) -> None:
    assert parse_address(text) == expected
    assert str(parse_address(text)) == text


@pytest.mark.parametrize("text", ["", "1clean", "a.b.c", "a[", "a[]", "a b", "a/b", "a.[0]", "a[x][y]"])
def test_a_malformed_address_is_refused_naming_the_forms_it_takes(text: str) -> None:
    with pytest.raises(ValueError, match="is not an address"):
        parse_address(text)


def test_an_address_without_its_key_is_its_base() -> None:
    assert parse_address("kfold.train[0]").base == Address("kfold", "train")


def test_a_port_accepts_subclasses_of_the_classes_it_names() -> None:
    class Base: ...

    class Derived(Base): ...

    port = Port("plans", DataType.OUTPUT, classes=(Base,))
    assert port.accepts_class(Derived)
    assert not port.accepts_class(int)
    assert Port("any", DataType.OUTPUT).accepts_class(int)


def test_an_evaluator_is_a_step_whose_ports_come_from_its_inputs_and_output_class() -> None:
    (port,) = DuplicatesEvaluator.input_ports()
    assert (port.name, port.type, port.count) == ("input", DataType.DATASET, SourceCount.ONE_OR_MORE)
    assert InputKind.STATS in port.derives
    (out,) = DuplicatesEvaluator.output_ports()
    assert (out.name, out.type, out.classes) == ("output", DataType.OUTPUT, (DuplicatesOutput,))
    assert DuplicatesEvaluator.kind == "evaluator"


def test_a_workflow_is_a_step_whose_output_is_its_result() -> None:
    (port,) = ToyCountWorkflow.input_ports()
    assert (port.type, port.count) == (DataType.DATASET, SourceCount.ONE)
    (out,) = ToyCountWorkflow.output_ports()
    assert (out.type, out.classes) == (DataType.WORKFLOW_RESULT, (ToyCountResult,))
    assert ToyCountWorkflow.kind == "workflow"


class _KeepConfig(TransformConfig):
    input: str


class _Keep(Transform[_KeepConfig]):
    name: ClassVar[str] = "keep"
    description: ClassVar[str] = "Hands its input on unchanged."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)

    def run(self, config: _KeepConfig, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
        return {"output": inputs["input"].value}


def test_a_transform_declares_its_ports_and_binds_its_config() -> None:
    assert _Keep.config_type is _KeepConfig
    assert _Keep.kind == "transform"
    assert [port.name for port in _Keep.input_ports()] == ["input"]


def test_a_transform_without_ports_is_refused_when_defined() -> None:
    with pytest.raises(TypeError, match="inputs"):

        class _Portless(Transform[_KeepConfig]):
            name: ClassVar[str] = "portless"
            description: ClassVar[str] = "No ports."

            def run(self, config: Any, inputs: Mapping[str, Any], context: TransformContext) -> Mapping[str, Any]:
                return {}


def test_a_transform_config_refuses_a_key_it_does_not_define() -> None:
    with pytest.raises(ValueError, match="not_a_field"):
        _KeepConfig.model_validate({"input": "a", "not_a_field": 1})


def test_a_plugin_transform_is_registered_with_its_origin(plugins) -> None:
    plugins["dataeval_flow.transforms"] = [("keep", "tests.test_step_foundation:_Keep")]
    assert get_transform("keep") is _Keep
    assert _Keep in list_transforms()
    assert TRANSFORMS.origin("keep") != "dataeval-flow"
