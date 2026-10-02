"""Which Datasets an Output was computed on, through combines that read only Outputs; ports that require their
Outputs to share Datasets, or to have been computed on the step's own Datasets (ood-detection spec §6.1)."""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import pytest
from dataeval.quality import DuplicatesOutput
from pydantic import BaseModel, ValidationError

from dataeval_flow._input_spec import SourceCount
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import Combine, CombineConfig, CombineContext, DataType, Port
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages


class Pairing(BaseModel):
    """How many Outputs were paired."""

    count: int


class PairConfig(CombineConfig):
    input: str | list[str]


class Pair(Combine[PairConfig]):
    """Pairs Outputs that must share their Datasets."""

    name: ClassVar[str] = "toy-pair"
    description: ClassVar[str] = "Pairs Outputs of one comparison."
    inputs: ClassVar[tuple[Port, ...]] = (
        Port("input", DataType.OUTPUT, classes=(DuplicatesOutput,), count=SourceCount.ONE_OR_MORE),
    )
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.OUTPUT, classes=(Pairing,)),)
    shared_datasets: ClassVar[tuple[str, ...]] = ("input",)

    def run(self, config: PairConfig, inputs: Mapping[str, Any], context: CombineContext) -> Mapping[str, Any]:
        return {"output": Pairing(count=len(inputs["input"]))}


class Loose(Combine[PairConfig]):
    """Pairs Outputs from anywhere: it declares nothing."""

    name: ClassVar[str] = "toy-loose"
    description: ClassVar[str] = "Pairs any Outputs."
    inputs: ClassVar[tuple[Port, ...]] = Pair.inputs
    outputs: ClassVar[tuple[Port, ...]] = Pair.outputs

    def run(self, config: PairConfig, inputs: Mapping[str, Any], context: CombineContext) -> Mapping[str, Any]:
        return {"output": Pairing(count=len(inputs["input"]))}


class TiedConfig(CombineConfig):
    paired: str
    input: str


class Tied(Combine[TiedConfig]):
    """Reads a Pairing that must have been computed on its `input`."""

    name: ClassVar[str] = "toy-tied"
    description: ClassVar[str] = "Reads a pairing of its own Dataset."
    inputs: ClassVar[tuple[Port, ...]] = (
        Port("paired", DataType.OUTPUT, classes=(Pairing,)),
        Port("input", DataType.DATASET),
    )
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.OUTPUT, classes=(Pairing,)),)
    computed_on: ClassVar[Mapping[str, tuple[str, ...]]] = {"paired": ("input",)}

    def run(self, config: TiedConfig, inputs: Mapping[str, Any], context: CombineContext) -> Mapping[str, Any]:
        return {"output": inputs["paired"].value}


@pytest.fixture(autouse=True)
def _toys(plugins):
    plugins.setdefault("dataeval_flow.combines", []).extend(
        [
            ("toy-pair", "tests.test_combine_datasets:Pair"),
            ("toy-loose", "tests.test_combine_datasets:Loose"),
            ("toy-tied", "tests.test_combine_datasets:Tied"),
        ]
    )


def _load(steps: Sequence[dict[str, Any]], inputs: Sequence[Any] = ("a", "b")) -> Any:
    workflow = {"name": "w", "inputs": list(inputs), "steps": list(steps)}
    datasets = {"src": ToyImages(), "other": ToyImages(seed=1)}
    return chain_pipeline(workflows=[workflow], evaluators=[DuplicatesConfig(name="dupes")], datasets=datasets)


def _dupes(name: str, on: str) -> dict[str, Any]:
    return {"name": name, "evaluator": "dupes", "input": on}


def test_an_output_made_from_outputs_was_computed_on_what_they_were() -> None:
    _load(
        [
            _dupes("d1", "a"),
            _dupes("d2", "a"),
            {"name": "pair", "combine": "toy-pair", "input": ["d1", "d2"]},
            {"name": "tied", "combine": "toy-tied", "paired": "pair", "input": "a"},
        ]
    )


def test_outputs_a_port_must_share_but_computed_on_different_datasets_are_refused() -> None:
    wanted = r"Step 'pair' reads Outputs on `input` computed on different Datasets \(`d1` on `a`; `d2` on `b`\)"
    with pytest.raises(ValidationError, match=wanted):
        _load([_dupes("d1", "a"), _dupes("d2", "b"), {"name": "pair", "combine": "toy-pair", "input": ["d1", "d2"]}])


def test_a_port_that_declares_nothing_takes_outputs_from_different_datasets() -> None:
    _load([_dupes("d1", "a"), _dupes("d2", "b"), {"name": "loose", "combine": "toy-loose", "input": ["d1", "d2"]}])


def test_an_output_tied_to_dataset_ports_but_computed_elsewhere_is_refused() -> None:
    wanted = r"Step 'tied' reads `pair`, which was computed on `a`, not on `b`"
    with pytest.raises(ValidationError, match=wanted):
        _load(
            [
                _dupes("d1", "a"),
                {"name": "pair", "combine": "toy-pair", "input": ["d1"]},
                {"name": "tied", "combine": "toy-tied", "paired": "pair", "input": "b"},
            ]
        )


def test_each_element_of_a_list_pairs_its_own_outputs() -> None:
    inputs = [{"name": "tests", "list": True}]
    _load(
        [
            _dupes("d1", "tests"),
            _dupes("d2", "tests"),
            {"name": "pair", "combine": "toy-pair", "input": ["d1", "d2"]},
            {"name": "tied", "combine": "toy-tied", "paired": "pair", "input": "tests"},
        ],
        inputs=inputs,
    )
