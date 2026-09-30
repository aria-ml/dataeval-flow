"""Toy presets, served through the `plugins` fixture: workflow types whose settings expand to a chain of steps."""

from typing import Any, ClassVar

from pydantic import Field

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.workflows import Workflow, WorkflowConfig
from dataeval_flow.workflows._preset import Preset, PresetChain


class ToyPresetConfig(WorkflowConfig[ChainResult]):
    """Duplicates found, judged against `exact`, and removed."""

    type: str = Field(default="toy-preset", description="The workflow type this entry configures.")
    # Clusters are optional, as for `quality.duplicates`, so a step running this preset may name an extractor.
    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.STATS}), optional=frozenset({InputKind.CLUSTERS}), sources=SourceCount.ONE
    )
    exact: float | None = Field(
        default=0.0, description="The share of images in exact-duplicate groups that warns; `None` judges nothing."
    )


class ToyPreset(Preset, Workflow[ToyPresetConfig, ChainResult]):
    """One evaluator, one check and one transform, with its own evaluator entry."""

    name: ClassVar[str] = "toy-preset"
    description: ClassVar[str] = "Finds, judges and removes duplicates."
    slots: ClassVar[tuple[str, ...]] = ("data",)
    outputs: ClassVar[tuple[Port, ...]] = (Port("kept", DataType.DATASET),)

    @classmethod
    def chain(cls, config: ToyPresetConfig) -> PresetChain:
        """Duplicates, the rate they occur at, and the data without them."""
        return PresetChain(
            steps=[
                {"name": "dupes", "evaluator": "dupes", "input": "data"},
                {"name": "rate", "check": "duplicate-rate", "input": "dupes", "exact": config.exact},
                {"name": "kept", "transform": "remove", "input": "data", "plans": {"dupes": {}}},
            ],
            evaluators=[DuplicatesConfig(name="dupes")],
        )


class BrokenPresetConfig(WorkflowConfig[ChainResult]):
    """A preset entry whose chain does not connect."""

    type: str = Field(default="toy-broken-preset", description="The workflow type this entry configures.")
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset(), sources=SourceCount.ONE)


class BrokenPreset(Preset, Workflow[BrokenPresetConfig, ChainResult]):
    """Removes by a plan no step made."""

    name: ClassVar[str] = "toy-broken-preset"
    description: ClassVar[str] = "Reads a step it does not have."
    slots: ClassVar[tuple[str, ...]] = ("data",)

    @classmethod
    def chain(cls, config: Any) -> PresetChain:  # noqa: ARG003
        """One step, reading a step that does not exist."""
        return PresetChain(steps=[{"name": "kept", "transform": "remove", "input": "data", "plans": {"nowhere": {}}}])


class GonePresetConfig(WorkflowConfig[ChainResult]):
    """A preset entry that declares an output its chain does not make."""

    type: str = Field(default="toy-gone-preset", description="The workflow type this entry configures.")
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.STATS}), sources=SourceCount.ONE)


class GonePreset(Preset, Workflow[GonePresetConfig, ChainResult]):
    """Declares `gone`, which no step makes."""

    name: ClassVar[str] = "toy-gone-preset"
    description: ClassVar[str] = "Declares an output no step makes."
    slots: ClassVar[tuple[str, ...]] = ("data",)
    outputs: ClassVar[tuple[Port, ...]] = (Port("gone", DataType.DATASET),)

    @classmethod
    def chain(cls, config: Any) -> PresetChain:  # noqa: ARG003
        """Duplicates, and nothing named `gone`."""
        return PresetChain(
            steps=[{"name": "dupes", "evaluator": "dupes", "input": "data"}],
            evaluators=[DuplicatesConfig(name="dupes")],
        )


_PRESETS = {
    "toy-preset": "tests.preset_toys:ToyPreset",
    "toy-broken-preset": "tests.preset_toys:BrokenPreset",
    "toy-gone-preset": "tests.preset_toys:GonePreset",
}


def register_presets(plugins: dict[str, list[tuple[str, str]]]) -> None:
    """Serve the toy presets through the `plugins` fixture, before the first registry lookup."""
    plugins.setdefault("dataeval_flow.workflows", []).extend(_PRESETS.items())
