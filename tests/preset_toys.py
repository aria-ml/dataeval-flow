"""Toy presets, served through the `plugins` fixture: workflow types whose settings expand to a chain of steps."""

from typing import Any, ClassVar

from pydantic import Field

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin
from dataeval_flow.evaluators.bias import FactorSummaryConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig, FactorTriageConfig, LabelHealthConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.workflows import Workflow, WorkflowConfig
from dataeval_flow.workflows._preset import Preset, PresetChain


class ToyPresetConfig(WorkflowConfig[ChainResult]):
    """Duplicates found, judged against `exact`, and removed."""

    type: str = Field(default="toy-preset", description="The workflow type this entry configures.")
    # Clusters are optional, as for `duplicates`, so a step running this preset may name an extractor.
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
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)
    outputs: ClassVar[tuple[Port, ...]] = (Port("kept", DataType.DATASET),)

    @classmethod
    def chain(cls, config: ToyPresetConfig) -> PresetChain:
        """Duplicates, the rate they occur at, and the data without them."""
        return PresetChain(
            steps=[
                {"name": "dupes", "evaluator": "dupes", "input": "data"},
                {"name": "rate", "check": "image-duplicates", "input": "dupes", "exact": config.exact},
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
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

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
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)
    outputs: ClassVar[tuple[Port, ...]] = (Port("gone", DataType.DATASET),)

    @classmethod
    def chain(cls, config: Any) -> PresetChain:  # noqa: ARG003
        """Duplicates, and nothing named `gone`."""
        return PresetChain(
            steps=[{"name": "dupes", "evaluator": "dupes", "input": "data"}],
            evaluators=[DuplicatesConfig(name="dupes")],
        )


class ToyPoolPresetConfig(WorkflowConfig[ChainResult]):
    """Duplicates found in a reference, and found and removed in each pool."""

    type: str = Field(default="toy-pool-preset", description="The workflow type this entry configures.")
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.STATS}), sources=SourceCount.TWO_OR_MORE)


class ToyPoolPreset(Preset, Workflow[ToyPoolPresetConfig, ChainResult]):
    """A single slot and a list slot: the reference's duplicates found once, each pool's found and removed."""

    name: ClassVar[str] = "toy-pool-preset"
    description: ClassVar[str] = "Finds duplicates in a reference, and removes each pool's own."
    slots: ClassVar[tuple[str | InputSlot, ...]] = (
        "reference",
        InputSlot.model_validate({"name": "pools", "list": True}),
    )
    outputs: ClassVar[tuple[Port, ...]] = (Port("kept", DataType.DATASET),)

    @classmethod
    def chain(cls, config: Any) -> PresetChain:  # noqa: ARG003
        """The reference's duplicates, each pool's, and each pool without its own."""
        return PresetChain(
            steps=[
                {"name": "reference-dupes", "evaluator": "dupes", "input": "reference"},
                {"name": "dupes", "evaluator": "dupes", "input": "pools"},
                {"name": "kept", "transform": "remove", "input": "pools", "plans": {"dupes": {}}},
            ],
            evaluators=[DuplicatesConfig(name="dupes")],
        )


class ToySplitPresetConfig(WorkflowConfig[ChainResult]):
    """A split into train and test, holding out no val."""

    type: str = Field(default="toy-split-preset", description="The workflow type this entry configures.")
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.METADATA}), sources=SourceCount.ONE)


class ToySplitPreset(Preset, Workflow[ToySplitPresetConfig, ChainResult]):
    """Declares `train`, `val` and `test`, read from its `parts` step's outputs; `val` is empty."""

    name: ClassVar[str] = "toy-split-preset"
    description: ClassVar[str] = "Splits off a test, holding out no val."
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)
    outputs: ClassVar[tuple[Port, ...]] = (
        Port("train", DataType.DATASET),
        Port("val", DataType.DATASET),
        Port("test", DataType.DATASET),
    )

    @classmethod
    def chain(cls, config: Any) -> PresetChain:  # noqa: ARG003
        """A `split` holding out a quarter as test."""
        return PresetChain(
            steps=[{"name": "parts", "transform": "split", "input": "data", "test_frac": 0.25}],
            outputs={"train": "parts.train", "val": "parts.val", "test": "parts.test"},
        )


class ToyFoldPresetConfig(WorkflowConfig[ChainResult]):
    """Two folds."""

    type: str = Field(default="toy-fold-preset", description="The workflow type this entry configures.")
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.METADATA}), sources=SourceCount.ONE)
    to_labels: bool = Field(default=False, description="Whether `train` is mapped to an Output, which is refused.")


class ToyFoldPreset(Preset, Workflow[ToyFoldPresetConfig, ChainResult]):
    """Declares `train`, read from `kfold`'s list of trains."""

    name: ClassVar[str] = "toy-fold-preset"
    description: ClassVar[str] = "Makes two folds."
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)
    outputs: ClassVar[tuple[Port, ...]] = (Port("train", DataType.DATASET),)

    @classmethod
    def chain(cls, config: ToyFoldPresetConfig) -> PresetChain:
        """Two folds, and the whole's label health."""
        return PresetChain(
            steps=[
                {"name": "parts", "transform": "kfold", "input": "data", "folds": 2},
                {"name": "labels", "evaluator": "labels", "input": "data"},
            ],
            evaluators=[LabelHealthConfig(name="labels")],
            outputs={"train": "labels" if config.to_labels else "parts.train"},
        )


_PRESETS = {
    "toy-split-preset": "tests.preset_toys:ToySplitPreset",
    "toy-fold-preset": "tests.preset_toys:ToyFoldPreset",
    "toy-pool-preset": "tests.preset_toys:ToyPoolPreset",
    "toy-preset": "tests.preset_toys:ToyPreset",
    "toy-broken-preset": "tests.preset_toys:BrokenPreset",
    "toy-gone-preset": "tests.preset_toys:GonePreset",
}


class ToyReferencePresetConfig(WorkflowConfig[ChainResult], MetadataConfigMixin):
    """Each split's metadata summary and triage, every split on train's encoding."""

    type: str = Field(default="toy-reference-preset", description="The workflow type this entry configures.")
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.METADATA}), sources=SourceCount.TWO_OR_MORE)


class ToyReferencePreset(Preset, Workflow[ToyReferencePresetConfig, ChainResult]):
    """`train` and a list `evals`, with `train` the reference."""

    name: ClassVar[str] = "toy-reference-preset"
    description: ClassVar[str] = "Summarizes every split on train's encoding."
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("train", InputSlot.model_validate({"name": "evals", "list": True}))

    @classmethod
    def chain(cls, config: Any) -> PresetChain:
        return PresetChain(
            steps=[
                {"name": "train-summary", "evaluator": "summary", "input": "train"},
                {"name": "evals-summary", "evaluator": "summary", "input": "evals"},
                {"name": "train-triage", "evaluator": "triage", "input": "train"},
                {"name": "evals-triage", "evaluator": "triage", "input": "evals"},
                {
                    "name": "parts",
                    "transform": "split",
                    "input": "evals",
                    "test_frac": 0.4,
                    "metadata": config.metadata,
                },
                {"name": "parts-summary", "evaluator": "summary", "input": "parts.train"},
            ],
            evaluators=[
                FactorSummaryConfig(name="summary", metadata=config.metadata),
                FactorTriageConfig(name="triage", metadata=config.metadata, verify=False),
            ],
            reference="train",
        )


_PRESETS["toy-reference-preset"] = "tests.preset_toys:ToyReferencePreset"


def register_presets(plugins: dict[str, list[tuple[str, str]]]) -> None:
    """Serve the toy presets through the `plugins` fixture, before the first registry lookup."""
    plugins.setdefault("dataeval_flow.workflows", []).extend(_PRESETS.items())
