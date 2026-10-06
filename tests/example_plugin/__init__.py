"""A plugin written against the public API alone: one workflow type over its own combine and check, one evaluator, one
extractor and one transform."""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from dataeval.flags import ImageStats
from dataeval.quality import Outliers, OutliersOutput
from pydantic import BaseModel, Field

from dataeval_flow import InputKind, InputSpec, SourceCount
from dataeval_flow.config import StatsConfigMixin
from dataeval_flow.config.extractors import Extractor, ExtractorConfig
from dataeval_flow.config.image_transforms import ImageTransform
from dataeval_flow.evaluators import (
    Evaluator,
    EvaluatorConfig,
    EvaluatorInputs,
    EvaluatorResult,
)
from dataeval_flow.steps import (
    ChainResult,
    Check,
    CheckConfig,
    CheckContext,
    Combine,
    CombineConfig,
    CombineContext,
    DataType,
    Finding,
    InputSlot,
    Port,
)
from dataeval_flow.workflows import Preset, PresetChain, Workflow, WorkflowConfig


class ItemCount(BaseModel):
    """How many items a Dataset holds."""

    items: int = Field(description="Items in the Dataset.")


class ItemsConfig(CombineConfig):
    """Settings for `example.items`."""

    input: str = Field(description="The Dataset to count.")


class ItemsCombine(Combine[ItemsConfig]):
    """Counts a Dataset's items."""

    name: ClassVar[str] = "example.items"
    description: ClassVar[str] = "Counts a Dataset's items."
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.DATASET),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.OUTPUT, classes=(ItemCount,)),)

    def run(self, config: ItemsConfig, inputs: Mapping[str, Any], context: CombineContext) -> Mapping[str, Any]:
        return {"output": ItemCount(items=len(inputs["input"].value))}


class AtLeastConfig(CheckConfig):
    """Settings for `example.at-least`."""

    input: str = Field(description="The count to judge.")
    minimum: int = Field(default=0, ge=0, description="Fewest items a Dataset may hold before it warns.")


class AtLeastCheck(Check[AtLeastConfig]):
    """Warns when a Dataset holds fewer than `minimum` items."""

    name: ClassVar[str] = "example.at-least"
    description: ClassVar[str] = "Warns below a count of items."
    title: ClassVar[str] = "Items"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(ItemCount,)),)

    def run(self, config: AtLeastConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:
        n = inputs["input"].value.items
        return [Finding(severity="warning" if n < config.minimum else "ok", title=self.title, brief=f"{n} items")]


class CountConfig(WorkflowConfig[ChainResult]):
    """Settings for `example.count`."""

    type: str = "example.count"
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset(), sources=SourceCount.ONE)
    minimum: int = Field(default=0, ge=0, description="Fewest items the source may hold before it warns.")


class CountWorkflow(Preset, Workflow[CountConfig, ChainResult]):
    """Counts the source's items and warns when it holds fewer than `minimum`."""

    name: ClassVar[str] = "example.count"
    description: ClassVar[str] = "Counts the items in a source."
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)

    @classmethod
    def chain(cls, config: CountConfig) -> PresetChain:
        return PresetChain(
            steps=[
                {"name": "items", "combine": "example.items", "input": "data"},
                {"name": "at-least", "check": "example.at-least", "input": "items", "minimum": config.minimum},
            ]
        )


class BrightnessResult(EvaluatorResult[OutliersOutput[Any]]):
    """The result of an `example.brightness` run: DataEval's `OutliersOutput`."""


class BrightnessConfig(EvaluatorConfig[BrightnessResult], StatsConfigMixin):
    """Settings for `example.brightness`."""

    type: str = "example.brightness"
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.STATS}), sources=SourceCount.ONE)

    def stats_request(self) -> dict[str, Any]:
        return {"outlier_flags": ImageStats.VISUAL_BRIGHTNESS}


class BrightnessEvaluator(Evaluator[BrightnessConfig, OutliersOutput[Any]]):
    """Flags images whose brightness is a z-score outlier."""

    name: ClassVar[str] = "example.brightness"
    description: ClassVar[str] = "Images whose brightness is an outlier."
    dataeval_class: ClassVar[type] = Outliers
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.STATS: "from_stats"}

    def run(self, config: BrightnessConfig, inputs: Sequence[EvaluatorInputs]) -> OutliersOutput[Any]:
        (source,) = inputs
        assert source.stats is not None
        return Outliers(flags=ImageStats.VISUAL_BRIGHTNESS, outlier_threshold="zscore").from_stats(source.stats)


class MeanConfig(ExtractorConfig):
    """Settings for `example.mean`."""

    model: str = "example.mean"


class _Means:
    """Each image's per-channel mean, after the entry's preprocessing: a DataEval `FeatureExtractor`."""

    def __init__(self, transforms: Any) -> None:
        self.transforms = transforms

    def __call__(self, data: Any, /) -> Any:
        images = data if self.transforms is None else [self.transforms(image) for image in data]
        return np.stack([np.asarray(image, dtype=np.float32).mean(axis=(-2, -1)) for image in images])


class MeanExtractor(Extractor[MeanConfig]):
    """Embeds each image as its per-channel means."""

    name: ClassVar[str] = "example.mean"
    description: ClassVar[str] = "Per-channel mean of each image."

    def build(self, config: MeanConfig, transforms: Any) -> Any:
        return _Means(transforms)


class Invert(ImageTransform):
    """Inverts an image whose values lie in [0, 1]."""

    name: ClassVar[str] = "example.Invert"
    description: ClassVar[str] = "1 - x for each value."

    def __call__(self, data: Any, /) -> Any:
        return 1 - data

    def __repr__(self) -> str:
        return "Invert()"  # stable across runs: the embedding cache keys on the preprocessing's repr
