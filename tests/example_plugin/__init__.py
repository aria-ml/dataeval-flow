"""A plugin written against the public API alone: one workflow, one evaluator, one extractor and one transform."""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from dataeval.flags import ImageStats
from dataeval.quality import Outliers, OutliersOutput
from pydantic import Field

from dataeval_flow import InputKind, InputSpec, ResultMetadata, SourceCount
from dataeval_flow.config import StatsConfigMixin
from dataeval_flow.config.extractors import Extractor, ExtractorConfig
from dataeval_flow.config.transforms import Transform
from dataeval_flow.evaluators import (
    Evaluator,
    EvaluatorConfig,
    EvaluatorInputs,
    EvaluatorResult,
)
from dataeval_flow.workflows import (
    Finding,
    Workflow,
    WorkflowConfig,
    WorkflowContext,
    WorkflowOutput,
    WorkflowRawOutput,
    WorkflowReport,
    WorkflowResult,
)


class CountRaw(WorkflowRawOutput):
    """Item counts per source."""

    counts: dict[str, int] = Field(default_factory=dict, description="Items in each source.")


class CountReport(WorkflowReport):
    """One finding per source."""


class CountOutput(WorkflowOutput[CountRaw, CountReport]):
    """What `example.count` produces."""


class CountMetadata(ResultMetadata):
    """The envelope of an `example.count` result."""


class CountResult(WorkflowResult[CountMetadata, CountOutput]):
    """The result of an `example.count` run."""


class CountConfig(WorkflowConfig[CountResult]):
    """Settings for `example.count`."""

    type: str = "example.count"
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset(), sources=SourceCount.ONE_OR_MORE)
    minimum: int = Field(default=0, ge=0, description="Fewest items a source may hold before it warns.")


class CountWorkflow(Workflow[CountConfig, CountResult]):
    """Counts each source's items and warns when one holds fewer than `minimum`."""

    name: ClassVar[str] = "example.count"
    description: ClassVar[str] = "Counts the items in each source."

    def run(self, config: CountConfig, context: WorkflowContext) -> CountResult:
        counts = {source: len(context.dataset(source)) for source in context.sources}
        findings = [
            Finding(
                report_type="key_value",
                severity="warning" if n < config.minimum else "ok",
                title=f"{source} items",
                data={"items": n},
            )
            for source, n in counts.items()
        ]
        raw = CountRaw(dataset_size=sum(counts.values()), counts=counts)
        output = CountOutput(raw=raw, report=CountReport(summary="Item counts", findings=findings))
        return CountResult(type=self.name, success=True, output=output, metadata=CountMetadata())


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


class Invert(Transform):
    """Inverts an image whose values lie in [0, 1]."""

    name: ClassVar[str] = "example.Invert"
    description: ClassVar[str] = "1 - x for each value."

    def __call__(self, data: Any, /) -> Any:
        return 1 - data

    def __repr__(self) -> str:
        return "Invert()"  # stable across runs: the embedding cache keys on the preprocessing's repr
