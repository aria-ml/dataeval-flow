"""`factor-predictors` and `factor-deviation`: the metadata factors behind the images OOD detectors flagged
(ood-detection spec §5.4)."""

__all__ = [
    "CollectedFactors",
    "FactorDeviation",
    "FactorDeviationCombine",
    "FactorDeviationConfig",
    "FactorDeviationOutput",
    "FactorPredictorsOutput",
    "FactorPredictorsCombine",
    "FactorPredictorsConfig",
    "collect_factors",
]

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from dataeval.shift import OODOutput
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from dataeval_flow._blocks import Block, Cell, Column, Paragraph, Table
from dataeval_flow._input_spec import InputKind
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin, StatsConfigMixin
from dataeval_flow.evaluators.bias._report import ranked_table
from dataeval_flow.steps._combine import Combine, CombineConfig, CombineContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.combines._ood import OODUnionOutput

_DERIVES = frozenset({InputKind.METADATA, InputKind.STATS})
_PORTS: tuple[Port, ...] = (
    Port("ood", DataType.OUTPUT, classes=(OODUnionOutput, OODOutput)),
    Port("reference", DataType.DATASET, derives=_DERIVES),
    Port("input", DataType.DATASET, derives=_DERIVES),
)
_COMPUTED_ON: Mapping[str, tuple[str, ...]] = {"ood": ("reference", "input")}
_NONE_LEFT = "No factor was left to compare: none both Datasets have is numeric, finite and varies in the test."


@dataclass(frozen=True)
class CollectedFactors:
    """The factors both Datasets have, by name, kept by legacy's rules; and what could not be read, and why."""

    reference: dict[str, NDArray[Any]]
    test: dict[str, NDArray[Any]]
    unavailable: list[str] = field(default_factory=list)


def _metadata_factors(metadata: Any) -> dict[str, NDArray[Any]]:
    """One Dataset's metadata factors, one value per item, without `id`; with `class_label` where its class labels
    are numeric and one per item."""
    rows = metadata.rows_at(metadata.item_level)
    factors = {name: rows[name].to_numpy() for name in metadata.factor_names if name in rows.columns}
    factors.pop("id", None)
    try:
        labels = metadata.class_labels
    except (AttributeError, ValueError):
        labels = None
    if labels is not None:
        labels = np.asarray(labels)
        if np.issubdtype(labels.dtype, np.number) and len(labels) == rows.height:
            factors["class_label"] = labels
    return factors


def _stats_factors(stats: Mapping[str, Any]) -> dict[str, NDArray[Any]]:
    """One Dataset's per-image statistics as factors named `f_<statistic>`, where numeric and one per image."""
    count = stats.get("image_count", 0)
    arrays = ((name, np.asarray(values)) for name, values in (stats.get("stats") or {}).items())
    return {
        f"f_{name}": array for name, array in arrays if np.issubdtype(array.dtype, np.number) and len(array) == count
    }


def _usable(reference: NDArray[Any], test: NDArray[Any]) -> bool:
    """Legacy's rule for a factor both Datasets have: numeric, one-dimensional, finite in both, not constant in the
    test."""
    if not (np.issubdtype(reference.dtype, np.number) and np.issubdtype(test.dtype, np.number)):
        return False
    if reference.ndim != 1 or len(test) == 0:
        return False
    return bool(np.all(np.isfinite(reference)) and np.all(np.isfinite(test)) and np.std(test) != 0)


def collect_factors(
    context: CombineContext, reference: Any, test: Any, keep: NDArray[np.intp] | None = None
) -> CollectedFactors:
    """The factors the reference and test Datasets both have, from their metadata and statistics, kept by legacy's
    rules. `keep` narrows the test to those images first. Where one Dataset's metadata or statistics cannot be read,
    the other half is read alone, and `unavailable` says what was missing and why."""
    unavailable: list[str] = []
    sides: list[dict[str, NDArray[Any]]] = []
    for node in (reference, test):
        readers: list[tuple[str, Callable[[], dict[str, NDArray[Any]]]]] = [
            ("metadata", lambda node=node: _metadata_factors(context.derive_metadata(node))),
            ("statistics", lambda node=node: _stats_factors(context.derive_stats(node))),
        ]
        parts: dict[str, NDArray[Any]] = {}
        for label, read in readers:
            try:
                parts |= read()
            except Exception as error:  # noqa: BLE001 - legacy read on with the other half (spec §5.4)
                unavailable.append(f"the {label} of `{node.address}`: {error}")
        sides.append(parts)
    reference_factors, test_factors = sides
    if keep is not None:
        test_factors = {name: values[keep] for name, values in test_factors.items()}
    kept = [
        name
        for name in sorted(set(reference_factors) & set(test_factors))
        if _usable(reference_factors[name], test_factors[name])
    ]
    return CollectedFactors(
        {name: reference_factors[name] for name in kept}, {name: test_factors[name] for name in kept}, unavailable
    )


@dataclass(frozen=True)
class _Flagged:
    flagged: list[int]
    agreed: list[int]
    assessed: NDArray[np.bool_]
    scores: list[float | None]


def _flagged(value: Any) -> _Flagged:
    """What an `ood-union` Output, or one OOD Output, flagged: every flagged image; the agreed ones, most out of
    distribution first; which images were assessed; and each image's score."""
    if isinstance(value, OODUnionOutput):
        ranked = sorted(value.mutual, key=lambda index: (-(value.scores[index] or 0.0), index))
        return _Flagged(value.union, ranked, np.asarray([score is not None for score in value.scores]), value.scores)
    scores = np.asarray(value.instance_score, dtype=float)
    assessed = np.isfinite(scores)
    flagged = [int(index) for index in np.flatnonzero(value.is_ood)]
    listed = [float(score) if ok else None for score, ok in zip(scores, assessed, strict=True)]
    return _Flagged(flagged, sorted(flagged, key=lambda index: (-scores[index], index)), assessed, listed)


class FactorPredictorsOutput(BaseModel):
    """How strongly each metadata factor goes with being flagged, strongest first."""

    factors: dict[str, float] = Field(
        description=(
            "Each factor's normalized mutual information with being flagged, 0 to 1, from DataEval's "
            "`factor_predictors`, rounded to 4 places, strongest first."
        )
    )
    flagged: int = Field(description="The flagged images read, among the assessed ones.")
    unavailable: list[str] = Field(
        default_factory=list, description="What could not be read, and why; the rest was read without it."
    )
    reason: str | None = Field(default=None, description="Why no factor was compared, where none was.")


class FactorDeviation(BaseModel):
    """One agreed image, and how far each factor sets it apart from the reference."""

    index: int = Field(description="The test image.")
    score: float | None = Field(description="Its agreement score, or its one detector's score.")
    deviations: dict[str, float] = Field(
        description=(
            "Each factor's deviation from the reference, most deviating first, from DataEval's `factor_deviation`."
        )
    )


class FactorDeviationOutput(BaseModel):
    """The factors that set each of the most out-of-distribution agreed images apart from the reference."""

    source: str | None = Field(description="The test Dataset, whose items the indices name.")
    items: list[FactorDeviation] = Field(description="The agreed images explained, most out of distribution first.")
    unavailable: list[str] = Field(
        default_factory=list, description="What could not be read, and why; the rest was read without it."
    )
    reason: str | None = Field(default=None, description="Why no image is explained, where none is.")


class _FactorsConfig(CombineConfig, MetadataConfigMixin, StatsConfigMixin):
    ood: str = Field(
        description="An `ood-union` Output, or one OOD evaluator's Output, computed on `reference` and `input`."
    )
    reference: str = Field(description="The reference Dataset the detectors fitted on.")
    input: str = Field(description="The test Dataset whose images were flagged.")


class FactorPredictorsConfig(_FactorsConfig):
    """A `factor-predictors` step's inputs, and the policies its metadata and statistics are read under."""


class FactorDeviationConfig(_FactorsConfig):
    """A `factor-deviation` step's inputs, its policies, and how many images it explains."""

    max_items: int = Field(
        default=50,
        gt=0,
        description=("The most out-of-distribution agreed images explained, at most."),
    )


def _notes(unavailable: list[str], reason: str | None) -> list[Block]:
    blocks: list[Block] = [Paragraph(text=reason)] if reason else []
    if unavailable:
        blocks.append(Paragraph(text=f"Read without {'; '.join(unavailable)}."))
    return blocks


class FactorPredictorsCombine(Combine[FactorPredictorsConfig]):
    """``factor-predictors``: how strongly each metadata factor goes with the images OOD detectors flagged."""

    name: ClassVar[str] = "factor-predictors"
    title: ClassVar[str] = "Factor Predictors"
    description: ClassVar[str] = "Ranks the metadata factors that go with the images OOD detectors flagged."
    inputs: ClassVar[tuple[Port, ...]] = _PORTS
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.OUTPUT, classes=(FactorPredictorsOutput,)),)
    computed_on: ClassVar[Mapping[str, tuple[str, ...]]] = _COMPUTED_ON

    def run(
        self,
        config: FactorPredictorsConfig,  # noqa: ARG002
        inputs: Mapping[str, Any],
        context: CombineContext,
    ) -> Mapping[str, Any]:
        """DataEval's `factor_predictors` over the assessed test images, those flagged marked."""
        from dataeval.core import factor_predictors

        flagged = _flagged(inputs["ood"].value)
        if not flagged.flagged:
            reason = "No image was flagged, so no factor was compared."
            return {"output": FactorPredictorsOutput(factors={}, flagged=0, reason=reason)}
        keep = np.flatnonzero(flagged.assessed)
        collected = collect_factors(context, inputs["reference"], inputs["input"], keep=keep)
        if not collected.test:
            return {
                "output": FactorPredictorsOutput(
                    factors={}, flagged=0, unavailable=collected.unavailable, reason=_NONE_LEFT
                )
            }
        position = {int(index): at for at, index in enumerate(keep)}
        indices = [position[index] for index in flagged.flagged if index in position]
        found = factor_predictors(collected.test, indices)
        ranked = {name: round(float(value), 4) for name, value in sorted(found.items(), key=lambda item: -item[1])}
        return {
            "output": FactorPredictorsOutput(factors=ranked, flagged=len(indices), unavailable=collected.unavailable)
        }

    def section(self, record: Any) -> list[Block]:
        """Each factor and its normalized mutual information, strongest first."""
        output = record.output
        if not isinstance(output, FactorPredictorsOutput):
            return []
        table: list[Block] = (
            [ranked_table(output.factors, headers=("Factor", "MI (normalized)"))] if output.factors else []
        )
        return [*table, *_notes(output.unavailable, output.reason)]


class FactorDeviationCombine(Combine[FactorDeviationConfig]):
    """``factor-deviation``: the factors that set each of the most out-of-distribution agreed images apart from the
    reference."""

    name: ClassVar[str] = "factor-deviation"
    title: ClassVar[str] = "Factor Deviation"
    description: ClassVar[str] = "Names the factors that set the most out-of-distribution agreed images apart."
    inputs: ClassVar[tuple[Port, ...]] = _PORTS
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.OUTPUT, classes=(FactorDeviationOutput,)),)
    computed_on: ClassVar[Mapping[str, tuple[str, ...]]] = _COMPUTED_ON

    def run(
        self, config: FactorDeviationConfig, inputs: Mapping[str, Any], context: CombineContext
    ) -> Mapping[str, Any]:
        """DataEval's `factor_deviation` for the `max_items` most out-of-distribution agreed images."""
        from dataeval.core import factor_deviation

        flagged = _flagged(inputs["ood"].value)
        source = inputs["input"].address
        if not flagged.agreed:
            reason = "No image was flagged by every detector, so none is explained."
            return {"output": FactorDeviationOutput(source=source, items=[], reason=reason)}
        collected = collect_factors(context, inputs["reference"], inputs["input"])
        if not collected.test:
            output = FactorDeviationOutput(
                source=source, items=[], unavailable=collected.unavailable, reason=_NONE_LEFT
            )
            return {"output": output}
        chosen = flagged.agreed[: config.max_items]
        found = factor_deviation(collected.reference, collected.test, chosen)
        items = [
            FactorDeviation(
                index=index,
                score=flagged.scores[index],
                deviations={name: float(value) for name, value in deviations.items()},
            )
            for index, deviations in zip(chosen, found, strict=True)
        ]
        return {"output": FactorDeviationOutput(source=source, items=items, unavailable=collected.unavailable)}

    def section(self, record: Any) -> list[Block]:
        """Each explained image, by item, with its three most deviating factors. Its thumbnail is in the agreement's
        section."""
        output = record.output
        if not isinstance(output, FactorDeviationOutput):
            return []
        rows: list[dict[str, Cell]] = [
            {
                "item": item.index,
                "score": item.score or 0.0,
                "factors": ", ".join(f"{name}={value:.2f}" for name, value in list(item.deviations.items())[:3]),
            }
            for item in output.items
        ]
        columns = [
            Column(key="item", header="Item"),
            Column(key="score", header="Score", format="{:.2f}x"),
            Column(key="factors", header="Top factors", align="left"),
        ]
        table: list[Block] = [Table(columns=columns, rows=rows)] if rows else []
        return [*table, *_notes(output.unavailable, output.reason)]
