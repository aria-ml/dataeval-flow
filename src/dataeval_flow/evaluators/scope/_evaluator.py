"""The scope evaluators: DataEval's Representation, Coverage and Prioritize.

``run`` is the only code here that calls DataEval.
"""

__all__ = [
    "CoverageEvaluator",
    "LabelAlignmentEvaluator",
    "PrioritizeEvaluator",
    "RepresentationEvaluator",
    "usable_labels",
]

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from dataeval.core import label_alignment
from dataeval.scope import (
    Coverage,
    CoverageOutput,
    Prioritize,
    PrioritizeOutput,
    Representation,
    RepresentationOutput,
)

from dataeval_flow._alignment import LabelAlignmentOutput, align_labels
from dataeval_flow._input_spec import InputKind
from dataeval_flow.evaluators._core import execution
from dataeval_flow.evaluators._evaluator import Evaluator
from dataeval_flow.evaluators._fields import dataeval_arguments, require
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators.scope._config import (
    CoverageConfig,
    LabelAlignmentConfig,
    PrioritizeConfig,
    RepresentationConfig,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_logger: logging.Logger = logging.getLogger(__name__)


class RepresentationEvaluator(Evaluator[RepresentationConfig, RepresentationOutput]):
    """``representation``: which of an ontology's classes fall short of their share, per DataEval's
    Representation."""

    name: ClassVar[str] = "representation"
    description: ClassVar[str] = "Class counts against an ontology's leaves (DataEval Representation)"
    dataeval_class: ClassVar[type] = Representation
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.LABELS: "evaluate"}
    output_extras: ClassVar[tuple[str, ...]] = ("leaf_coverage", "total_deficit", "violations", "dark_branches")

    def run(self, config: RepresentationConfig, inputs: Sequence[EvaluatorInputs]) -> RepresentationOutput:
        """Count the source's labels against the task's ontology, or one synthesized from its ``index2label``."""
        from dataeval_flow.workflows._ontology import synthesize_ontology

        (source,) = inputs
        labels = require(source.labels, "labels", source.source)
        if len(labels) == 0:
            raise ValueError(
                f"Source '{source.source}' has no labels: its dataset's targets carry none, and `representation` "
                "counts class labels."
            )
        index2label = dict(source.index2label or {})
        ontology = source.ontology if source.ontology is not None else synthesize_ontology(index2label)[0]
        return Representation(ontology, **dataeval_arguments(config)).evaluate(labels, index2label=index2label)


@dataclass(frozen=True)
class _Labels:
    """Class labels and their names: what DataEval's ``LabelsLike`` readers take from a dataset."""

    class_labels: "NDArray[np.intp]"
    index2label: Mapping[int, str]


def usable_labels(source: EvaluatorInputs, count: int, evaluator: str) -> "NDArray[np.intp] | None":
    """The source's labels where it has one per item; otherwise ``None``, after a warning that says why.

    A detection dataset has one label per target rather than per item, so its labels do not line up with one
    embedding per item: it runs as a dataset without labels does.
    """
    labels = require(source.labels, "labels", source.source)
    if len(labels) == count and count > 0:
        return labels
    reason = (
        "has no labels"
        if len(labels) == 0
        else f"has {len(labels)} labels for {count} items, one per target rather than one per item"
    )
    _logger.warning("%s: source '%s' %s, so it runs without a class breakdown.", evaluator, source.source, reason)
    return None


class CoverageEvaluator(Evaluator[CoverageConfig, CoverageOutput]):
    """``coverage``: which items the rest of the data does not cover, per DataEval's Coverage."""

    name: ClassVar[str] = "coverage"
    description: ClassVar[str] = "Embedding-space coverage, broken down by class (DataEval Coverage)"
    dataeval_class: ClassVar[type] = Coverage
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {
        InputKind.EMBEDDINGS: "evaluate",
        InputKind.LABELS: "evaluate",
    }
    output_extras: ClassVar[tuple[str, ...]] = ("uncovered_indices", "coverage_radius", "critical_value_radii")

    def run(self, config: CoverageConfig, inputs: Sequence[EvaluatorInputs]) -> CoverageOutput:
        """Measure the source's coverage, broken down by class where its labels allow, else as one class, ``0``."""
        (source,) = inputs
        embeddings = require(source.embeddings, "embeddings", source.source)
        labels = usable_labels(source, len(embeddings), self.name)
        classes = (
            _Labels(labels, dict(source.index2label or {}))
            if labels is not None
            else _Labels(np.zeros(len(embeddings), dtype=np.intp), {})
        )
        return Coverage(**dataeval_arguments(config)).evaluate(classes, embeddings=embeddings)


class PrioritizeEvaluator(Evaluator[PrioritizeConfig, PrioritizeOutput]):
    """``prioritize``: the first source's items ranked by difficulty, per DataEval's Prioritize."""

    name: ClassVar[str] = "prioritize"
    description: ClassVar[str] = "Items ranked by difficulty, optionally against a reference (DataEval Prioritize)"
    dataeval_class: ClassVar[type] = Prioritize
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {
        InputKind.EMBEDDINGS: "evaluate",
        InputKind.LABELS: "evaluate",
    }
    output_extras: ClassVar[tuple[str, ...]] = ("scores",)

    def run(self, config: PrioritizeConfig, inputs: Sequence[EvaluatorInputs]) -> PrioritizeOutput:
        """Rank the first source's items, relative to the second source's where the task names one."""
        data, *rest = inputs
        embeddings = require(data.embeddings, "embeddings", data.source)
        reference = require(rest[0].embeddings, "embeddings", rest[0].source) if rest else None
        labels = usable_labels(data, len(embeddings), self.name)
        prioritize = Prioritize(**dataeval_arguments(config), reference=reference)
        return prioritize.evaluate(embeddings, class_labels=labels)


class LabelAlignmentEvaluator(Evaluator[LabelAlignmentConfig, LabelAlignmentOutput]):
    """``label-alignment``: how the Dataset's class names align to an ontology, per DataEval's label_alignment."""

    name: ClassVar[str] = "label-alignment"
    description: ClassVar[str] = (
        "How a Dataset's class names align to an ontology: the remap, and whether it is lossless."
    )
    dataeval_class: ClassVar[Any] = label_alignment
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.LABELS: "__call__"}

    def run(self, config: LabelAlignmentConfig, inputs: Sequence[EvaluatorInputs]) -> LabelAlignmentOutput:
        """Align the source's class names to the task's ontology with DataEval's ``label_alignment``."""
        (source,) = inputs
        if source.ontology is None:
            raise ValueError("`label-alignment` needs its `ontology:` to load.")
        index2label = dict(source.index2label or {})
        names = [index2label[index] for index in sorted(index2label)]
        started, clock = datetime.now(UTC), time.monotonic()
        alignment = align_labels(source.ontology, names, threshold=config.threshold)
        meta = execution(
            "dataeval.core.label_alignment", started, time.monotonic() - clock, {"threshold": config.threshold}
        )
        return LabelAlignmentOutput(alignment, source.ontology, meta, ontology_source=source.ontology_source)
