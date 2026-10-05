"""The scope evaluators: DataEval's Representation, Coverage and Prioritize.

``run`` is the only code here that calls DataEval.
"""

__all__ = [
    "CompletenessEvaluator",
    "CoverageEvaluator",
    "LabelAlignmentEvaluator",
    "LabelReconciliationEvaluator",
    "OntologyValidationEvaluator",
    "PrioritizationEvaluator",
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
from dataeval.core import RankResult, completeness, label_alignment, label_reconciliation, ontology_validation
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
    CompletenessConfig,
    CoverageConfig,
    LabelAlignmentConfig,
    LabelReconciliationConfig,
    OntologyValidationConfig,
    PrioritizationConfig,
    RepresentationConfig,
)
from dataeval_flow.evaluators.scope._result import (
    CompletenessOutput,
    LabelReconciliationOutput,
    OntologyValidationOutput,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

_logger: logging.Logger = logging.getLogger(__name__)


class RepresentationEvaluator(Evaluator[RepresentationConfig, RepresentationOutput]):
    """``representation``: which of an ontology's classes fall short of their share, per DataEval's
    Representation."""

    name: ClassVar[str] = "representation"
    title: ClassVar[str] = "Representation"
    description: ClassVar[str] = "Class counts against an ontology's leaves (DataEval Representation)."
    dataeval_class: ClassVar[type] = Representation
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.LABELS: "evaluate"}
    output_extras: ClassVar[tuple[str, ...]] = (
        "leaf_coverage",
        "total_deficit",
        "violations",
        "dark_branches",
        "ignored_expected",
    )
    reads_factors: ClassVar[bool] = False

    def run(self, config: RepresentationConfig, inputs: Sequence[EvaluatorInputs]) -> RepresentationOutput:
        """Count the source's labels against the task's ontology, or one synthesized from its ``index2label``.

        The output also records the ``expected`` names that resolve to no concept or to several, which
        Representation drops with only a log warning, and how the ontology was named (coverage spec §3.3).
        """
        from dataeval_flow.workflows._ontology import synthesize_ontology

        (source,) = inputs
        labels = require(source.labels, "labels", source.source)
        if len(labels) == 0:
            raise ValueError(
                f"Source '{source.source}' has no labels: its dataset's targets carry none, and `representation` "
                "counts class labels."
            )
        index2label = dict(source.index2label or {})
        if source.ontology is not None:
            ontology, named = source.ontology, source.ontology_source
        else:
            ontology, named = synthesize_ontology(index2label)
        output = Representation(ontology, **dataeval_arguments(config)).evaluate(labels, index2label=index2label)
        output.ignored_expected = sorted(  # pyright: ignore[reportAttributeAccessIssue]
            name for name in (config.expected or {}) if len(ontology.find(name)) != 1
        )
        output.ontology_source = named  # pyright: ignore[reportAttributeAccessIssue]
        return output


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
    title: ClassVar[str] = "Coverage"
    description: ClassVar[str] = "Embedding-space coverage, broken down by class (DataEval Coverage)."
    dataeval_class: ClassVar[type] = Coverage
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {
        InputKind.EMBEDDINGS: "evaluate",
        InputKind.LABELS: "evaluate",
    }
    output_extras: ClassVar[tuple[str, ...]] = (
        "uncovered_indices",
        "coverage_radius",
        "critical_value_radii",
        "uncovered_classes",
    )
    reads_factors: ClassVar[bool] = False

    def run(self, config: CoverageConfig, inputs: Sequence[EvaluatorInputs]) -> CoverageOutput:
        """Measure the source's coverage, broken down by class where its labels allow, else as one class, ``0``, and
        name each uncovered item's class (coverage spec §5.3)."""
        (source,) = inputs
        embeddings = require(source.embeddings, "embeddings", source.source)
        if len(embeddings) == 0:
            raise ValueError(f"`coverage` has no items to embed; the source has {len(embeddings)}.")
        labels = usable_labels(source, len(embeddings), self.name)
        names = dict(source.index2label or {})
        classes = (
            _Labels(labels, names) if labels is not None else _Labels(np.zeros(len(embeddings), dtype=np.intp), {})
        )
        output = Coverage(**dataeval_arguments(config)).evaluate(classes, embeddings=embeddings)
        output.uncovered_classes = [  # pyright: ignore[reportAttributeAccessIssue]
            None if labels is None else names.get(int(labels[index]), str(int(labels[index])))
            for index in output.uncovered_indices
        ]
        return output


class PrioritizationEvaluator(Evaluator[PrioritizationConfig, PrioritizeOutput]):
    """``prioritization``: the first source's items ranked by difficulty, per DataEval's Prioritize."""

    name: ClassVar[str] = "prioritization"
    title: ClassVar[str] = "Prioritization"
    description: ClassVar[str] = "Items ranked by difficulty, optionally against a reference (DataEval Prioritize)."
    dataeval_class: ClassVar[type] = Prioritize
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {
        InputKind.EMBEDDINGS: "evaluate",
        InputKind.LABELS: "evaluate",
    }
    output_extras: ClassVar[tuple[str, ...]] = ("scores",)
    reads_factors: ClassVar[bool] = False

    def run(self, config: PrioritizationConfig, inputs: Sequence[EvaluatorInputs]) -> PrioritizeOutput:
        """Rank the first source's items, relative to the second source's where the task names one."""
        data, *rest = inputs
        embeddings = require(data.embeddings, "embeddings", data.source)
        reference = require(rest[0].embeddings, "embeddings", rest[0].source) if rest else None
        if reference is not None and not len(reference):
            raise ValueError(f"`{rest[0].source}` has no items to rank against.")
        prioritize = Prioritize(**dataeval_arguments(config), reference=reference)
        if not len(embeddings):
            # DataEval refuses no embeddings; a pool emptied by cleaning ranks to an empty ranking.
            _logger.warning("%s: source '%s' has no items, so its ranking is empty.", self.name, data.source)
            empty = RankResult(indices=np.empty(0, dtype=np.intp), scores=np.empty(0, dtype=np.float32))
            return PrioritizeOutput(empty, prioritize.method, prioritize.order, prioritize.policy, prioritize.num_bins)
        labels = usable_labels(data, len(embeddings), self.name)
        return prioritize.evaluate(embeddings, class_labels=labels)


class LabelAlignmentEvaluator(Evaluator[LabelAlignmentConfig, LabelAlignmentOutput]):
    """``label-alignment``: how the Dataset's class names align to an ontology, per DataEval's label_alignment."""

    name: ClassVar[str] = "label-alignment"
    title: ClassVar[str] = "Label Alignment"
    description: ClassVar[str] = (
        "How a Dataset's class names align to an ontology: the remap, and whether it is lossless."
    )
    dataeval_class: ClassVar[Any] = label_alignment
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.LABELS: "__call__"}
    reads_factors: ClassVar[bool] = False

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


class CompletenessEvaluator(Evaluator[CompletenessConfig, CompletenessOutput]):
    """``completeness``: dimensional completeness of the embeddings, per DataEval's completeness."""

    name: ClassVar[str] = "completeness"
    title: ClassVar[str] = "Completeness"
    description: ClassVar[str] = "How much of the embedding space's dimensions the data fills (DataEval completeness)."
    dataeval_class: ClassVar[Any] = completeness
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.EMBEDDINGS: "__call__"}
    reads_factors: ClassVar[bool] = False

    def run(self, config: CompletenessConfig, inputs: Sequence[EvaluatorInputs]) -> CompletenessOutput:  # noqa: ARG002
        """Score the source's embeddings, rescaled to the unit interval per dimension, constant dimensions at 0
        (coverage spec §6.1)."""
        from dataeval_flow.workflows._common import normalize_unit_interval

        (source,) = inputs
        embeddings = np.asarray(require(source.embeddings, "embeddings", source.source))
        if len(embeddings) < 2:
            raise ValueError(f"`completeness` needs at least two embeddings; the source has {len(embeddings)}.")
        started, clock = datetime.now(UTC), time.monotonic()
        result = completeness(normalize_unit_interval(embeddings))
        meta = execution("dataeval.core.completeness", started, time.monotonic() - clock, {})
        pairs = [[int(a), int(b)] for a, b in result.get("nearest_neighbor_pairs", [])]
        return CompletenessOutput(
            {"completeness": float(result["completeness"]), "nearest_neighbor_pairs": pairs}, meta
        )


class LabelReconciliationEvaluator(Evaluator[LabelReconciliationConfig, LabelReconciliationOutput]):
    """``label-reconciliation``: which class names resolve to one ontology concept, per DataEval's
    label_reconciliation."""

    name: ClassVar[str] = "label-reconciliation"
    title: ClassVar[str] = "Label Reconciliation"
    description: ClassVar[str] = "Which of a Dataset's class names resolve to exactly one ontology concept."
    dataeval_class: ClassVar[Any] = label_reconciliation
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.LABELS: "__call__"}
    reads_factors: ClassVar[bool] = False

    def run(self, config: LabelReconciliationConfig, inputs: Sequence[EvaluatorInputs]) -> LabelReconciliationOutput:  # noqa: ARG002
        """Reconcile the source's class names, in index order, against the task's ontology."""
        (source,) = inputs
        if source.ontology is None:
            raise ValueError("`label-reconciliation` needs its `ontology:` to load.")
        index2label = dict(source.index2label or {})
        names = [index2label[index] for index in sorted(index2label)]
        started, clock = datetime.now(UTC), time.monotonic()
        result = label_reconciliation(names, source.ontology)
        meta = execution("dataeval.core.label_reconciliation", started, time.monotonic() - clock, {})
        unmatched = list(result["unmatched"])
        ambiguous = {name: list(ids) for name, ids in result["ambiguous"].items()}
        data = {
            "conforms": not unmatched and not ambiguous,
            "matched": dict(result["matched"]),
            "unmatched": unmatched,
            "ambiguous": ambiguous,
        }
        return LabelReconciliationOutput(data, meta)


class OntologyValidationEvaluator(Evaluator[OntologyValidationConfig, OntologyValidationOutput]):
    """``ontology-validation``: an ontology's structural and naming facts, per DataEval's ontology_validation."""

    name: ClassVar[str] = "ontology-validation"
    title: ClassVar[str] = "Ontology Validation"
    description: ClassVar[str] = "An ontology's structural and naming facts: depth, roots, collisions, and more."
    dataeval_class: ClassVar[Any] = ontology_validation
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.LABELS: "__call__"}
    reads_factors: ClassVar[bool] = False

    def run(self, config: OntologyValidationConfig, inputs: Sequence[EvaluatorInputs]) -> OntologyValidationOutput:
        """Validate the task's ontology; the source is read only for its ontology."""
        (source,) = inputs
        ontology = source.ontology
        if ontology is None:
            raise ValueError("`ontology-validation` needs its `ontology:` to load.")
        started, clock = datetime.now(UTC), time.monotonic()
        result = ontology_validation(ontology, label_pattern=config.label_pattern)
        meta = execution(
            "dataeval.core.ontology_validation",
            started,
            time.monotonic() - clock,
            {"label_pattern": config.label_pattern},
        )
        depths = result["depth"]
        data = {
            "concept_count": len(ontology.ids),
            "leaf_count": len(result["leaves"]),
            "max_depth": max(depths.values()) if depths else 0,
            "roots": list(result["roots"]),
            "isolated": list(result["isolated"]),
            "external_ancestors": {cid: list(ids) for cid, ids in result["external_ancestors"].items()},
            # DataEval returns tuples; JSON has none, so each pair is a two-element list.
            "redundant_edges": [list(edge) for edge in result["redundant_edges"]],
            "ancestor_siblings": [list(pair) for pair in result["ancestor_siblings"]],
            "unary_parents": list(result["unary_parents"]),
            "label_collisions": {name: list(ids) for name, ids in result["label_collisions"].items()},
            "nonconforming_labels": dict(result["nonconforming_labels"]),
        }
        return OntologyValidationOutput(data, meta)
