"""The quality evaluators: DataEval's Duplicates and Outliers, one determination each.

Both read image statistics, and in cluster mode the clusters over the task's embeddings as
well. ``find_duplicates`` and ``find_outliers`` are the only code here that calls DataEval,
and they merge cluster results through the same functions ``data-cleaning`` uses.
"""

__all__ = ["DuplicatesEvaluator", "LabelHealthEvaluator", "OutliersEvaluator", "find_duplicates", "find_outliers"]

import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, TypeVar

from dataeval.core import label_stats
from dataeval.quality import Duplicates, DuplicatesOutput, Outliers, OutliersOutput

from dataeval_flow._input_spec import InputKind
from dataeval_flow._stats import columns_for, restrict_columns
from dataeval_flow.evaluators._core import execution
from dataeval_flow.evaluators._evaluator import Evaluator
from dataeval_flow.evaluators._fields import require
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators.quality._config import DuplicatesConfig, LabelHealthConfig, OutliersConfig
from dataeval_flow.evaluators.quality._merge import merge_duplicate_outputs, merge_outlier_outputs
from dataeval_flow.evaluators.quality._result import LabelHealthOutput

_T = TypeVar("_T")

_DATAEVAL_METHODS: Mapping[InputKind, str] = {InputKind.STATS: "from_stats", InputKind.CLUSTERS: "from_clusters"}


def _require(value: _T | None, what: str, source: str) -> _T:
    if value is None:
        raise ValueError(f"Source '{source}' arrived without {what}; its producer did not run.")
    return value


def find_duplicates(params: DuplicatesConfig, inputs: Sequence[EvaluatorInputs]) -> DuplicatesOutput[Any, Any]:
    """Hash duplicates across every source; in cluster mode, merged with the embedding-space groups."""
    duplicates = params.dataeval_evaluator()
    columns = columns_for([None], params.stats_flags())
    stats = [restrict_columns(_require(i.stats, "stats", i.source), columns) for i in inputs]
    target = stats[0] if len(stats) == 1 else stats
    # `from_stats` narrows `per_target` to literals in its overloads; the values are DataEval's to check.
    found = duplicates.from_stats(target, **params.call_kwargs())  # type: ignore[call-overload]
    if params.cluster_mode():
        (source,) = inputs
        clusters = _require(source.clusters, "clusters", source.source)
        found = merge_duplicate_outputs(found, duplicates.from_clusters(clusters))
    return found


def find_outliers(params: OutliersConfig, inputs: Sequence[EvaluatorInputs]) -> OutliersOutput[Any]:
    """Statistical outliers across every source; in cluster mode, merged with the embedding-space ones."""
    outliers = params.dataeval_evaluator()
    flags = params.stats_flags()
    stats = []
    for i in inputs:
        policy = _require(i.stats_policy, "a stats policy", i.source)
        stats.append(restrict_columns(_require(i.stats, "stats", i.source), columns_for(policy.outliers_from, flags)))
    target = stats[0] if len(stats) == 1 else stats
    found = outliers.from_stats(target, **params.call_kwargs())  # type: ignore[call-overload]
    if params.cluster_mode():
        (source,) = inputs
        cluster_output = outliers.from_clusters(
            _require(source.embeddings, "embeddings", source.source),
            _require(source.clusters, "clusters", source.source),
            cluster_threshold=params.cluster_threshold,
        )
        found = merge_outlier_outputs(found, cluster_output)
    return found


class DuplicatesEvaluator(Evaluator[DuplicatesConfig, DuplicatesOutput[Any, Any]]):
    """``quality.duplicates``: which images are exact or near duplicates, per DataEval's Duplicates."""

    name: ClassVar[str] = "quality.duplicates"
    description: ClassVar[str] = "Exact and near duplicate groups (DataEval Duplicates)"
    dataeval_class: ClassVar[type] = Duplicates
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _DATAEVAL_METHODS
    output_extras: ClassVar[tuple[str, ...]] = ("annotation_divergences", "factor_cardinality")

    def run(self, config: DuplicatesConfig, inputs: Sequence[EvaluatorInputs]) -> DuplicatesOutput[Any, Any]:
        """Find duplicates in the prepared inputs."""
        return find_duplicates(config, inputs)


class OutliersEvaluator(Evaluator[OutliersConfig, OutliersOutput[Any]]):
    """``quality.outliers``: which images' statistics sit outside the threshold, per DataEval's Outliers."""

    name: ClassVar[str] = "quality.outliers"
    description: ClassVar[str] = "Images whose statistics are outliers (DataEval Outliers)"
    dataeval_class: ClassVar[type] = Outliers
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _DATAEVAL_METHODS

    def run(self, config: OutliersConfig, inputs: Sequence[EvaluatorInputs]) -> OutliersOutput[Any]:
        """Find outliers in the prepared inputs."""
        return find_outliers(config, inputs)


class LabelHealthEvaluator(Evaluator[LabelHealthConfig, LabelHealthOutput]):
    """``quality.label-health``: how a Dataset's labels spread over its classes, per DataEval's ``label_stats``."""

    name: ClassVar[str] = "quality.label-health"
    description: ClassVar[str] = "How a Dataset's labels spread over its classes (DataEval label_stats)"
    dataeval_class: ClassVar[Any] = label_stats
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.METADATA: "__call__"}

    def run(self, config: LabelHealthConfig, inputs: Sequence[EvaluatorInputs]) -> LabelHealthOutput:  # noqa: ARG002
        """Count the source's labels by class with DataEval's ``label_stats``, naming each class."""
        (source,) = inputs
        metadata = require(source.metadata, "metadata", source.source)
        index2label = {int(index): str(name) for index, name in (metadata.index2label or {}).items()}
        started, clock = datetime.now(UTC), time.monotonic()
        stats = label_stats(
            [int(label) for label in metadata.class_labels],
            [int(index) for index in metadata.item_indices],
            index2label,
            image_count=int(metadata.item_count),
        )
        meta = execution("dataeval.core.label_stats", started, time.monotonic() - clock, {})

        def named(counts: Mapping[int, int]) -> dict[str, int]:
            return {index2label.get(label, str(label)): int(count) for label, count in counts.items()}

        provenance = source.label_source
        data = {
            "item_count": int(metadata.item_count),
            "class_count": len(index2label),
            "label_count": int(stats["label_count"]),
            "label_counts_per_class": named(stats["label_counts_per_class"]),
            "image_counts_per_class": named(stats["image_counts_per_class"]),
            "empty_image_count": int(stats["empty_image_count"]),
            "label_source": provenance if provenance is None or isinstance(provenance, str) else list(provenance),
        }
        return LabelHealthOutput(data, meta)
