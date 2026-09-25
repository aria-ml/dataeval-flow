"""The quality evaluators: DataEval's Duplicates and Outliers, one determination each.

Both read image statistics, and in cluster mode the clusters over the task's embeddings as
well. ``find_duplicates`` and ``find_outliers`` are the only code here that calls DataEval,
and they merge cluster results through the same functions ``data-cleaning`` uses.
"""

__all__ = ["DuplicatesEvaluator", "OutliersEvaluator", "find_duplicates", "find_outliers"]

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, TypeVar

from dataeval.quality import Duplicates, DuplicatesOutput, Outliers, OutliersOutput

from dataeval_flow._input_spec import InputKind
from dataeval_flow._stats import columns_for, restrict_columns
from dataeval_flow.evaluators._evaluator import Evaluator
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators.quality._config import DuplicatesConfig, OutliersConfig
from dataeval_flow.evaluators.quality._merge import merge_duplicate_outputs, merge_outlier_outputs

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
