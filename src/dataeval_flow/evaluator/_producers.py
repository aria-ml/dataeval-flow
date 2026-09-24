"""One producer per input kind: the only place an evaluator touches the cache, views and policies."""

__all__ = ["PRODUCERS", "Producer", "ProducerContext"]

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from dataeval_flow.evaluator.base import EvaluatorParametersBase, InputKind

if TYPE_CHECKING:
    from dataeval.protocols import AnnotatedDataset

    from dataeval_flow.workflow import DatasetContext, WorkflowContext


@runtime_checkable
class _StatsConsumer(Protocol):
    def stats_request(self) -> dict[str, Any]: ...


@runtime_checkable
class _ClusterConsumer(Protocol):
    cluster_algorithm: Literal["kmeans", "hdbscan"] | None
    n_clusters: int | None


@dataclass(frozen=True)
class ProducerContext:
    """What a producer reads: one source's dataset after its view, and the run it belongs to."""

    dataset: "AnnotatedDataset[Any]"
    dataset_context: "DatasetContext"
    workflow_context: "WorkflowContext"
    params: EvaluatorParametersBase


Producer = Callable[[ProducerContext], dict[str, Any]]


def produce_stats(pc: ProducerContext) -> dict[str, Any]:
    """Image statistics under the resolved stats policy, computed exactly as ``data-cleaning`` computes them.

    Always per image and per target, so the two share one cache entry. The evaluator's own
    ``per_image`` / ``per_target`` apply when it calls ``from_stats``.
    """
    from dataeval_flow.cache import get_or_compute_stats
    from dataeval_flow.stats import stats_policy_for

    if not isinstance(pc.params, _StatsConsumer):
        raise TypeError(f"{type(pc.params).__name__} consumes stats but declares no stats_request()")
    policy = stats_policy_for(pc.workflow_context, **pc.params.stats_request())
    stats = get_or_compute_stats(policy, dataset=pc.dataset, value_range=pc.dataset_context.value_range)
    return {"stats": stats, "stats_policy": policy}


def produce_clusters(pc: ProducerContext) -> dict[str, Any]:
    """Embeddings from the task's extractor, then clusters over them. Sets both."""
    from dataeval.types import ClusterConfigMixin

    from dataeval_flow.cache import get_or_compute_cluster_result, get_or_compute_embeddings

    params = pc.params
    if not isinstance(params, _ClusterConsumer):
        raise TypeError(f"{type(params).__name__} consumes clusters but declares no clustering fields")
    dc = pc.dataset_context
    if dc.extractor is None:
        raise ValueError("Cluster mode needs an extractor on the task.")
    embeddings = get_or_compute_embeddings(pc.dataset, dc.extractor, dc.transforms, dc.batch_size)
    algorithm = params.cluster_algorithm or ClusterConfigMixin.model_fields["cluster_algorithm"].default
    clusters = get_or_compute_cluster_result(
        embeddings,
        algorithm=algorithm,
        n_clusters=params.n_clusters,
        extractor_config=dc.extractor,
        transforms=dc.transforms,
    )
    return {"embeddings": embeddings, "clusters": clusters}


#: The producer for each kind this build can produce. Later phases add metadata, labels and embeddings.
PRODUCERS: Mapping[InputKind, Producer] = {
    InputKind.STATS: produce_stats,
    InputKind.CLUSTERS: produce_clusters,
}
