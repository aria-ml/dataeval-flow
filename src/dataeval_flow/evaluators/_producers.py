"""One producer per input kind: the only place an evaluator touches the cache, views and policies."""

__all__ = ["PRODUCERS", "Producer", "ProducerContext"]

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np

from dataeval_flow._input_spec import InputKind
from dataeval_flow.evaluators._base import EvaluatorConfig

if TYPE_CHECKING:
    from dataeval.protocols import AnnotatedDataset

    from dataeval_flow._stats import ResolvedStatsPolicy
    from dataeval_flow.workflows._context import DatasetContext, WorkflowContext


@runtime_checkable
class _ClusterConsumer(Protocol):
    cluster_algorithm: Literal["kmeans", "hdbscan"] | None
    n_clusters: int | None


@dataclass(frozen=True)
class ProducerContext:
    """What a producer reads: one source's dataset after its view, and the run it belongs to."""

    source: str
    """The source's name, as the task names it."""
    dataset: "AnnotatedDataset[Any]"
    dataset_context: "DatasetContext"
    workflow_context: "WorkflowContext"
    config: "EvaluatorConfig[Any]"
    stats_union: "ResolvedStatsPolicy | None" = None
    """What a chain planned to compute for this source, so that one pass serves every evaluator step reading it: the
    union of those steps' stats requests that can share this one's cache entry. ``None`` where nothing is planned."""


Producer = Callable[[ProducerContext], dict[str, Any]]


def produce_stats(pc: ProducerContext) -> dict[str, Any]:
    """Image statistics under the resolved stats policy, computed exactly as ``data-cleaning`` computes them.

    Always per image and per target, so the two share one cache entry. The evaluator's own
    ``per_image`` / ``per_target`` apply when it calls ``from_stats``.

    Under a cache, where a chain planned a :attr:`~ProducerContext.stats_union` for the source, that union is
    requested instead: the first step to ask computes it in one pass, and every later one reads it from the cache. The
    evaluator still gets its own policy, and keeps only its own columns. Without a cache every request is computed in
    full, so asking for the union would compute it again for every step: each step asks for its own.

    The first step to read a node computes the families the node's other readers need as well, so an error computing
    one of those families surfaces on that first step, not on the step that reads it.
    """
    from dataeval_flow._cache import caching_active, get_or_compute_stats
    from dataeval_flow._stats import stats_policy_for

    policy = stats_policy_for(pc.workflow_context, **pc.config.stats_request())
    request = pc.stats_union if pc.stats_union is not None and caching_active() else policy
    stats = get_or_compute_stats(request, dataset=pc.dataset, value_range=pc.dataset_context.value_range)
    return {"stats": stats, "stats_policy": policy}


def produce_clusters(pc: ProducerContext) -> dict[str, Any]:
    """Embeddings from the task's extractor, then clusters over them. Sets both."""
    from dataeval.types import ClusterConfigMixin

    from dataeval_flow._cache import get_or_compute_cluster_result, get_or_compute_embeddings
    from dataeval_flow._embeddings import node_embeddings

    config = pc.config
    if not isinstance(config, _ClusterConsumer):
        raise TypeError(f"{type(config).__name__} consumes clusters but declares no clustering fields")
    dc = pc.dataset_context
    if dc.extractor is None:
        raise ValueError("Cluster mode needs an extractor on the task.")
    embeddings = node_embeddings(
        dc, lambda: get_or_compute_embeddings(pc.dataset, dc.extractor, dc.transforms, dc.batch_size)
    )
    algorithm = config.cluster_algorithm or ClusterConfigMixin.model_fields["cluster_algorithm"].default
    clusters = get_or_compute_cluster_result(
        embeddings,
        algorithm=algorithm,
        n_clusters=config.n_clusters,
        extractor_config=dc.extractor,
        transforms=dc.transforms,
        batch_size=dc.batch_size,
    )
    return {"embeddings": embeddings, "clusters": clusters}


def produce_metadata(pc: ProducerContext) -> dict[str, Any]:
    """The source's metadata under the task's metadata policy, read as every workflow reads it. Sets both."""
    context = pc.workflow_context
    return {"metadata": context.metadata(pc.source), "metadata_policy": context.metadata_policy}


def produce_labels(pc: ProducerContext) -> dict[str, Any]:
    """The source's class labels and their names, from one read of its metadata.

    ``WorkflowContext.labels`` reads the same metadata, so reading it once here keeps an uncached run to one walk.
    """
    metadata = pc.workflow_context.metadata(pc.source)
    return {
        "labels": np.asarray(metadata.class_labels, dtype=np.intp),
        "index2label": dict(metadata.index2label or {}),
    }


def produce_embeddings(pc: ProducerContext) -> dict[str, Any]:
    """The task's extractor over the source, cached, with one fitted stateful extractor for the whole task. An
    extractor that runs a model gives each row's entropy instead, with the predictions the rows came from."""
    from dataeval_flow._predictions import runs_model, uncertainty_rows

    if runs_model(pc.dataset_context.extractor):
        predictions = pc.workflow_context.predictions(pc.source)
        return {"embeddings": uncertainty_rows(predictions), "predictions": predictions}
    return {"embeddings": np.asarray(pc.workflow_context.embeddings(pc.source))}


# The producer for each input kind.
PRODUCERS: Mapping[InputKind, Producer] = {
    InputKind.STATS: produce_stats,
    InputKind.CLUSTERS: produce_clusters,
    InputKind.METADATA: produce_metadata,
    InputKind.LABELS: produce_labels,
    InputKind.EMBEDDINGS: produce_embeddings,
}
