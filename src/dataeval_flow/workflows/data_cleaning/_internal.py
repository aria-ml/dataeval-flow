"""Internal helper functions shared by data cleaning and parameter sweep workflows."""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from dataeval.protocols import AnnotatedDataset
from dataeval.quality import Duplicates, DuplicatesOutput, Outliers, OutliersOutput

from dataeval_flow._cache import get_or_compute_cluster_result, get_or_compute_embeddings
from dataeval_flow.evaluators.quality._merge import merge_duplicate_outputs, merge_outlier_outputs

_logger: logging.Logger = logging.getLogger(__name__)


def _compute_embeddings(
    dataset: AnnotatedDataset[Any],
    extractor: Callable,
    run_ctx: Any,  # CleaningRunContext or shim
) -> npt.NDArray[np.float32]:
    """Compute and cache embeddings for the dataset."""
    if run_ctx is not None and getattr(run_ctx, "extractor_config", None) is not None:
        return get_or_compute_embeddings(
            dataset=dataset,
            extractor_config=run_ctx.extractor_config,
            transforms=run_ctx.transforms,
            batch_size=run_ctx.batch_size,
        )

    from dataeval.utils import flatten_samples, to_numpy

    images = [item[0] if isinstance(item, tuple) else item for item in dataset]
    embeddings = extractor(images)  # type: ignore[misc]
    return flatten_samples(to_numpy(embeddings))


def _cluster_extractor(run_ctx: Any) -> dict[str, Any]:
    """What to key a cleaning run's clusters by: the extractor that made the embeddings they cluster.

    Clusters are only valid beside the embeddings they were computed from, so they are keyed as those
    embeddings are. A stateless extractor's key names it; a stateful one's names its fit and is cached
    only under a seed; clusters over an extractor given as an object are never cached, since no key can
    name it.
    """
    extractor_config = getattr(run_ctx, "extractor_config", None)
    if extractor_config is None:
        return {}
    return {
        "extractor_config": extractor_config,
        "transforms": getattr(run_ctx, "transforms", None),
        "batch_size": getattr(run_ctx, "batch_size", None),
    }


def _merge_outlier_outputs(
    outliers_eval: Outliers,
    stats_output: OutliersOutput,
    embeddings: npt.NDArray[np.float32],
    params: Any,  # DataCleaningConfig or shim
    run_ctx: Any,  # CleaningRunContext or shim
) -> OutliersOutput:
    """Run cluster-based outlier detection and merge with stats-based results."""
    _logger.debug("Running cluster-based outlier detection")
    cluster_result = get_or_compute_cluster_result(
        embeddings,
        algorithm=params.outlier_cluster_algorithm or "hdbscan",
        n_clusters=params.outlier_n_clusters if hasattr(params, "outlier_n_clusters") else None,
        **_cluster_extractor(run_ctx),
    )
    cluster_output = outliers_eval.from_clusters(
        embeddings,
        cluster_result,
        cluster_threshold=params.outlier_cluster_threshold,
    )
    return merge_outlier_outputs(stats_output, cluster_output)


def _merge_duplicate_results(
    hash_result: DuplicatesOutput,
    embeddings: npt.NDArray[np.float32],
    params: Any,  # DataCleaningConfig or shim
    run_ctx: Any,  # CleaningRunContext or shim
) -> DuplicatesOutput:
    """Run cluster-based duplicate detection and merge with hash-based results."""
    _logger.debug("Running cluster-based duplicate detection")
    cluster_result = get_or_compute_cluster_result(
        embeddings,
        algorithm=params.duplicate_cluster_algorithm or "hdbscan",
        n_clusters=params.duplicate_n_clusters if hasattr(params, "duplicate_n_clusters") else None,
        **_cluster_extractor(run_ctx),
    )
    dup_eval = Duplicates(
        cluster_sensitivity=params.duplicate_cluster_sensitivity,
        merge_near_duplicates=params.duplicate_merge_near,
    )
    return merge_duplicate_outputs(hash_result, dup_eval.from_clusters(cluster_result))
