"""Internal helper functions shared by data cleaning and parameter sweep workflows."""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from dataeval.protocols import AnnotatedDataset
from dataeval.quality import Duplicates, DuplicatesOutput, Outliers, OutliersOutput

from dataeval_flow.cache import get_or_compute_cluster_result, get_or_compute_embeddings

logger: logging.Logger = logging.getLogger(__name__)


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

    from dataeval.utils.arrays import flatten_samples, to_numpy

    images = [item[0] if isinstance(item, tuple) else item for item in dataset]
    embeddings = extractor(images)  # type: ignore[misc]
    return flatten_samples(to_numpy(embeddings))


def _merge_outlier_outputs(
    outliers_eval: Outliers,
    stats_output: OutliersOutput,
    embeddings: npt.NDArray[np.float32],
    params: Any,  # DataCleaningParameters or shim
    _run_ctx: Any,  # CleaningRunContext or shim
) -> OutliersOutput:
    """Run cluster-based outlier detection and merge with stats-based results."""
    logger.debug("Running cluster-based outlier detection")
    cluster_result = get_or_compute_cluster_result(
        embeddings,
        algorithm=params.outlier_cluster_algorithm or "hdbscan",
        n_clusters=params.outlier_n_clusters if hasattr(params, "outlier_n_clusters") else None,
    )
    cluster_output = outliers_eval.from_clusters(
        embeddings,
        cluster_result,
        cluster_threshold=params.outlier_cluster_threshold,
    )

    # Standardize: ensure target_index column exists for merge
    column_order = ["item_index", "target_index", "metric_name", "metric_value"]
    stats_df = stats_output.data()
    cluster_df = cluster_output.data()

    dfs: list[pl.DataFrame] = []
    for df in [stats_df, cluster_df]:
        if len(df) > 0:
            if "target_index" not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias("target_index"))
            # Ensure all columns are present and in correct order
            for col in column_order:
                if col not in df.columns:
                    df = df.with_columns(pl.lit(None).alias(col))
            dfs.append(df.select(column_order))

    if not dfs:
        return OutliersOutput(pl.DataFrame(schema=dict.fromkeys(column_order, pl.Utf8)))

    merged_df = pl.concat(dfs).unique(subset=["item_index", "target_index", "metric_name"])

    # If all target_indices are null, drop the column to match stats-only output format for image-level
    if "target_index" in merged_df.columns and merged_df["target_index"].null_count() == len(merged_df):
        merged_df = merged_df.drop("target_index")

    return OutliersOutput(merged_df)


def _merge_duplicate_results(
    hash_result: DuplicatesOutput,
    embeddings: npt.NDArray[np.float32],
    params: Any,  # DataCleaningParameters or shim
    _run_ctx: Any,  # CleaningRunContext or shim
) -> DuplicatesOutput:
    """Run cluster-based duplicate detection and merge with hash-based results."""
    logger.debug("Running cluster-based duplicate detection")
    cluster_result = get_or_compute_cluster_result(
        embeddings,
        algorithm=params.duplicate_cluster_algorithm or "hdbscan",
        n_clusters=params.duplicate_n_clusters if hasattr(params, "duplicate_n_clusters") else None,
    )
    dup_eval = Duplicates(
        cluster_sensitivity=params.duplicate_cluster_sensitivity,
    )
    cluster_output = dup_eval.from_clusters(cluster_result)

    hash_df = hash_result.data()
    cluster_df = cluster_output.data()

    # If one is empty, return the other (unique handles alignment)
    if len(hash_df) == 0:
        return cluster_output
    if len(cluster_df) == 0:
        return hash_result

    # Re-number cluster group_ids to avoid collision with hash group_ids
    max_group_id = cast_to_int(hash_df["group_id"].max()) + 1 if len(hash_df) > 0 else 0
    cluster_df = cluster_df.with_columns(pl.col("group_id") + max_group_id)

    # Align columns before concat
    all_cols = list(set(hash_df.columns) | set(cluster_df.columns))
    for col in all_cols:
        if col not in hash_df.columns:
            hash_df = hash_df.with_columns(pl.lit(None).alias(col).cast(cluster_df[col].dtype))
        if col not in cluster_df.columns:
            cluster_df = cluster_df.with_columns(pl.lit(None).alias(col).cast(hash_df[col].dtype))

    merged_df = pl.concat([hash_df, cluster_df])
    return DuplicatesOutput(merged_df)


def cast_to_int(val: Any) -> int:
    """Cast a value to int, handling None."""
    try:
        return int(val) if val is not None else 0
    except (TypeError, ValueError):
        return 0
