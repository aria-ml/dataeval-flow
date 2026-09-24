"""Merging cluster-mode results into the statistics-mode ones.

Shared by ``data-cleaning`` and the quality evaluators, so the two cannot disagree on the
same data.
"""

__all__ = ["cast_to_int", "merge_duplicate_outputs", "merge_outlier_outputs"]

from typing import Any

import polars as pl
from dataeval.quality import DuplicatesOutput, OutliersOutput


def merge_outlier_outputs(stats_output: OutliersOutput, cluster_output: OutliersOutput) -> OutliersOutput:
    """Union the statistical and cluster outliers, one row per item, target and metric.

    Keeps every column either side reports (stats-mode's ``direction``, ``bound``,
    ``percentile``, ``population_mean`` and ``population_std`` included), filling a row
    with null on the side that does not report a given column. Deduplication keeps the
    first matching row and preserves input order, so merging the same inputs twice gives
    identical frames.
    """
    base_columns = ["item_index", "target_index", "metric_name", "metric_value"]
    stats_df = stats_output.data()
    cluster_df = cluster_output.data()

    dfs: list[pl.DataFrame] = []
    for df in [stats_df, cluster_df]:
        if len(df) > 0:
            if "target_index" not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias("target_index"))
            dfs.append(df)

    if not dfs:
        return OutliersOutput(pl.DataFrame(schema=dict.fromkeys(base_columns, pl.Utf8)))

    merged_df = pl.concat(dfs, how="diagonal_relaxed").unique(
        subset=["item_index", "target_index", "metric_name"],
        keep="first",
        maintain_order=True,
    )

    # If all target_indices are null, drop the column to match stats-only output format for image-level
    if "target_index" in merged_df.columns and merged_df["target_index"].null_count() == len(merged_df):
        merged_df = merged_df.drop("target_index")

    return OutliersOutput(merged_df)


def merge_duplicate_outputs(hash_result: DuplicatesOutput, cluster_output: DuplicatesOutput) -> DuplicatesOutput:
    """Concatenate the hash and cluster groups, renumbering the cluster groups past the hash ones."""
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
