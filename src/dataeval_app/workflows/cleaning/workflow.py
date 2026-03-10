"""Data Cleaning Workflow — orchestration + processor + factory helpers."""

__all__ = ["DataCleaningWorkflow"]

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl
from dataeval import Metadata
from dataeval.flags import ImageStats
from dataeval.protocols import AnnotatedDataset
from dataeval.quality import Duplicates, DuplicatesOutput, Outliers
from pydantic import BaseModel

from dataeval_app.cache import WorkflowCache, get_or_compute_metadata
from dataeval_app.embeddings import build_extractor
from dataeval_app.workflow import WorkflowContext, WorkflowProtocol, WorkflowResult
from dataeval_app.workflow.base import Reportable
from dataeval_app.workflows.cleaning.outputs import (
    ClasswisePivotDict,
    ClasswiseRowDict,
    DataCleaningMetadata,
    DataCleaningOutputs,
    DataCleaningRawOutputs,
    DataCleaningReport,
    DetectionDict,
    DuplicatesDict,
    IndexValue,
    LabelStatsDict,
    OutlierIssuesDict,
    SourceIndexDict,
)
from dataeval_app.workflows.cleaning.params import DataCleaningParameters

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols for private DataEval types
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

FLAG_MAP: dict[str, ImageStats] = {
    "dimension": ImageStats.DIMENSION,
    "pixel": ImageStats.PIXEL,
    "visual": ImageStats.VISUAL,
}

HASH_FLAG_MAP: dict[str, ImageStats] = {
    "hash_basic": ImageStats.HASH_DUPLICATES_BASIC,
    "hash_d4": ImageStats.HASH_DUPLICATES_D4,
}


def _build_outliers(
    params: DataCleaningParameters,
    extractor: Callable | None = None,
) -> Outliers:
    """Build Outliers evaluator from cleaning parameters."""
    flags = ImageStats.NONE
    for name in params.outlier_flags:
        flags |= FLAG_MAP[name]

    # Validate cluster params require an extractor
    has_cluster = (
        params.outlier_cluster_threshold is not None
        or params.outlier_cluster_algorithm is not None
        or params.outlier_n_clusters is not None
    )
    if has_cluster and extractor is None:
        raise ValueError(
            "Cluster-based outlier detection requires an extractor. "
            "Configure a model/extractor or remove cluster params."
        )

    return Outliers(
        flags=flags,
        outlier_threshold=(params.outlier_method, params.outlier_threshold),
        cluster_threshold=params.outlier_cluster_threshold,
        cluster_algorithm=params.outlier_cluster_algorithm,
        n_clusters=params.outlier_n_clusters,
        extractor=extractor,
    )


def _build_duplicates(
    params: DataCleaningParameters,
    extractor: Callable | None = None,
    batch_size: int | None = None,
) -> Duplicates:
    """Build Duplicates evaluator from cleaning parameters."""
    # Build hash flags
    flags = ImageStats.NONE
    if params.duplicate_flags is not None:
        for name in params.duplicate_flags:
            flags |= HASH_FLAG_MAP[name]

    # Validate cluster params require an extractor
    has_cluster = (
        params.duplicate_cluster_sensitivity is not None
        or params.duplicate_cluster_algorithm is not None
        or params.duplicate_n_clusters is not None
    )
    if has_cluster and extractor is None:
        raise ValueError(
            "Cluster-based duplicate detection requires an extractor. "
            "Configure a model/extractor or remove cluster params."
        )

    # Pass flags only if explicitly configured; otherwise let DataEval use its default.
    kwargs: dict[str, object] = {
        "merge_near_duplicates": params.duplicate_merge_near,
        "cluster_sensitivity": params.duplicate_cluster_sensitivity,
        "cluster_algorithm": params.duplicate_cluster_algorithm,
        "n_clusters": params.duplicate_n_clusters,
        "extractor": extractor,
        "batch_size": batch_size,
    }
    if params.duplicate_flags is not None:
        kwargs["flags"] = flags

    return Duplicates(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Result serialization helpers
# ---------------------------------------------------------------------------


def _serialize_outlier_issues(issues: "pl.DataFrame") -> "OutlierIssuesDict":
    """Serialize outlier issues Polars DataFrame to plain dict.

    Parameters
    ----------
    issues : polars.DataFrame
        Polars DataFrame with columns: item_index, metric_name, metric_value,
        and optionally target_index.
    """
    return {
        # to_dicts() returns list[dict[str, Any]]; rows match OutlierIssueRecord shape at runtime.
        "issues": issues.to_dicts(),  # type: ignore[typeddict-item]
        "count": len(issues),
    }


def _serialize_duplicates(result: "DuplicatesOutput") -> "DuplicatesDict":
    """Serialize DuplicatesOutput to plain dict from its DataFrame.

    DuplicatesOutput wraps a DataFrame with columns: group_id, level,
    dup_type, item_indices, target_indices, methods, orientation.
    """

    def _indices_from_row(row: dict[str, Any]) -> list[IndexValue]:
        """Build index list from a DataFrame row, using SourceIndexDict for targets."""
        items = row["item_indices"]
        targets = row.get("target_indices")
        if targets is not None:
            return [SourceIndexDict(item=i, target=t, channel=None) for i, t in zip(items, targets, strict=True)]
        return [int(i) for i in items]

    def _detection_from_df(df: "pl.DataFrame") -> "DetectionDict":
        out: DetectionDict = {}
        exact_df = df.filter(pl.col("dup_type") == "exact")
        if len(exact_df) > 0:
            out["exact"] = [_indices_from_row(row) for row in exact_df.iter_rows(named=True)]

        near_df = df.filter(pl.col("dup_type") == "near")
        if len(near_df) > 0:
            out["near"] = [
                {
                    "indices": _indices_from_row(row),
                    "methods": sorted(row["methods"]),
                    "orientation": row.get("orientation"),
                }
                for row in near_df.iter_rows(named=True)
            ]
        return out

    items_df = result.data().filter(pl.col("level") == "item")
    targets_df = result.data().filter(pl.col("level") == "target")
    return {
        "items": _detection_from_df(items_df),
        "targets": _detection_from_df(targets_df),
    }


def _compute_label_stats(metadata: Metadata) -> "LabelStatsDict":
    """Compute label statistics from Metadata instance."""
    _, _, label_counts = _build_class_labels_df(metadata)

    return {
        "item_count": metadata.item_count,
        "class_count": len(metadata.index2label),
        "index2label": dict(metadata.index2label),
        "label_counts_per_class": label_counts,
    }


# ---------------------------------------------------------------------------
# Findings generation
# ---------------------------------------------------------------------------


def _duplicate_finding(raw: DataCleaningRawOutputs) -> Reportable | None:
    """Build a Duplicates finding from raw results, or None if no duplicates."""
    exact_groups = raw.duplicates.get("items", {}).get("exact", [])
    near_groups = raw.duplicates.get("items", {}).get("near", [])
    if not exact_groups and not near_groups:
        return None

    exact_affected = sum(len(g) for g in exact_groups)
    near_affected = sum(len(g["indices"]) for g in near_groups)
    # Collect methods and orientations from near groups
    all_methods: set[str] = set()
    orientations: dict[str, int] = {}
    for g in near_groups:
        all_methods.update(g.get("methods", []))
        orient = g.get("orientation")
        if orient is not None:
            orientations[orient] = orientations.get(orient, 0) + 1
    detail_lines: list[str] = []
    if exact_groups:
        detail_lines.append(f"{len(exact_groups)} exact-duplicate groups ({exact_affected} images)")
    if near_groups:
        detail_lines.append(f"{len(near_groups)} near-duplicate groups ({near_affected} images)")
        if all_methods:
            detail_lines.append(f"  Methods: {', '.join(sorted(all_methods))}")
        if orientations:
            parts = [f"{c} {o}" for o, c in sorted(orientations.items())]
            detail_lines.append(f"  Orientations: {', '.join(parts)}")
    return Reportable(
        report_type="key_value",
        title="Duplicates",
        data={
            "brief": f"{len(exact_groups)} exact, {len(near_groups)} near",
            "detail_lines": detail_lines,
            "exact_groups": len(exact_groups),
            "near_groups": len(near_groups),
            "exact_affected": exact_affected,
            "near_affected": near_affected,
            "near_methods": sorted(all_methods),
            "near_orientations": orientations,
        },
        description=(f"{len(exact_groups)} exact duplicate groups, {len(near_groups)} near-duplicate groups found."),
    )


def _label_distribution_finding(
    raw: DataCleaningRawOutputs,
    label_source: str | None = None,
) -> Reportable | None:
    """Build a Label Distribution finding from raw results, or None if no label stats."""
    if not raw.label_stats:
        return None

    class_count = raw.label_stats.get("class_count", 0)
    if class_count == 0:
        return None  # No labels — suppress finding

    label_counts = raw.label_stats.get("label_counts_per_class", {})
    item_count = raw.label_stats.get("item_count", 0)
    counts_list = list(label_counts.values()) if label_counts else []
    imbalance_ratio = round(max(counts_list) / min(counts_list), 1) if counts_list and min(counts_list) > 0 else 0.0
    footer_lines: list[str] = []
    if label_source:
        footer_lines.append(f"Labels {label_source}")
    if imbalance_ratio == 1.0:
        footer_lines.append("Balanced: all classes have equal counts")
    elif imbalance_ratio != 0.0:
        footer_lines.append(f"Imbalance ratio: {imbalance_ratio} (max/min)")
    return Reportable(
        report_type="table",
        title=(
            "Label/Directory_Name Distribution"
            if label_source == "inferred from directory names"
            else "Label Distribution"
        ),
        data={
            "brief": f"{class_count} classes, {item_count} items",
            "table_data": label_counts,
            "table_headers": ("Class", "Count"),
            "footer_lines": footer_lines,
            # Keep existing keys for JSON/YAML consumers
            "label_counts": label_counts,
            "class_count": class_count,
            "item_count": item_count,
            "imbalance_ratio": imbalance_ratio,
            "label_source": label_source,
        },
        description=(f"{class_count} classes, {item_count} items."),
    )


def _classwise_finding(raw: DataCleaningRawOutputs) -> Reportable | None:
    """Build a Classwise Outliers finding from raw results, or None if unavailable."""
    pivot = raw.classwise_outliers
    if pivot is None:
        return None

    rows = pivot.get("rows", None)
    if not rows:
        return None

    level = pivot.get("level", "image")

    # The last row is the "Total" row
    total_row = rows[-1] if rows else {}
    class_rows = rows[:-1] if len(rows) > 1 else rows
    total_count = total_row.get("count", 0)
    total_pct = total_row.get("pct", 0.0)
    subject = "targets" if level == "target" else "images"

    return Reportable(
        report_type="pivot_table",
        title="Classwise Outliers",
        data={
            "brief": f"{len(class_rows)} classes, {total_count} {subject} ({total_pct}%)",
            "level": level,
            "table_data": rows,
            "table_headers": ["class_name", "count", "%"],
        },
        description=(f"{total_count} {subject} ({total_pct}%) flagged as outliers across {len(class_rows)} classes."),
    )


def _build_findings(
    raw: DataCleaningRawOutputs,
    metadata: Metadata | None,  # noqa: ARG001 - reserved for future metadata-based findings
    label_source: str | None = None,
) -> list[Reportable]:
    """Generate human-readable findings from raw results."""
    findings: list[Reportable] = []

    # Outlier findings — count distinct images, not total flags
    outlier_issues = raw.img_outliers.get("issues", [])
    outlier_image_count = len({issue["item_index"] for issue in outlier_issues})
    if outlier_image_count > 0:
        pct = (outlier_image_count / raw.dataset_size) * 100 if raw.dataset_size else 0
        # Per-metric breakdown: count distinct images per metric
        _per_metric_sets: dict[str, set[int]] = {}
        for issue in outlier_issues:
            _per_metric_sets.setdefault(issue["metric_name"], set()).add(issue["item_index"])
        per_metric = {k: len(v) for k, v in _per_metric_sets.items()}
        findings.append(
            Reportable(
                report_type="key_value",
                title="Image Outliers",
                data={
                    "brief": f"{outlier_image_count} images ({round(pct, 1)}%)",
                    "multi_metric_subject": "images",
                    "count": outlier_image_count,
                    "percentage": round(pct, 1),
                    "per_metric": per_metric,
                    "total_flags": len(outlier_issues),
                    "dataset_size": raw.dataset_size,
                },
                description=f"{outlier_image_count} images ({pct:.1f}%) flagged as outliers.",
            )
        )

    # Target outlier findings — count distinct (item, target) pairs
    target_issues = raw.target_outliers.get("issues", []) if raw.target_outliers else []
    target_pair_count = len({(issue["item_index"], issue.get("target_index")) for issue in target_issues})
    if target_pair_count > 0:
        # Total target count from label stats for percentage
        total_targets = sum(raw.label_stats.get("label_counts_per_class", {}).values()) if raw.label_stats else 0
        target_pct = round((target_pair_count / total_targets) * 100, 1) if total_targets > 0 else 0.0
        # Per-metric breakdown for targets
        _target_metric_sets: dict[str, set[tuple[int, int | None]]] = {}
        for issue in target_issues:
            key = (issue["item_index"], issue.get("target_index"))
            _target_metric_sets.setdefault(issue["metric_name"], set()).add(key)
        target_per_metric = {k: len(v) for k, v in _target_metric_sets.items()}
        findings.append(
            Reportable(
                report_type="key_value",
                title="Target Outliers",
                data={
                    "brief": f"{target_pair_count} targets ({target_pct}%)",
                    "multi_metric_subject": "targets",
                    "count": target_pair_count,
                    "percentage": target_pct,
                    "per_metric": target_per_metric,
                    "total_flags": len(target_issues),
                    "total_targets": total_targets,
                },
                description=f"{target_pair_count} bounding-box targets ({target_pct}%) flagged as outliers.",
            )
        )

    # Classwise outlier pivot — right after image/target outliers
    classwise = _classwise_finding(raw)
    if classwise:
        findings.append(classwise)

    # Duplicate findings
    dup_finding = _duplicate_finding(raw)
    if dup_finding:
        findings.append(dup_finding)

    # Label distribution finding
    label_finding = _label_distribution_finding(raw, label_source=label_source)
    if label_finding:
        findings.append(label_finding)

    return findings


def _item_id_of(idx: IndexValue) -> int:
    """Extract the item ID from an :class:`IndexValue`.

    Returns the ``int`` directly for image-level indices, or the ``"item"``
    field from a :class:`SourceIndexDict` for target-level indices.
    """
    if isinstance(idx, dict):
        return idx["item"]
    return idx


def _collect_flagged_indices(raw: DataCleaningRawOutputs) -> set[int]:
    """Collect all unique item indices flagged by outlier or duplicate detection."""
    flagged: set[int] = set()

    # Outlier-flagged items
    for issue in raw.img_outliers.get("issues", []):
        flagged.add(issue["item_index"])

    # Duplicate-flagged items (keep first in each group, flag the rest)
    for group in raw.duplicates.get("items", {}).get("exact", []):
        for idx in group[1:]:  # keep first, flag rest
            flagged.add(_item_id_of(idx))
    for group in raw.duplicates.get("items", {}).get("near", []):
        for idx in group["indices"][1:]:
            flagged.add(_item_id_of(idx))

    return flagged


# ---------------------------------------------------------------------------
# Processor (internal — not exported)
# ---------------------------------------------------------------------------


def _resolve_flags(params: DataCleaningParameters) -> tuple[ImageStats, ImageStats]:
    """Resolve outlier and hash flags from cleaning parameters."""
    outlier_flags = ImageStats.NONE
    for name in params.outlier_flags:
        outlier_flags |= FLAG_MAP[name]

    hash_flags = ImageStats.NONE
    if params.duplicate_flags is not None:
        for name in params.duplicate_flags:
            hash_flags |= HASH_FLAG_MAP[name]
    else:
        # DataEval default when no flags are specified
        hash_flags = ImageStats.HASH_DUPLICATES_BASIC

    return outlier_flags, hash_flags


def _split_outlier_issues(
    issues_df: "pl.DataFrame",
) -> tuple["pl.DataFrame", "pl.DataFrame | None"]:
    """Split outlier issues into image-level and target-level DataFrames."""
    if "target_index" in issues_df.columns:
        img_issues = issues_df.filter(issues_df["target_index"].is_null())
        target_issues = issues_df.filter(issues_df["target_index"].is_not_null())
    else:
        img_issues = issues_df
        target_issues = None
    return img_issues, target_issues


def _validate_cluster_params(params: DataCleaningParameters, extractor: Callable | None) -> None:
    """Validate that cluster-based detection params have an accompanying extractor."""
    has_outlier_cluster = (
        params.outlier_cluster_threshold is not None
        or params.outlier_cluster_algorithm is not None
        or params.outlier_n_clusters is not None
    )
    if has_outlier_cluster and extractor is None:
        raise ValueError(
            "Cluster-based outlier detection requires an extractor. "
            "Configure a model/extractor or remove cluster params."
        )

    has_dup_cluster = (
        params.duplicate_cluster_sensitivity is not None
        or params.duplicate_cluster_algorithm is not None
        or params.duplicate_n_clusters is not None
    )
    if has_dup_cluster and extractor is None:
        raise ValueError(
            "Cluster-based duplicate detection requires an extractor. "
            "Configure a model/extractor or remove cluster params."
        )


@dataclass(frozen=True)
class CleaningRunContext:
    """Cache and extractor plumbing passed from execute() to _run_cleaning()."""

    cache: "WorkflowCache | None" = None
    sel_key: str | None = None
    extractor_config: Any = None
    transforms: Callable | None = None
    batch_size: int | None = None


def _build_class_labels_df(
    metadata: "Metadata",
) -> tuple[pl.DataFrame, list[str], dict[str, int]]:
    """Build a DataFrame mapping items/targets to class names and label counts.

    Returns
    -------
    labels_df
        DataFrame with ``item_index``, optionally ``target_index``, and ``class_name``.
    id_cols
        Column names to use as join keys (``["item_index"]`` or
        ``["item_index", "target_index"]``).
    label_counts
        Number of items/targets per class name.
    """
    index2label = metadata.index2label
    has_targets = metadata.has_targets()

    label_counts: dict[str, int] = {}
    for lbl in metadata.class_labels:
        name = index2label.get(lbl, str(lbl))
        label_counts[name] = label_counts.get(name, 0) + 1

    if has_targets and hasattr(metadata, "target_data"):
        td = metadata.target_data.select("item_index", "target_index", "class_label")
        names = [index2label.get(int(c), str(c)) for c in td["class_label"].to_list()]
        labels_df = td.with_columns(pl.Series("class_name", names)).select("item_index", "target_index", "class_name")
        id_cols = ["item_index", "target_index"]
    else:
        item_ids = getattr(metadata, "item_indices", None) or list(range(len(metadata.class_labels)))
        names = [index2label.get(lbl, str(lbl)) for lbl in metadata.class_labels]
        labels_df = pl.DataFrame({"item_index": item_ids, "class_name": names})
        id_cols = ["item_index"]

    return labels_df, id_cols, label_counts


def _compute_classwise_pivot(
    target_issues: "pl.DataFrame | None",
    img_issues: "pl.DataFrame",
    metadata: "Metadata | None",
) -> "ClasswisePivotDict | None":
    """Compute classwise outlier pivot from the globally-detected outlier issues.

    Groups the same target (or image) outlier issues used for the headline
    count by class, so the per-class rows sum to the headline total.

    Returns a simplified summary with count and % of labels flagged per class.
    """
    if metadata is None:
        return None
    try:
        # Use target-level issues for OD datasets, image-level otherwise
        has_targets = metadata.has_targets()
        if has_targets:
            if target_issues is None or target_issues.shape[0] == 0:
                return None
            issues_df = target_issues
        else:
            if img_issues.shape[0] == 0:
                return None
            issues_df = img_issues

        labels_df, id_cols, label_counts = _build_class_labels_df(metadata)
        total_labels = sum(label_counts.values())

        for col in id_cols:
            if col in issues_df.columns and issues_df[col].dtype != labels_df[col].dtype:
                labels_df = labels_df.with_columns(pl.col(col).cast(issues_df[col].dtype))

        # Count unique outlier items/targets per class (not per metric flag)
        unique_per_class = (
            issues_df.join(labels_df, on=id_cols, how="left")
            .select(id_cols + ["class_name"])
            .unique()
            .group_by("class_name")
            .len()
            .sort("len", descending=True)
        )

        rows: list[ClasswiseRowDict] = []
        grand_total = 0
        for row_dict in unique_per_class.to_dicts():
            name = str(row_dict.get("class_name", ""))
            count = int(row_dict.get("len", 0))
            grand_total += count
            denom = label_counts.get(name, 0)
            pct = round((count / denom) * 100, 1) if denom > 0 else 0.0
            rows.append({"class_name": name, "count": count, "pct": pct})

        total_pct = round((grand_total / total_labels) * 100, 1) if total_labels > 0 else 0.0
        rows.append({"class_name": "Total", "count": grand_total, "pct": total_pct})

        return ClasswisePivotDict(
            level="target" if has_targets else "image",
            rows=rows,
        )
    except Exception:  # noqa: BLE001
        logger.warning("Classwise pivot unavailable", exc_info=True)
    return None


def _run_cleaning(
    dataset: AnnotatedDataset[Any],
    params: DataCleaningParameters,
    extractor: Callable | None = None,
    metadata: Metadata | None = None,
    run_ctx: CleaningRunContext | None = None,
) -> DataCleaningRawOutputs:
    """Run outlier + duplicate detection on dataset.

    Stats are obtained via :func:`~dataeval_app.cache.get_or_compute_stats`,
    which transparently handles disk caching when *run_ctx* provides a cache
    and selection key.  Evaluators consume pre-computed stats via
    ``from_stats()``; cluster-based detection (when an extractor is
    configured) is handled separately via ``from_clusters()`` / ``evaluate()``.
    """
    import polars as pl
    from dataeval.quality import OutliersOutput

    from dataeval_app.cache import get_or_compute_stats

    _validate_cluster_params(params, extractor)

    outlier_flags, hash_flags = _resolve_flags(params)

    # --- Centralized stats: cache-aware load / compute / save ---
    _cache = run_ctx.cache if run_ctx else None
    _sel_key = run_ctx.sel_key if run_ctx else None
    calc_result = get_or_compute_stats(
        desired_flags=outlier_flags | hash_flags,
        dataset=dataset,
        cache=_cache,
        selection_key=_sel_key,
    )

    # --- Outlier detection via from_stats() ---
    outliers_eval = Outliers(
        flags=outlier_flags,
        outlier_threshold=(params.outlier_method, params.outlier_threshold),
    )
    outlier_output = outliers_eval.from_stats(calc_result, per_target=True)  # type: ignore[arg-type]

    # Cluster-based outlier detection (separate path, uses embeddings)
    has_outlier_cluster = extractor is not None and params.outlier_cluster_threshold is not None
    if has_outlier_cluster:
        from dataeval.core._clusterer import cluster

        if run_ctx is not None and run_ctx.extractor_config is not None:
            from dataeval_app.cache import get_or_compute_embeddings

            embeddings_array = get_or_compute_embeddings(
                dataset,
                run_ctx.extractor_config,
                run_ctx.transforms,
                run_ctx.batch_size,
                cache=_cache,
                selection_key=_sel_key,
            )
        else:
            from dataeval.utils.arrays import flatten_samples, to_numpy

            images = [item[0] if isinstance(item, tuple) else item for item in dataset]
            embeddings = extractor(images)  # type: ignore[misc]
            embeddings_array = flatten_samples(to_numpy(embeddings))

        cluster_result = cluster(
            embeddings_array,
            algorithm=params.outlier_cluster_algorithm or "hdbscan",
            n_clusters=params.outlier_n_clusters,
        )
        cluster_outlier_output = outliers_eval.from_clusters(
            embeddings_array,
            cluster_result,
            cluster_threshold=params.outlier_cluster_threshold,
        )

        # Merge stats-based + cluster-based issues via concat
        # Normalize column order to match DataEval's evaluate() behavior
        column_order = ["item_index", "target_index", "metric_name", "metric_value"]
        stats_issues = outlier_output.data()
        cluster_issues = cluster_outlier_output.data()
        dfs: list[pl.DataFrame] = []
        for df in [stats_issues, cluster_issues]:
            if "target_index" not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias("target_index"))
            dfs.append(df.select(column_order))
        merged_issues = pl.concat(dfs).sort(["item_index", "metric_name"])
        if merged_issues["target_index"].null_count() == len(merged_issues):
            merged_issues = merged_issues.drop("target_index")
        outlier_output = OutliersOutput(merged_issues)

    img_issues, target_issues = _split_outlier_issues(outlier_output.data())

    # --- Classwise outlier pivot (uses the same globally-detected issues as the headline count) ---
    classwise_pivot = _compute_classwise_pivot(target_issues, img_issues, metadata)

    # --- Duplicate detection ---
    has_dup_cluster = extractor is not None and params.duplicate_cluster_sensitivity is not None
    if has_dup_cluster:
        # Fall back to evaluate() when cluster detection is configured, because
        # the merge logic (_merge_item_results) is private in DataEval.
        _batch = run_ctx.batch_size if run_ctx else None
        duplicates_full = _build_duplicates(params, extractor=extractor, batch_size=_batch)
        duplicates_result = duplicates_full.evaluate(dataset)  # type: ignore[arg-type]
    else:
        # Pure hash-based: use from_stats()
        dup_kwargs: dict[str, object] = {"merge_near_duplicates": params.duplicate_merge_near}
        if params.duplicate_flags is not None:
            dup_kwargs["flags"] = hash_flags
        duplicates_eval = Duplicates(**dup_kwargs)  # type: ignore[arg-type]
        duplicates_result = duplicates_eval.from_stats(calc_result)  # type: ignore[arg-type]

    # Label stats
    label_stats: LabelStatsDict = _compute_label_stats(metadata) if metadata else {}  # type: ignore[assignment]

    return DataCleaningRawOutputs(
        dataset_size=len(dataset),
        img_outliers=_serialize_outlier_issues(img_issues),
        target_outliers=_serialize_outlier_issues(target_issues)
        if target_issues is not None and len(target_issues) > 0
        else None,
        duplicates=_serialize_duplicates(duplicates_result),
        label_stats=label_stats,
        classwise_outliers=classwise_pivot,
    )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class DataCleaningWorkflow(WorkflowProtocol[DataCleaningMetadata, DataCleaningOutputs]):
    """Data cleaning workflow using DataEval evaluators."""

    @property
    def name(self) -> str:
        """Workflow identifier."""
        return "data-cleaning"

    @property
    def description(self) -> str:
        """Human-readable description."""
        return "Outlier and duplicate detection for image datasets"

    @property
    def params_schema(self) -> type[DataCleaningParameters]:
        """Pydantic model for workflow parameters."""
        return DataCleaningParameters

    @property
    def output_schema(self) -> type[DataCleaningOutputs]:
        """Pydantic model for workflow output."""
        return DataCleaningOutputs

    def execute(
        self,
        context: WorkflowContext,
        params: BaseModel | None = None,
    ) -> WorkflowResult[DataCleaningMetadata, DataCleaningOutputs]:
        """Run data cleaning workflow on dataset."""
        from dataeval_app.selection import build_selection

        if not isinstance(context, WorkflowContext):
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=[f"Expected WorkflowContext, got {type(context).__name__}"],
            )

        if params is None:
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=["DataCleaningParameters required (no defaults per CR-4.14-G-1)"],
            )

        if not isinstance(params, DataCleaningParameters):
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=[f"Expected DataCleaningParameters, got {type(params).__name__}"],
            )

        try:
            # All arg-type suppressions in this block: MaiteDataset (and Select wrapper)
            # conforms to DataEval's dataset protocol at runtime via duck typing;
            # pyright can't verify cross-library structural conformance.
            from dataeval_app.cache import selection_repr as _sel_repr

            # Resolve the single dataset context (cleaning is single-dataset)
            dc = next(iter(context.dataset_contexts.values()))

            # 1. Apply selection if configured
            dataset = dc.dataset
            if dc.selection_steps:
                logger.info("Applying selection (%d steps)", len(dc.selection_steps))
                dataset = build_selection(dataset, dc.selection_steps)  # type: ignore[arg-type]

            # Compute selection key (shared by metadata + stats caching)
            sel_key = _sel_repr(dataset)

            # 2. Build extractor if configured
            extractor = None
            if dc.extractor:
                logger.info("Building extractor")
                extractor = build_extractor(
                    extractor_config=dc.extractor,
                    transforms=dc.transforms,
                )
                logger.info("Extractor complete")

            # 3. Build metadata for label stats (with cache)
            metadata = get_or_compute_metadata(
                dataset,
                auto_bin_method=context.metadata_auto_bin_method,
                exclude=context.metadata_exclude or None,
                continuous_factor_bins=context.metadata_continuous_factor_bins,
                cache=context.cache,
                selection_key=sel_key,
            )

            # 4. Run cleaning evaluators (cache-aware when cache is configured)
            logger.info("Running outlier and duplicate detection on %d items", len(dataset))
            run_ctx = CleaningRunContext(
                cache=context.cache,
                sel_key=sel_key,
                extractor_config=dc.extractor,
                transforms=dc.transforms,
                batch_size=dc.batch_size,
            )
            raw = _run_cleaning(
                dataset,
                params,
                extractor,
                metadata,  # type: ignore[arg-type]
                run_ctx,
            )
            logger.info(
                "Detection complete: %d outliers, %d exact dup groups, %d near dup groups",
                raw.img_outliers.get("count", 0),
                len(raw.duplicates.get("items", {}).get("exact", [])),
                len(raw.duplicates.get("items", {}).get("near", [])),
            )

            # 5. Generate findings from raw results
            findings = _build_findings(raw, metadata, label_source=dc.label_source)

            # 6. Preparatory mode: compute clean indices (exclude flagged items)
            result_metadata = DataCleaningMetadata(
                mode=params.mode,
                evaluators=["outliers", "duplicates"],
            )
            if params.mode == "preparatory":
                flagged = _collect_flagged_indices(raw)
                all_indices = set(range(raw.dataset_size))
                clean_indices = sorted(all_indices - flagged)
                result_metadata.flagged_indices = sorted(flagged)
                result_metadata.clean_indices = clean_indices
                result_metadata.removed_count = len(flagged)
                findings.append(
                    Reportable(
                        report_type="key_value",
                        title="Preparatory Mode",
                        data={
                            "brief": f"{len(flagged)} flagged, {len(clean_indices)} retained",
                            "flagged": len(flagged),
                            "retained": len(clean_indices),
                        },
                        description=(
                            f"Preparatory mode: {len(flagged)} items flagged for removal, "
                            f"{len(clean_indices)} items retained."
                        ),
                    )
                )

            # 7. Build report
            report = DataCleaningReport(
                summary=f"Data cleaning complete. Dataset: {raw.dataset_size} items. Mode: {params.mode}.",
                findings=findings,
            )

            return WorkflowResult(
                name=self.name,
                success=True,
                data=DataCleaningOutputs(raw=raw, report=report),
                metadata=result_metadata,
            )
        except Exception as e:
            logger.exception("Workflow '%s' failed", self.name)
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=[f"Workflow execution failed: {e}"],
            )

    def _empty_outputs(self) -> DataCleaningOutputs:
        return DataCleaningOutputs(
            raw=DataCleaningRawOutputs(dataset_size=0),
            report=DataCleaningReport(summary="Workflow failed", findings=[]),
        )
