"""Data Cleaning Workflow — orchestration + processor + factory helpers."""

__all__ = ["DataCleaningWorkflow"]

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import polars as pl
from dataeval import Metadata
from dataeval.flags import ImageStats
from dataeval.protocols import AnnotatedDataset
from dataeval.quality import Duplicates, Outliers

from dataeval_flow._binning import attach_binning
from dataeval_flow._blocks import Fields
from dataeval_flow._cache import active_cache, get_or_compute_metadata
from dataeval_flow._classwise import class_labels_frame, classwise_pivot, split_outlier_issues
from dataeval_flow._embeddings import build_extractor
from dataeval_flow._policy import policy_for
from dataeval_flow._stats import HASH_FLAG_MAP, columns_for, restrict_columns
from dataeval_flow._stats import OUTLIER_FLAG_MAP as FLAG_MAP
from dataeval_flow.workflows._base import Finding, Workflow, effective_value_range
from dataeval_flow.workflows._context import WorkflowContext
from dataeval_flow.workflows.data_cleaning._config import DataCleaningConfig
from dataeval_flow.workflows.data_cleaning._internal import (
    _compute_embeddings,
    _merge_duplicate_results,
    _merge_outlier_outputs,
)
from dataeval_flow.workflows.data_cleaning._outputs import (
    DataCleaningMetadata,
    DataCleaningOutput,
    DataCleaningRawOutput,
    DataCleaningReport,
    DataCleaningResult,
    DetectionDict,
    DuplicatesDict,
    IndexValue,
    LabelStatsDict,
    OutlierIssuesDict,
    SourceIndexDict,
)
from dataeval_flow.workflows.data_cleaning._report import build_findings, collect_flagged_indices

_logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _build_outliers(
    params: DataCleaningConfig,
    extractor: Callable | None = None,
) -> Outliers:
    """Build Outliers evaluator from cleaning parameters."""
    flags = ImageStats.NONE
    for name in params.outlier_flags:
        flags |= FLAG_MAP[name]

    return Outliers(
        flags=flags,
        outlier_threshold=(params.outlier_method, params.outlier_threshold),
        cluster_threshold=params.outlier_cluster_threshold,
        cluster_algorithm=params.outlier_cluster_algorithm,
        n_clusters=params.outlier_n_clusters,
        extractor=extractor,
    )


def _build_duplicates(
    params: DataCleaningConfig,
    extractor: Callable | None = None,
    batch_size: int | None = None,
) -> Duplicates:
    """Build Duplicates evaluator from cleaning parameters."""
    # Build hash flags
    flags = ImageStats.NONE
    if params.duplicate_flags is not None:
        for name in params.duplicate_flags:
            flags |= HASH_FLAG_MAP[name]

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
    """Serialize outlier issues Polars DataFrame to plain dict."""
    return {
        "issues": issues.to_dicts(),  # type: ignore[typeddict-item]
        "count": len(issues),
    }


def _serialize_duplicates(result: Any) -> "DuplicatesDict":
    """Serialize DuplicatesOutput to plain dict from its DataFrame."""

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
    _, _, label_counts = class_labels_frame(metadata)

    return {
        "item_count": metadata.item_count,
        "class_count": len(metadata.index2label),
        "index2label": dict(metadata.index2label),
        "label_counts_per_class": label_counts,
    }


# ---------------------------------------------------------------------------
# Processor (internal — not exported)
# ---------------------------------------------------------------------------


def _resolve_flags(params: DataCleaningConfig) -> tuple[ImageStats, ImageStats]:
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


@dataclass(frozen=True)
class CleaningRunContext:
    """Extractor plumbing passed from execute() to _run_cleaning()."""

    extractor_config: Any = None
    transforms: Callable | None = None
    batch_size: int | None = None


def _compute_classwise_pivot(
    target_issues: "pl.DataFrame | None",
    img_issues: "pl.DataFrame",
    metadata: "Metadata | None",
) -> Any:
    """The classwise outlier pivot, or ``None`` where there is none or it could not be built.

    data-cleaning reports a pivot it could not build as no outliers. That is the hidden failure a chain's
    `classwise-outliers` step makes visible (spec §10.6).
    """
    if metadata is None:
        return None
    try:
        return classwise_pivot(target_issues, img_issues, metadata)
    except Exception:
        _logger.warning("Classwise pivot unavailable", exc_info=True)
    return None


def _run_duplicate_detection(
    params: DataCleaningConfig,
    hash_flags: ImageStats,
    calc_result: Any,
    embeddings_array: Any | None,
    run_ctx: CleaningRunContext | None,
) -> Any:
    """Run hash-based and optionally cluster-based duplicate detection."""
    import time as _time

    _logger.info("  [4e] Running duplicate detection…")
    _t0 = _time.monotonic()

    # Hash-based duplicate detection (always, using cached stats)
    dup_kwargs: dict[str, object] = {"merge_near_duplicates": params.duplicate_merge_near}
    if params.duplicate_flags is not None:
        dup_kwargs["flags"] = hash_flags
    duplicates_eval = Duplicates(**dup_kwargs)  # type: ignore[arg-type]
    hash_dup_result = duplicates_eval.from_stats(restrict_columns(calc_result, columns_for([None], hash_flags)))

    if embeddings_array is not None and params.duplicate_cluster_sensitivity is not None:
        duplicates_result = _merge_duplicate_results(hash_dup_result, embeddings_array, params, run_ctx)
    else:
        duplicates_result = hash_dup_result

    _logger.info("  [4e] Duplicate detection done in %.1fs", _time.monotonic() - _t0)
    return duplicates_result


def _run_cleaning(
    dataset: AnnotatedDataset[Any],
    params: DataCleaningConfig,
    extractor: Callable | None = None,
    metadata: Metadata | None = None,
    run_ctx: CleaningRunContext | None = None,
    value_range: tuple[float, float] | None = None,
    *,
    context: WorkflowContext | None = None,
) -> DataCleaningRawOutput:
    """Run outlier + duplicate detection on dataset."""
    import time as _time

    from dataeval_flow._cache import get_or_compute_stats
    from dataeval_flow._stats import stats_policy_for

    outlier_flags, hash_flags = _resolve_flags(params)

    stats_policy = stats_policy_for(context, outlier_flags=outlier_flags, duplicate_flags=hash_flags)

    # --- Centralized stats: cache-aware load / compute / save ---
    _t0 = _time.monotonic()
    calc_result = get_or_compute_stats(
        stats_policy,
        dataset=dataset,
        value_range=value_range,
    )
    _logger.info("  [4a] Image stats ready in %.1fs", _time.monotonic() - _t0)

    # --- Outlier detection via from_stats() ---
    _logger.info("  [4b] Running stats-based outlier detection…")
    _t0 = _time.monotonic()
    outliers_eval = Outliers(
        flags=outlier_flags,
        outlier_threshold=(params.outlier_method, params.outlier_threshold),
    )
    outlier_output = outliers_eval.from_stats(
        restrict_columns(calc_result, columns_for(stats_policy.outliers_from, outlier_flags)),
        per_target=True,
    )
    _logger.info("  [4b] Stats-based outlier detection done in %.1fs", _time.monotonic() - _t0)

    # --- Shared embeddings for cluster-based detection ---
    has_outlier_cluster = extractor is not None and params.outlier_cluster_threshold is not None
    has_dup_cluster = extractor is not None and params.duplicate_cluster_sensitivity is not None
    embeddings_array = None

    if (has_outlier_cluster or has_dup_cluster) and extractor is not None:
        embeddings_array = _compute_embeddings(dataset, extractor, run_ctx)

    # --- Cluster-based outlier detection ---
    if has_outlier_cluster and embeddings_array is not None:
        outlier_output = _merge_outlier_outputs(outliers_eval, outlier_output, embeddings_array, params, run_ctx)

    img_issues, target_issues = split_outlier_issues(outlier_output.data())

    # --- Classwise outlier pivot ---
    classwise = _compute_classwise_pivot(target_issues, img_issues, metadata)

    # --- Duplicate detection ---
    duplicates_result = _run_duplicate_detection(params, hash_flags, calc_result, embeddings_array, run_ctx)

    # Label stats
    label_stats: LabelStatsDict = _compute_label_stats(metadata) if metadata else {}  # type: ignore[assignment]

    return DataCleaningRawOutput(
        dataset_size=len(dataset),
        img_outliers=_serialize_outlier_issues(img_issues),
        target_outliers=_serialize_outlier_issues(target_issues)
        if target_issues is not None and len(target_issues) > 0
        else None,
        duplicates=_serialize_duplicates(duplicates_result),
        label_stats=label_stats,
        classwise_outliers=classwise,
    )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class DataCleaningWorkflow(Workflow[DataCleaningConfig, DataCleaningResult]):
    """Data cleaning workflow using DataEval evaluators."""

    name: ClassVar[str] = "data-cleaning"
    description: ClassVar[str] = "Outlier and duplicate detection for image datasets"

    def run(self, config: DataCleaningConfig, context: WorkflowContext) -> DataCleaningResult:
        """Run data cleaning workflow on dataset."""
        import time as _time

        from dataeval_flow._cache import selection_repr as _sel_repr
        from dataeval_flow._view import build_view

        # All arg-type suppressions in this method: MaiteDataset (and Select wrapper)
        # conforms to DataEval's dataset protocol at runtime via duck typing;
        # pyright can't verify cross-library structural conformance.
        policy = policy_for(context, config)

        # Resolve the single dataset context (cleaning is single-dataset)
        dc = next(iter(context.dataset_contexts.values()))

        # 1. Apply selection if configured
        dataset = dc.dataset
        if dc.view_operations:
            _logger.info("[1/4] Applying selection (%d steps)…", len(dc.view_operations))
            _t0 = _time.monotonic()
            dataset = build_view(dataset, dc.view_operations)  # type: ignore[arg-type]
            _logger.info("[1/4] Selection applied in %.1fs", _time.monotonic() - _t0)

        # Compute selection key (shared by metadata + stats caching)
        sel_key = _sel_repr(dataset)

        # 2. Build extractor if configured
        extractor = None
        if dc.extractor:
            _logger.info("[2/4] Building extractor…")
            _t0 = _time.monotonic()
            extractor = build_extractor(
                extractor_config=dc.extractor,
                transforms=dc.transforms,
            )
            _logger.info("[2/4] Extractor built in %.1fs", _time.monotonic() - _t0)

        # 3–4. Activate cache context so all downstream get_or_compute_*
        # calls automatically use the cache without explicit threading.
        run_ctx = CleaningRunContext(
            extractor_config=dc.extractor,
            transforms=dc.transforms,
            batch_size=dc.batch_size,
        )
        with contextlib.ExitStack() as stack:
            if dc.cache is not None:
                stack.enter_context(active_cache(dc.cache, sel_key))

            # 3. Build metadata for label stats (cache-aware via active_cache)
            _logger.info("[3/4] Loading metadata…")
            _t0 = _time.monotonic()
            metadata = get_or_compute_metadata(dataset, policy)
            _logger.info("[3/4] Metadata ready in %.1fs", _time.monotonic() - _t0)

            # 4. Run cleaning evaluators (cache-aware via active_cache)
            _logger.info("[4/4] Running outlier and duplicate detection on %d items…", len(dataset))
            _t0 = _time.monotonic()
            raw = _run_cleaning(
                dataset,
                config,
                extractor,
                metadata,  # type: ignore[arg-type]
                run_ctx,
                effective_value_range(dc, config),
                context=context,
            )
        _logger.info(
            "[4/4] Detection complete in %.1fs: %d outliers, %d exact dup groups, %d near dup groups",
            _time.monotonic() - _t0,
            raw.img_outliers.get("count", 0),
            len(raw.duplicates.get("items", {}).get("exact", [])),
            len(raw.duplicates.get("items", {}).get("near", [])),
        )

        # 5. Generate findings from raw results
        findings = build_findings(
            raw,
            metadata,
            config.health_thresholds,
            label_source=dc.label_source,
            source=dc.name,
            outlier_method=config.outlier_method,
            outlier_threshold=config.outlier_threshold,
        )

        # 6. Preparatory mode: compute clean indices (exclude flagged items)
        result_metadata = DataCleaningMetadata(
            mode=config.mode,
            evaluators=["outliers", "duplicates"],
        )
        attach_binning(result_metadata, metadata, policy)
        if config.mode == "preparatory":
            flagged = collect_flagged_indices(raw)
            all_indices = set(range(raw.dataset_size))
            clean_indices = sorted(all_indices - flagged)
            result_metadata.flagged_indices = sorted(flagged)
            result_metadata.clean_indices = clean_indices
            result_metadata.removed_count = len(flagged)
            findings.append(
                Finding(
                    title="Preparatory Mode",
                    brief=f"{len(flagged)} flagged, {len(clean_indices)} retained",
                    description=(
                        f"Preparatory mode: {len(flagged)} items flagged for removal, "
                        f"{len(clean_indices)} items retained."
                    ),
                    blocks=[Fields(items=[("Flagged", len(flagged)), ("Retained", len(clean_indices))])],
                )
            )

        # 7. Build report
        report = DataCleaningReport(
            summary=f"Data cleaning complete. Dataset: {raw.dataset_size} items. Mode: {config.mode}.",
            findings=findings,
        )

        return DataCleaningResult(
            type=self.name,
            success=True,
            output=DataCleaningOutput(raw=raw, report=report),
            metadata=result_metadata,
            dataset=dataset,
        )
