"""Data prioritization workflow."""

from __future__ import annotations

import contextlib
import logging
import time as _time
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from dataeval.flags import ImageStats
from dataeval.protocols import AnnotatedDataset
from dataeval.quality import Duplicates, Outliers
from dataeval.scope import Prioritize
from numpy.typing import NDArray

from dataeval_flow._cache import (
    active_cache,
    get_or_compute_embeddings,
    get_or_compute_stats,
    selection_repr,
)
from dataeval_flow._embeddings import build_extractor
from dataeval_flow._stats import HASH_FLAG_MAP as _HASH_FLAG_MAP
from dataeval_flow._stats import OUTLIER_FLAG_MAP as _OUTLIER_FLAG_MAP
from dataeval_flow._stats import columns_for, restrict_columns, stats_policy_for
from dataeval_flow.workflows._base import Workflow, effective_value_range
from dataeval_flow.workflows._context import DatasetContext, WorkflowContext
from dataeval_flow.workflows.data_prioritization._config import (
    DataPrioritizationCleaningConfig,
    DataPrioritizationConfig,
)
from dataeval_flow.workflows.data_prioritization._outputs import (
    CleaningSummaryDict,
    DataPrioritizationMetadata,
    DataPrioritizationOutput,
    DataPrioritizationRawOutput,
    DataPrioritizationReport,
    DataPrioritizationResult,
    PerDatasetPrioritizationDict,
)
from dataeval_flow.workflows.data_prioritization._report import build_findings

_logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding extraction helper (mirrors OOD pattern)
# ---------------------------------------------------------------------------


def _get_embeddings_for_context(
    dc: DatasetContext,
    dataset: AnnotatedDataset[Any],
) -> NDArray[np.float32]:
    """Extract embeddings for a dataset context, using cache if available."""
    if dc.extractor is None:
        raise ValueError(
            "Data prioritization requires a model/extractor to compute embeddings. "
            "Configure an extractor in the task config."
        )
    sel_key = selection_repr(dataset)
    with contextlib.ExitStack() as stack:
        if dc.cache is not None:
            stack.enter_context(active_cache(dc.cache, sel_key))
        return get_or_compute_embeddings(
            dataset,
            dc.extractor,
            dc.transforms,
            dc.batch_size,
        )


# ---------------------------------------------------------------------------
# Flag resolution for cleaning
# ---------------------------------------------------------------------------


def _resolve_cleaning_flags(
    cleaning: DataPrioritizationCleaningConfig,
) -> tuple[ImageStats, ImageStats]:
    """Resolve outlier and hash flags from cleaning config."""
    outlier_flags = ImageStats.NONE
    for flag_name in cleaning.outlier_flags:
        outlier_flags |= _OUTLIER_FLAG_MAP[flag_name]

    hash_flags = ImageStats.NONE
    if cleaning.duplicate_flags is not None:
        for flag_name in cleaning.duplicate_flags:
            hash_flags |= _HASH_FLAG_MAP[flag_name]
    else:
        hash_flags = ImageStats.HASH_DUPLICATES_BASIC

    return outlier_flags, hash_flags


# ---------------------------------------------------------------------------
# Cleaning step
# ---------------------------------------------------------------------------


def _run_outlier_detection_per_source(
    cleaning: DataPrioritizationCleaningConfig,
    outlier_flags: ImageStats,
    hash_flags: ImageStats,
    dc: DatasetContext,
    dataset: AnnotatedDataset[Any],
    value_range: tuple[float, float] | None = None,
    *,
    context: WorkflowContext | None = None,
) -> set[int]:
    """Run stats-based outlier detection on a single dataset.

    Returns the set of flagged item indices.
    """
    stats_policy = stats_policy_for(context, outlier_flags=outlier_flags, duplicate_flags=hash_flags)
    sel_key = selection_repr(dataset)
    with contextlib.ExitStack() as stack:
        if dc.cache is not None:
            stack.enter_context(active_cache(dc.cache, sel_key))
        calc_result = get_or_compute_stats(
            stats_policy,
            dataset=dataset,
            value_range=value_range,
        )

    outliers_eval = Outliers(
        flags=outlier_flags,
        outlier_threshold=(cleaning.outlier_method, cleaning.outlier_threshold),
    )
    outlier_output = outliers_eval.from_stats(
        restrict_columns(calc_result, columns_for(stats_policy.outliers_from, outlier_flags))
    )
    return set(outlier_output.outliers.keys())


def _run_duplicate_detection_cross_dataset(
    cleaning: DataPrioritizationCleaningConfig,
    hash_flags: ImageStats,
    all_datasets: list[tuple[str, DatasetContext, AnnotatedDataset[Any]]],
) -> dict[int, set[int]]:
    """Run hash-based duplicate detection across all datasets.

    Returns a mapping of dataset_index -> set of flagged item indices.
    Cross-dataset duplicates that include items from the reference (dataset 0)
    only flag the non-reference side.
    """
    datasets = [ds for _, _, ds in all_datasets]
    if len(datasets) < 2:
        return {}

    dup_kwargs: dict[str, object] = {"merge_near_duplicates": cleaning.duplicate_merge_near}
    if cleaning.duplicate_flags is not None:
        dup_kwargs["flags"] = hash_flags
    duplicates_eval = Duplicates(**dup_kwargs)  # type: ignore[arg-type]
    dup_result = duplicates_eval.evaluate(*datasets)

    flagged: dict[int, set[int]] = {i: set() for i in range(len(datasets))}
    near = {} if cleaning.duplicate_exact_only else dup_result.near
    _collect_flagged_from_groups(dup_result.exact, near, flagged)
    return flagged


def _collect_flagged_from_groups(
    exact_groups: Mapping[int, Sequence[Sequence[int]]],
    near_groups: Mapping[int, Sequence[tuple[Sequence[int], Sequence[str]]]],
    flagged: dict[int, set[int]],
) -> None:
    """Populate *flagged* from exact and near duplicate groups."""
    for ds_idx, groups in exact_groups.items():
        for group in groups:
            flagged[ds_idx].update(group[1:])

    for ds_idx, groups in near_groups.items():
        for indices, _methods in groups:
            flagged[ds_idx].update(indices[1:])


def _run_cleaning(
    cleaning: DataPrioritizationCleaningConfig,
    ref_dc: DatasetContext,
    ref_dataset: AnnotatedDataset[Any],
    add_datasets: list[tuple[str, DatasetContext, AnnotatedDataset[Any]]],
    value_range: tuple[float, float] | None = None,
    *,
    context: WorkflowContext | None = None,
) -> tuple[
    dict[str, set[int]],  # per-source flagged indices
    CleaningSummaryDict,
]:
    """Run the optional cleaning step across all datasets.

    Returns per-source flagged indices and a summary.
    """
    _logger.info("[3/?] Running pre-prioritization cleaning…")
    t0 = _time.monotonic()

    outlier_flags, hash_flags = _resolve_cleaning_flags(cleaning)

    # --- Per-dataset outlier detection ---
    all_sources = [("__reference__", ref_dc, ref_dataset)] + list(add_datasets)
    flagged_outliers: dict[str, set[int]] = {}
    total_outliers = 0
    for name, dc, ds in all_sources:
        flagged = _run_outlier_detection_per_source(
            cleaning, outlier_flags, hash_flags, dc, ds, value_range, context=context
        )
        flagged_outliers[name] = flagged
        total_outliers += len(flagged)
        _logger.info("  Outliers in %s: %d", name, len(flagged))

    # --- Cross-dataset duplicate detection ---
    dup_flagged = _run_duplicate_detection_cross_dataset(cleaning, hash_flags, all_sources)
    flagged_duplicates: dict[str, set[int]] = {}
    total_duplicates = 0
    for i, (name, _, _) in enumerate(all_sources):
        ds_flagged = dup_flagged.get(i, set())
        flagged_duplicates[name] = ds_flagged
        total_duplicates += len(ds_flagged)
        if ds_flagged:
            _logger.info("  Duplicates in %s: %d", name, len(ds_flagged))

    # --- Combine flagged sets ---
    combined_flagged: dict[str, set[int]] = {}
    for name, _, _ in all_sources:
        combined_flagged[name] = flagged_outliers.get(name, set()) | flagged_duplicates.get(name, set())

    total_combined = sum(len(ds) for _, _, ds in all_sources)
    total_removed = sum(len(s) for s in combined_flagged.values())

    summary = CleaningSummaryDict(
        total_combined=total_combined,
        outliers_flagged=total_outliers,
        duplicates_flagged=total_duplicates,
        total_removed=total_removed,
    )

    _logger.info(
        "[3/?] Cleaning complete in %.1fs: removed %d/%d items",
        _time.monotonic() - t0,
        total_removed,
        total_combined,
    )

    return combined_flagged, summary


# ---------------------------------------------------------------------------
# Index remapping
# ---------------------------------------------------------------------------


def _build_clean_mapping(
    total: int,
    flagged: set[int],
) -> tuple[NDArray[np.intp], list[int]]:
    """Build a boolean mask and clean-to-original index mapping.

    Returns
    -------
    mask : NDArray[np.bool_]
        Boolean mask where True = clean (not flagged).
    clean_to_original : list[int]
        Maps clean-space index to original-space index.
    """
    mask = np.ones(total, dtype=bool)
    for idx in flagged:
        if 0 <= idx < total:
            mask[idx] = False
    clean_to_original = [i for i in range(total) if mask[i]]
    return mask, clean_to_original


# ---------------------------------------------------------------------------
# Workflow class
# ---------------------------------------------------------------------------


class DataPrioritizationWorkflow(Workflow[DataPrioritizationConfig, DataPrioritizationResult]):
    """Data prioritization workflow using DataEval Prioritize."""

    name: ClassVar[str] = "data-prioritization"
    description: ClassVar[str] = (
        "Prioritize unlabeled data for labeling based on a reference dataset and optional cleaning"
    )

    def run(self, config: DataPrioritizationConfig, context: WorkflowContext) -> DataPrioritizationResult:
        """Rank each additional source for labeling against the reference, after optional cleaning."""
        dc_items = list(context.dataset_contexts.items())

        # --- 2. Prepare datasets ---
        ref_dc, ref_dataset, add_datasets = self._prepare_datasets(dc_items)

        # --- 3. Extract embeddings ---
        ref_embeddings, add_embeddings = self._extract_all_embeddings(ref_dc, ref_dataset, add_datasets)

        # --- 4. Optional cleaning ---
        cleaning_summary: CleaningSummaryDict | None = None
        per_source_flagged: dict[str, set[int]] = {}
        total_removed = 0

        if config.cleaning is not None:
            per_source_flagged, cleaning_summary = _run_cleaning(
                config.cleaning,
                ref_dc,
                ref_dataset,
                add_datasets,
                effective_value_range(ref_dc, config),
                context=context,
            )
            total_removed = cleaning_summary["total_removed"]

        # --- 5. Build clean embeddings ---
        ref_size = len(ref_dataset)
        ref_flagged = per_source_flagged.get("__reference__", set())
        ref_mask, ref_clean_to_orig = _build_clean_mapping(ref_size, ref_flagged)
        clean_ref_embeddings = ref_embeddings[ref_mask]

        add_clean_info: dict[str, tuple[NDArray[np.float32], list[int], int]] = {}
        for name, _dc, ds in add_datasets:
            ds_size = len(ds)
            ds_flagged = per_source_flagged.get(name, set())
            ds_mask, ds_clean_to_orig = _build_clean_mapping(ds_size, ds_flagged)
            clean_emb = add_embeddings[name][ds_mask]
            add_clean_info[name] = (clean_emb, ds_clean_to_orig, ds_size)

        # --- 6. Prioritization ---
        prioritization_results = self._run_prioritization(config, ref_dc, clean_ref_embeddings, add_clean_info)

        # --- 7. Build outputs ---
        return self._build_workflow_result(
            config,
            ref_size,
            cleaning_summary,
            total_removed,
            prioritization_results,
            ref_clean_to_orig,
            add_clean_info,
        )

    def _prepare_datasets(
        self,
        dc_items: list[tuple[str, DatasetContext]],
    ) -> tuple[DatasetContext, AnnotatedDataset[Any], list[tuple[str, DatasetContext, AnnotatedDataset[Any]]]]:
        """Identify reference vs additional datasets and apply selections."""
        from dataeval_flow._view import build_view

        ref_name, ref_dc = dc_items[0]
        add_contexts = dc_items[1:]

        _logger.info(
            "[1/?] Preparing datasets: reference=%s, additional=%s",
            ref_name,
            [n for n, _ in add_contexts],
        )

        ref_dataset: AnnotatedDataset[Any] = ref_dc.dataset
        if ref_dc.view_operations:
            ref_dataset = build_view(ref_dataset, ref_dc.view_operations)  # type: ignore[arg-type]

        add_datasets: list[tuple[str, DatasetContext, AnnotatedDataset[Any]]] = []
        for a_name, a_dc in add_contexts:
            a_ds = a_dc.dataset
            if a_dc.view_operations:
                a_ds = build_view(a_ds, a_dc.view_operations)  # type: ignore[arg-type]
            add_datasets.append((a_name, a_dc, a_ds))

        return ref_dc, ref_dataset, add_datasets

    def _extract_all_embeddings(
        self,
        ref_dc: DatasetContext,
        ref_dataset: AnnotatedDataset[Any],
        add_datasets: list[tuple[str, DatasetContext, AnnotatedDataset[Any]]],
    ) -> tuple[NDArray[np.float32], dict[str, NDArray[np.float32]]]:
        """Extract embeddings for reference and additional datasets."""
        _logger.info("[2/?] Extracting embeddings…")
        t0 = _time.monotonic()

        ref_embeddings = _get_embeddings_for_context(ref_dc, ref_dataset)
        _logger.info("  Reference embeddings: %s", ref_embeddings.shape)

        add_embeddings: dict[str, NDArray[np.float32]] = {}
        for a_name, a_dc, a_ds in add_datasets:
            emb = _get_embeddings_for_context(a_dc, a_ds)
            add_embeddings[a_name] = emb
            _logger.info("  Additional embeddings (%s): %s", a_name, emb.shape)

        _logger.info("[2/?] Embeddings ready in %.1fs", _time.monotonic() - t0)
        return ref_embeddings, add_embeddings

    def _run_prioritization(
        self,
        config: DataPrioritizationConfig,
        ref_dc: DatasetContext,
        clean_ref_embeddings: NDArray[np.float32],
        add_clean_info: dict[str, tuple[NDArray[np.float32], list[int], int]],
    ) -> dict[str, tuple[list[int], list[float] | None]]:
        """Run prioritization for each additional dataset.

        Returns a mapping of source_name -> (original_indices, scores).
        """
        _logger.info(
            "[4/?] Running prioritization (method=%s, order=%s, policy=%s)…",
            config.method,
            config.order,
            config.policy,
        )
        t0 = _time.monotonic()

        # Build extractor for Prioritize constructor
        extractor = build_extractor(ref_dc.extractor, ref_dc.transforms)  # type: ignore[arg-type]

        results: dict[str, tuple[list[int], list[float] | None]] = {}

        for name, (clean_emb, clean_to_orig, _orig_size) in add_clean_info.items():
            if len(clean_emb) == 0:
                _logger.warning("  %s: no items after cleaning, skipping", name)
                results[name] = ([], None)
                continue

            prioritizer = Prioritize(
                extractor=extractor,
                method=config.method,
                k=config.k,
                c=config.c,
                n_init=config.n_init,
                max_cluster_size=config.max_cluster_size,
                order=config.order,
                policy=config.policy,
                num_bins=config.num_bins,
                reference=clean_ref_embeddings,
            )
            p_result = prioritizer.evaluate(clean_emb)

            # Map clean-space indices back to original-space indices
            original_indices = [clean_to_orig[int(i)] for i in p_result.indices]
            scores: list[float] | None = None
            if p_result.scores is not None:
                scores = [float(s) for s in p_result.scores]

            results[name] = (original_indices, scores)
            _logger.info("  %s: %d items prioritized", name, len(original_indices))

        _logger.info("[4/?] Prioritization complete in %.1fs", _time.monotonic() - t0)
        return results

    def _build_workflow_result(
        self,
        config: DataPrioritizationConfig,
        ref_size: int,
        cleaning_summary: CleaningSummaryDict | None,
        total_removed: int,
        prioritization_results: dict[str, tuple[list[int], list[float] | None]],
        ref_clean_to_orig: list[int],
        add_clean_info: dict[str, tuple[NDArray[np.float32], list[int], int]],
    ) -> DataPrioritizationResult:
        """Build the final workflow result."""
        total_prioritized = sum(len(indices) for indices, _ in prioritization_results.values())

        prioritizations: list[PerDatasetPrioritizationDict] = []
        for name, (indices, scores) in prioritization_results.items():
            _, clean_to_orig, orig_size = add_clean_info[name]
            prioritizations.append(
                PerDatasetPrioritizationDict(
                    source_name=name,
                    original_size=orig_size,
                    cleaned_size=len(clean_to_orig),
                    prioritized_indices=indices,
                    scores=scores,
                )
            )

        raw = DataPrioritizationRawOutput(
            dataset_size=ref_size + sum(info[2] for info in add_clean_info.values()),
            reference_size=ref_size,
            method=config.method,
            order=config.order,
            policy=config.policy,
            cleaning_summary=cleaning_summary,
            prioritizations=prioritizations,
        )

        findings = build_findings(raw, config)

        summary = f"Prioritization complete. {total_prioritized} items ranked via {config.method}."

        report = DataPrioritizationReport(summary=summary, findings=findings)

        # Build metadata
        per_source_clean: dict[str, list[int]] = {}
        per_source_prioritized: dict[str, list[int]] = {}

        if config.mode == "preparatory":
            per_source_clean["__reference__"] = ref_clean_to_orig
            for name, (_, clean_to_orig, _) in add_clean_info.items():
                per_source_clean[name] = clean_to_orig
            for name, (indices, _) in prioritization_results.items():
                per_source_prioritized[name] = indices

        metadata = DataPrioritizationMetadata(
            mode=config.mode,
            method=config.method,
            order=config.order,
            policy=config.policy,
            cleaning_enabled=config.cleaning is not None,
            items_removed_by_cleaning=total_removed,
            per_source_clean_indices=per_source_clean,
            per_source_prioritized_indices=per_source_prioritized,
        )

        return DataPrioritizationResult(
            type=self.name,
            success=True,
            output=DataPrioritizationOutput(raw=raw, report=report),
            metadata=metadata,
        )
