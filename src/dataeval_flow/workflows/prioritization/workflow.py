"""Data prioritization workflow."""

from __future__ import annotations

import contextlib
import logging
import time as _time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from dataeval.flags import ImageStats
from dataeval.protocols import AnnotatedDataset
from dataeval.quality import Duplicates, Outliers
from dataeval.scope import Prioritize
from numpy.typing import NDArray
from pydantic import BaseModel

from dataeval_flow.cache import (
    active_cache,
    get_or_compute_embeddings,
    get_or_compute_stats,
    selection_repr,
)
from dataeval_flow.embeddings import build_extractor
from dataeval_flow.workflow import DatasetContext, WorkflowContext, WorkflowProtocol, WorkflowResult
from dataeval_flow.workflows.prioritization.outputs import (
    CleaningSummaryDict,
    DataPrioritizationMetadata,
    DataPrioritizationOutputs,
    DataPrioritizationRawOutputs,
    DataPrioritizationReport,
    PerDatasetPrioritizationDict,
)
from dataeval_flow.workflows.prioritization.params import (
    CleaningConfig,
    DataPrioritizationParameters,
)
from dataeval_flow.workflows.prioritization.report import build_findings

logger: logging.Logger = logging.getLogger(__name__)


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

_OUTLIER_FLAG_MAP: dict[str, ImageStats] = {
    "dimension": ImageStats.DIMENSION,
    "pixel": ImageStats.PIXEL,
    "visual": ImageStats.VISUAL,
}

_HASH_FLAG_MAP: dict[str, ImageStats] = {
    "hash_basic": ImageStats.HASH_DUPLICATES_BASIC,
    "hash_d4": ImageStats.HASH_DUPLICATES_D4,
}


def _resolve_cleaning_flags(
    cleaning: CleaningConfig,
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
    cleaning: CleaningConfig,
    outlier_flags: ImageStats,
    hash_flags: ImageStats,
    dc: DatasetContext,
    dataset: AnnotatedDataset[Any],
) -> set[int]:
    """Run stats-based outlier detection on a single dataset.

    Returns the set of flagged item indices.
    """
    sel_key = selection_repr(dataset)
    with contextlib.ExitStack() as stack:
        if dc.cache is not None:
            stack.enter_context(active_cache(dc.cache, sel_key))
        calc_result = get_or_compute_stats(
            desired_flags=outlier_flags | hash_flags,
            dataset=dataset,
        )

    outliers_eval = Outliers(
        flags=outlier_flags,
        outlier_threshold=(cleaning.outlier_method, cleaning.outlier_threshold),
    )
    outlier_output = outliers_eval.from_stats(calc_result)  # type: ignore[arg-type]
    return set(outlier_output.outliers.keys())


def _run_duplicate_detection_cross_dataset(
    cleaning: CleaningConfig,
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
    cleaning: CleaningConfig,
    ref_dc: DatasetContext,
    ref_dataset: AnnotatedDataset[Any],
    add_datasets: list[tuple[str, DatasetContext, AnnotatedDataset[Any]]],
) -> tuple[
    dict[str, set[int]],  # per-source flagged indices
    CleaningSummaryDict,
]:
    """Run the optional cleaning step across all datasets.

    Returns per-source flagged indices and a summary.
    """
    logger.info("[3/?] Running pre-prioritization cleaning…")
    t0 = _time.monotonic()

    outlier_flags, hash_flags = _resolve_cleaning_flags(cleaning)

    # --- Per-dataset outlier detection ---
    all_sources = [("__reference__", ref_dc, ref_dataset)] + list(add_datasets)
    flagged_outliers: dict[str, set[int]] = {}
    total_outliers = 0
    for name, dc, ds in all_sources:
        flagged = _run_outlier_detection_per_source(cleaning, outlier_flags, hash_flags, dc, ds)
        flagged_outliers[name] = flagged
        total_outliers += len(flagged)
        logger.info("  Outliers in %s: %d", name, len(flagged))

    # --- Cross-dataset duplicate detection ---
    dup_flagged = _run_duplicate_detection_cross_dataset(cleaning, hash_flags, all_sources)
    flagged_duplicates: dict[str, set[int]] = {}
    total_duplicates = 0
    for i, (name, _, _) in enumerate(all_sources):
        ds_flagged = dup_flagged.get(i, set())
        flagged_duplicates[name] = ds_flagged
        total_duplicates += len(ds_flagged)
        if ds_flagged:
            logger.info("  Duplicates in %s: %d", name, len(ds_flagged))

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

    logger.info(
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


class DataPrioritizationWorkflow(WorkflowProtocol[DataPrioritizationMetadata, DataPrioritizationOutputs]):
    """Data prioritization workflow using DataEval Prioritize."""

    @property
    def name(self) -> str:
        """Workflow identifier used in configs and task routing."""
        return "data-prioritization"

    @property
    def description(self) -> str:
        """Human-readable description of the workflow."""
        return "Prioritize unlabeled data for labeling based on a reference dataset and optional cleaning"

    @property
    def params_schema(self) -> type[DataPrioritizationParameters]:
        """Pydantic model for workflow parameters."""
        return DataPrioritizationParameters

    @property
    def output_schema(self) -> type[DataPrioritizationOutputs]:
        """Pydantic model for workflow output."""
        return DataPrioritizationOutputs

    def execute(
        self,
        context: WorkflowContext,
        params: BaseModel | None = None,
    ) -> WorkflowResult[DataPrioritizationMetadata, DataPrioritizationOutputs]:
        """Run the data-prioritization workflow."""
        if not isinstance(context, WorkflowContext):
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=[f"Expected WorkflowContext, got {type(context).__name__}"],
                metadata=DataPrioritizationMetadata(),
            )

        if params is None:
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=["DataPrioritizationParameters required"],
                metadata=DataPrioritizationMetadata(),
            )

        if not isinstance(params, DataPrioritizationParameters):
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=[f"Expected DataPrioritizationParameters, got {type(params).__name__}"],
                metadata=DataPrioritizationMetadata(),
            )

        try:
            return self._run(context, params)
        except Exception as e:
            logger.exception("Workflow '%s' failed", self.name)
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=[f"Workflow execution failed: {e}"],
                metadata=DataPrioritizationMetadata(),
            )

    def _run(
        self,
        context: WorkflowContext,
        params: DataPrioritizationParameters,
    ) -> WorkflowResult[DataPrioritizationMetadata, DataPrioritizationOutputs]:
        # --- 1. Validate: need 2+ datasets and an extractor ---
        dc_items = list(context.dataset_contexts.items())
        if len(dc_items) < 2:
            return WorkflowResult(
                name=self.name,
                success=False,
                data=self._empty_outputs(),
                errors=[
                    f"Data prioritization requires at least 2 datasets (reference + data to prioritize), "
                    f"got {len(dc_items)}: {[n for n, _ in dc_items]}"
                ],
                metadata=DataPrioritizationMetadata(),
            )

        # --- 2. Prepare datasets ---
        ref_dc, ref_dataset, add_datasets = self._prepare_datasets(dc_items)

        # --- 3. Extract embeddings ---
        ref_embeddings, add_embeddings = self._extract_all_embeddings(ref_dc, ref_dataset, add_datasets)

        # --- 4. Optional cleaning ---
        cleaning_summary: CleaningSummaryDict | None = None
        per_source_flagged: dict[str, set[int]] = {}
        total_removed = 0

        if params.cleaning is not None:
            per_source_flagged, cleaning_summary = _run_cleaning(params.cleaning, ref_dc, ref_dataset, add_datasets)
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
        prioritization_results = self._run_prioritization(params, ref_dc, clean_ref_embeddings, add_clean_info)

        # --- 7. Build outputs ---
        return self._build_workflow_result(
            params,
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
        from dataeval_flow.selection import build_selection

        ref_name, ref_dc = dc_items[0]
        add_contexts = dc_items[1:]

        logger.info(
            "[1/?] Preparing datasets: reference=%s, additional=%s",
            ref_name,
            [n for n, _ in add_contexts],
        )

        ref_dataset: AnnotatedDataset[Any] = ref_dc.dataset
        if ref_dc.selection_steps:
            ref_dataset = build_selection(ref_dataset, ref_dc.selection_steps)  # type: ignore[arg-type]

        add_datasets: list[tuple[str, DatasetContext, AnnotatedDataset[Any]]] = []
        for a_name, a_dc in add_contexts:
            a_ds = a_dc.dataset
            if a_dc.selection_steps:
                a_ds = build_selection(a_ds, a_dc.selection_steps)  # type: ignore[arg-type]
            add_datasets.append((a_name, a_dc, a_ds))

        return ref_dc, ref_dataset, add_datasets

    def _extract_all_embeddings(
        self,
        ref_dc: DatasetContext,
        ref_dataset: AnnotatedDataset[Any],
        add_datasets: list[tuple[str, DatasetContext, AnnotatedDataset[Any]]],
    ) -> tuple[NDArray[np.float32], dict[str, NDArray[np.float32]]]:
        """Extract embeddings for reference and additional datasets."""
        logger.info("[2/?] Extracting embeddings…")
        t0 = _time.monotonic()

        ref_embeddings = _get_embeddings_for_context(ref_dc, ref_dataset)
        logger.info("  Reference embeddings: %s", ref_embeddings.shape)

        add_embeddings: dict[str, NDArray[np.float32]] = {}
        for a_name, a_dc, a_ds in add_datasets:
            emb = _get_embeddings_for_context(a_dc, a_ds)
            add_embeddings[a_name] = emb
            logger.info("  Additional embeddings (%s): %s", a_name, emb.shape)

        logger.info("[2/?] Embeddings ready in %.1fs", _time.monotonic() - t0)
        return ref_embeddings, add_embeddings

    def _run_prioritization(
        self,
        params: DataPrioritizationParameters,
        ref_dc: DatasetContext,
        clean_ref_embeddings: NDArray[np.float32],
        add_clean_info: dict[str, tuple[NDArray[np.float32], list[int], int]],
    ) -> dict[str, tuple[list[int], list[float] | None]]:
        """Run prioritization for each additional dataset.

        Returns a mapping of source_name -> (original_indices, scores).
        """
        logger.info(
            "[4/?] Running prioritization (method=%s, order=%s, policy=%s)…",
            params.method,
            params.order,
            params.policy,
        )
        t0 = _time.monotonic()

        # Build extractor for Prioritize constructor
        extractor = build_extractor(ref_dc.extractor, ref_dc.transforms)  # type: ignore[arg-type]

        results: dict[str, tuple[list[int], list[float] | None]] = {}

        for name, (clean_emb, clean_to_orig, _orig_size) in add_clean_info.items():
            if len(clean_emb) == 0:
                logger.warning("  %s: no items after cleaning, skipping", name)
                results[name] = ([], None)
                continue

            prioritizer = Prioritize(
                extractor=extractor,
                method=params.method,
                k=params.k,
                c=params.c,
                n_init=params.n_init,
                max_cluster_size=params.max_cluster_size,
                order=params.order,
                policy=params.policy,
                num_bins=params.num_bins,
                reference=clean_ref_embeddings,
            )
            p_result = prioritizer.evaluate(clean_emb)

            # Map clean-space indices back to original-space indices
            original_indices = [clean_to_orig[int(i)] for i in p_result.indices]
            scores: list[float] | None = None
            if p_result.scores is not None:
                scores = [float(s) for s in p_result.scores]

            results[name] = (original_indices, scores)
            logger.info("  %s: %d items prioritized", name, len(original_indices))

        logger.info("[4/?] Prioritization complete in %.1fs", _time.monotonic() - t0)
        return results

    def _build_workflow_result(
        self,
        params: DataPrioritizationParameters,
        ref_size: int,
        cleaning_summary: CleaningSummaryDict | None,
        total_removed: int,
        prioritization_results: dict[str, tuple[list[int], list[float] | None]],
        ref_clean_to_orig: list[int],
        add_clean_info: dict[str, tuple[NDArray[np.float32], list[int], int]],
    ) -> WorkflowResult[DataPrioritizationMetadata, DataPrioritizationOutputs]:
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

        raw = DataPrioritizationRawOutputs(
            dataset_size=ref_size + sum(info[2] for info in add_clean_info.values()),
            reference_size=ref_size,
            method=params.method,
            order=params.order,
            policy=params.policy,
            cleaning_summary=cleaning_summary,
            prioritizations=prioritizations,
        )

        findings = build_findings(raw, params)

        summary = f"Prioritization complete. {total_prioritized} items ranked via {params.method}."

        report = DataPrioritizationReport(summary=summary, findings=findings)

        # Build metadata
        per_source_clean: dict[str, list[int]] = {}
        per_source_prioritized: dict[str, list[int]] = {}

        if params.mode == "preparatory":
            per_source_clean["__reference__"] = ref_clean_to_orig
            for name, (_, clean_to_orig, _) in add_clean_info.items():
                per_source_clean[name] = clean_to_orig
            for name, (indices, _) in prioritization_results.items():
                per_source_prioritized[name] = indices

        metadata = DataPrioritizationMetadata(
            mode=params.mode,
            method=params.method,
            order=params.order,
            policy=params.policy,
            cleaning_enabled=params.cleaning is not None,
            items_removed_by_cleaning=total_removed,
            per_source_clean_indices=per_source_clean,
            per_source_prioritized_indices=per_source_prioritized,
        )

        return WorkflowResult(
            name=self.name,
            success=True,
            data=DataPrioritizationOutputs(raw=raw, report=report),
            metadata=metadata,
        )

    def _empty_outputs(self) -> DataPrioritizationOutputs:
        return DataPrioritizationOutputs(
            raw=DataPrioritizationRawOutputs(dataset_size=0),
            report=DataPrioritizationReport(summary="Workflow failed", findings=[]),
        )
