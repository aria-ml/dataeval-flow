"""Dataset splitting workflow implementation."""

import logging
import traceback
from typing import Any

import numpy as np
from pydantic import BaseModel

from dataeval_flow.workflow import WorkflowContext, WorkflowResult
from dataeval_flow.workflows.splitting.outputs import (
    DataSplittingMetadata,
    DataSplittingOutputs,
    DataSplittingRawOutputs,
    DataSplittingReport,
    SplitInfo,
)
from dataeval_flow.workflows.splitting.params import DataSplittingParameters
from dataeval_flow.workflows.splitting.report import build_findings

__all__ = ["DataSplittingWorkflow"]

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_label_stats(stats: Any) -> dict[str, Any]:
    """Convert a LabelStatsResult to a plain dict."""
    if stats is None:
        return {}
    result: dict[str, Any] = {}
    for key in (
        "label_counts_per_class",
        "image_counts_per_class",
        "class_count",
        "label_count",
        "image_count",
        "index2label",
    ):
        val = stats.get(key, None) if hasattr(stats, "get") else getattr(stats, key, None)
        if val is not None:
            if hasattr(val, "tolist"):
                val = val.tolist()
            result[key] = val
    return result


def _serialize_balance(output: Any) -> dict[str, Any]:
    """Convert BalanceOutput to a plain dict."""
    result: dict[str, Any] = {}
    for attr in ("balance", "factors", "classwise"):
        df = getattr(output, attr, None)
        if df is not None:
            result[attr] = df.to_dicts() if hasattr(df, "to_dicts") else str(df)
    return result


def _serialize_diversity(output: Any) -> dict[str, Any]:
    """Convert DiversityOutput to a plain dict."""
    result: dict[str, Any] = {}
    for attr in ("factors", "classwise"):
        df = getattr(output, attr, None)
        if df is not None:
            result[attr] = df.to_dicts() if hasattr(df, "to_dicts") else str(df)
    return result


def _serialize_coverage(coverage_result: Any) -> dict[str, Any]:
    """Convert CoverageResult to a plain dict."""
    result: dict[str, Any] = {}
    for key in ("uncovered_indices", "critical_value_radii", "coverage_radius"):
        val = coverage_result.get(key, None) if hasattr(coverage_result, "get") else getattr(coverage_result, key, None)
        if val is not None:
            if hasattr(val, "tolist"):
                val = val.tolist()
            result[key] = val
    return result


# ---------------------------------------------------------------------------
# Embedding-space coverage helper (step 7)
# Extracted from _execute to satisfy C901 complexity limit.
# "Coverage" here refers to dataeval.core.coverage_adaptive(), not test coverage.
# ---------------------------------------------------------------------------


def _run_coverage(
    ds_ctx: Any,
    dataset: Any,
    fold_infos: list[SplitInfo],
    test_indices: list[int],
    params: DataSplittingParameters,
) -> dict[str, Any] | None:
    """Run per-split coverage assessment if model is provided."""
    if ds_ctx.extractor is None:
        logger.info("Step 7: Skipping coverage (no model provided)")
        return None

    from dataeval.core import coverage_adaptive

    from dataeval_flow.embeddings import build_embeddings

    logger.info("Step 7: Per-split coverage assessment")
    embeddings_obj = build_embeddings(
        dataset,
        ds_ctx.extractor,
        transforms=ds_ctx.transforms,
        batch_size=ds_ctx.batch_size,
    )
    all_embeddings = np.array(embeddings_obj)

    # Normalize to [0, 1]
    emb_min = all_embeddings.min(axis=0, keepdims=True)
    emb_max = all_embeddings.max(axis=0, keepdims=True)
    emb_range = emb_max - emb_min
    emb_range[emb_range == 0] = 1.0
    all_embeddings = (all_embeddings - emb_min) / emb_range

    for fold_info in fold_infos:
        train_embs = all_embeddings[fold_info.train_indices]
        val_embs = all_embeddings[fold_info.val_indices]
        fold_info.coverage_train = _serialize_coverage(
            coverage_adaptive(train_embs, params.num_observations, params.coverage_percent)
        )
        fold_info.coverage_val = _serialize_coverage(
            coverage_adaptive(val_embs, params.num_observations, params.coverage_percent)
        )

    coverage_test_data: dict[str, Any] | None = None
    if test_indices:
        test_embs = all_embeddings[test_indices]
        coverage_test_data = _serialize_coverage(
            coverage_adaptive(test_embs, params.num_observations, params.coverage_percent)
        )
    return coverage_test_data


# ---------------------------------------------------------------------------
# Workflow class
# ---------------------------------------------------------------------------


class DataSplittingWorkflow:
    """Dataset splitting workflow.

    Assesses dataset balance/diversity, produces stratified train/val/test
    splits, and optionally rebalances the train split.
    """

    @property
    def name(self) -> str:
        """Workflow identifier."""
        return "data-splitting"

    @property
    def description(self) -> str:
        """Human-readable description."""
        return (
            "Assess dataset balance/diversity, produce stratified train/val/test "
            "splits, and optionally rebalance the train split."
        )

    @property
    def params_schema(self) -> type[BaseModel]:
        """Pydantic model for workflow parameters."""
        return DataSplittingParameters

    @property
    def output_schema(self) -> type[BaseModel]:
        """Pydantic model for workflow output."""
        return DataSplittingOutputs

    def execute(
        self,
        context: WorkflowContext,
        params: BaseModel | None = None,
    ) -> WorkflowResult[DataSplittingMetadata, DataSplittingOutputs]:
        """Execute the splitting workflow."""
        if not isinstance(context, WorkflowContext):
            msg = f"Expected WorkflowContext, got {type(context).__name__}"
            raise TypeError(msg)

        if params is not None and not isinstance(params, DataSplittingParameters):
            msg = f"Expected DataSplittingParameters, got {type(params).__name__}"
            raise TypeError(msg)

        p = params or DataSplittingParameters()

        try:
            return self._execute(context, p)
        except Exception:
            logger.exception("Splitting workflow failed")
            return WorkflowResult(
                name=self.name,
                success=False,
                data=_empty_outputs(),
                errors=[traceback.format_exc()],
                metadata=DataSplittingMetadata(),
            )

    def _execute(
        self,
        context: WorkflowContext,
        params: DataSplittingParameters,
    ) -> WorkflowResult[DataSplittingMetadata, DataSplittingOutputs]:
        from dataeval.bias import Balance, Diversity
        from dataeval.core import label_stats
        from dataeval.utils.data import split_dataset

        from dataeval_flow.metadata import build_metadata
        from dataeval_flow.selection import build_selection

        # --- Resolve single dataset ---
        if not context.dataset_contexts:
            msg = "No datasets provided"
            raise ValueError(msg)

        ds_name = next(iter(context.dataset_contexts))
        ds_ctx = context.dataset_contexts[ds_name]
        dataset: Any = ds_ctx.dataset

        # Apply selection if configured
        if ds_ctx.selection_steps:
            dataset = build_selection(dataset, list(ds_ctx.selection_steps))

        dataset_size = len(dataset)
        logger.info("Step 1: Building metadata for %s (%d items)", ds_name, dataset_size)

        # --- Step 1: Build Metadata ---
        metadata = build_metadata(dataset)

        # --- Step 2: Pre-split bias assessment ---
        logger.info("Step 2: Pre-split bias assessment")
        balance_output = Balance().evaluate(metadata)
        diversity_output = Diversity().evaluate(metadata)

        pre_split_balance = _serialize_balance(balance_output)
        pre_split_diversity = _serialize_diversity(diversity_output)

        # --- Step 3: Full-dataset label stats ---
        logger.info("Step 3: Label statistics (full dataset)")
        class_labels = metadata.class_labels
        index2label = metadata.index2label if hasattr(metadata, "index2label") else None
        full_stats = label_stats(class_labels, index2label=index2label)
        label_stats_full = _serialize_label_stats(full_stats)

        # --- Step 4: Split ---
        logger.info(
            "Step 4: Splitting dataset (num_folds=%d, stratify=%s, test_frac=%s, val_frac=%s)",
            params.num_folds,
            params.stratify,
            params.test_frac,
            params.val_frac,
        )
        # Use metadata if split_on is specified, otherwise use dataset directly
        split_input: Any = metadata if params.split_on else dataset
        splits = split_dataset(
            split_input,
            num_folds=params.num_folds,
            stratify=params.stratify,
            split_on=params.split_on,
            test_frac=params.test_frac,
            val_frac=params.val_frac,
        )

        test_indices = splits.test.tolist()

        # --- Step 5: Optional rebalancing ---
        # ClassBalance is a selection step that operates on a Select-wrapped dataset.
        # For now we store the raw indices; rebalancing modifies train indices.
        fold_infos: list[SplitInfo] = []
        for i, fold in enumerate(splits.folds):
            train_idx = fold.train.tolist()
            val_idx = fold.val.tolist()

            if params.rebalance_method is not None:
                logger.info("Step 5: Rebalancing fold %d train split (method=%s)", i, params.rebalance_method)
                from dataeval.selection import ClassBalance, Indices, Select

                train_selected = Select(dataset, Indices(train_idx))
                ClassBalance(method=params.rebalance_method)(train_selected)
                train_idx = train_selected.resolve_indices()

            fold_infos.append(
                SplitInfo(
                    fold=i,
                    train_indices=train_idx,
                    val_indices=val_idx,
                )
            )

        # --- Step 6: Per-split label stats ---
        logger.info("Step 6: Per-split label statistics")
        for fold_info in fold_infos:
            train_labels = class_labels[fold_info.train_indices]
            val_labels = class_labels[fold_info.val_indices]
            fold_info.label_stats_train = _serialize_label_stats(label_stats(train_labels, index2label=index2label))
            fold_info.label_stats_val = _serialize_label_stats(label_stats(val_labels, index2label=index2label))

        test_labels = class_labels[test_indices] if test_indices else np.array([], dtype=np.intp)
        label_stats_test = (
            _serialize_label_stats(label_stats(test_labels, index2label=index2label)) if len(test_labels) > 0 else {}
        )

        # --- Step 7: Per-split coverage (if model provided) ---
        coverage_test_data = _run_coverage(
            ds_ctx,
            dataset,
            fold_infos,
            test_indices,
            params,
        )

        # --- Build raw outputs ---
        raw = DataSplittingRawOutputs(
            dataset_size=dataset_size,
            pre_split_balance=pre_split_balance,
            pre_split_diversity=pre_split_diversity,
            label_stats_full=label_stats_full,
            test_indices=test_indices,
            label_stats_test=label_stats_test,
            coverage_test=coverage_test_data,
            folds=fold_infos,
        )

        # --- Build findings ---
        findings = build_findings(raw)

        # --- Build split sizes for metadata ---
        fold0 = fold_infos[0] if fold_infos else None
        split_sizes = {
            "train": len(fold0.train_indices) if fold0 else 0,
            "val": len(fold0.val_indices) if fold0 else 0,
            "test": len(test_indices),
        }

        report = DataSplittingReport(
            summary=f"Dataset splitting: {dataset_size} items → {len(fold_infos)} fold(s)",
            findings=findings,
        )

        outputs = DataSplittingOutputs(raw=raw, report=report)

        metadata = DataSplittingMetadata(
            num_folds=params.num_folds,
            stratified=params.stratify,
            split_on=params.split_on,
            rebalance_method=params.rebalance_method,
            split_sizes=split_sizes,
        )

        return WorkflowResult(
            name=self.name,
            success=True,
            data=outputs,
            metadata=metadata,
        )


def _empty_outputs() -> DataSplittingOutputs:
    """Create empty outputs for error cases."""
    return DataSplittingOutputs(
        raw=DataSplittingRawOutputs(dataset_size=0),
        report=DataSplittingReport(summary="Splitting workflow failed"),
    )
