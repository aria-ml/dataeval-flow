"""Data Cleaning Workflow — orchestration + processor + factory helpers."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Protocol

from dataeval import Embeddings, Metadata
from dataeval.flags import ImageStats
from dataeval.quality import Duplicates, Outliers
from pydantic import BaseModel

from dataeval_app.workflow import WorkflowContext, WorkflowResult
from dataeval_app.workflow.base import Reportable
from dataeval_app.workflows.cleaning.outputs import (
    DataCleaningMetadata,
    DataCleaningOutputs,
    DataCleaningRawOutputs,
    DataCleaningReport,
    DetectionDict,
    DuplicatesDict,
    LabelStatsDict,
    OutlierIssuesDict,
)
from dataeval_app.workflows.cleaning.params import DataCleaningParameters

if TYPE_CHECKING:
    import polars as pl
    from dataeval.quality import DuplicatesOutput

    from dataeval_app.dataset import MaiteDataset

__all__ = ["DataCleaningWorkflow"]


# ---------------------------------------------------------------------------
# Protocols for private DataEval types
# ---------------------------------------------------------------------------


class _NearDuplicateGroup(Protocol):
    @property
    def indices(self) -> Sequence[int]: ...
    @property
    def methods(self) -> frozenset[str]: ...
    @property
    def orientation(self) -> Literal["rotated", "same"] | None: ...


class _DetectionResult(Protocol):
    @property
    def exact(self) -> Sequence[Sequence[int]] | None: ...
    @property
    def near(self) -> Sequence[_NearDuplicateGroup] | None: ...


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
    extractor: Embeddings | None = None,
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
    extractor: Embeddings | None = None,
) -> Duplicates:
    """Build Duplicates evaluator from cleaning parameters."""
    # Build hash flags
    flags = ImageStats.NONE
    if params.duplicate_flags is not None:
        for name in params.duplicate_flags:
            flags |= HASH_FLAG_MAP[name]

    # Validate cluster params require an extractor
    has_cluster = (
        params.duplicate_cluster_threshold is not None
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
        "cluster_threshold": params.duplicate_cluster_threshold,
        "cluster_algorithm": params.duplicate_cluster_algorithm,
        "n_clusters": params.duplicate_n_clusters,
        "extractor": extractor,
    }
    if params.duplicate_flags is not None:
        kwargs["flags"] = flags

    return Duplicates(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Result serialization helpers
# ---------------------------------------------------------------------------


def _serialize_detection(det: _DetectionResult) -> "DetectionDict":
    """Serialize a DuplicateDetectionResult to plain dict."""
    out: DetectionDict = {}
    if det.exact is not None:
        out["exact"] = [list(group) for group in det.exact]
    if det.near is not None:
        out["near"] = [
            {
                "indices": list(g.indices),
                "methods": sorted(g.methods),
                "orientation": g.orientation,
            }
            for g in det.near
        ]
    return out


def _serialize_outlier_issues(issues: "pl.DataFrame") -> "OutlierIssuesDict":
    """Serialize outlier issues Polars DataFrame to plain dict.

    Parameters
    ----------
    issues : polars.DataFrame
        Polars DataFrame with columns: item_id, metric_name, metric_value,
        and optionally target_id.
    """
    return {
        # to_dicts() returns list[dict[str, Any]]; rows match OutlierIssueRecord shape at runtime.
        "issues": issues.to_dicts(),  # type: ignore[typeddict-item]
        "count": len(issues),
    }


def _serialize_duplicates(result: "DuplicatesOutput") -> "DuplicatesDict":
    """Serialize DuplicatesOutput to plain dict.

    DuplicatesOutput has items and targets fields, each a
    DuplicateDetectionResult with exact and near groups.
    """
    # DuplicateDetectionResult is generic; our _DetectionResult Protocol matches
    # the concrete .exact/.near attributes at runtime.
    return {
        "items": _serialize_detection(result.items),  # type: ignore[arg-type]
        "targets": _serialize_detection(result.targets),  # type: ignore[arg-type]
    }


def _compute_label_stats(metadata: Metadata) -> "LabelStatsDict":
    """Compute label statistics from Metadata instance."""
    class_labels = metadata.class_labels
    index2label = metadata.index2label

    # Count labels per class
    label_counts: dict[str, int] = {}
    for label_idx in class_labels:
        name = index2label.get(label_idx, str(label_idx))
        label_counts[name] = label_counts.get(name, 0) + 1

    return {
        "item_count": metadata.item_count,
        "class_count": len(index2label),
        "index2label": dict(index2label),
        "label_counts_per_class": label_counts,
    }


# ---------------------------------------------------------------------------
# Findings generation
# ---------------------------------------------------------------------------


def _build_findings(
    raw: DataCleaningRawOutputs,
    metadata: Metadata | None,  # noqa: ARG001 - reserved for future metadata-based findings
) -> list[Reportable]:
    """Generate human-readable findings from raw results."""
    findings: list[Reportable] = []

    # Outlier findings
    outlier_count = raw.img_outliers.get("count", 0)
    if outlier_count > 0:
        pct = (outlier_count / raw.dataset_size) * 100 if raw.dataset_size else 0
        findings.append(
            Reportable(
                report_type="key_value",
                title="Image Outliers",
                data={"count": outlier_count, "percentage": round(pct, 1)},
                description=f"{outlier_count} images ({pct:.1f}%) flagged as outliers.",
            )
        )

    # Target outlier findings
    target_count = raw.target_outliers.get("count", 0) if raw.target_outliers else 0
    if target_count > 0:
        findings.append(
            Reportable(
                report_type="key_value",
                title="Target Outliers",
                data={"count": target_count},
                description=f"{target_count} bounding-box targets flagged as outliers.",
            )
        )

    # Duplicate findings
    exact_groups = raw.duplicates.get("items", {}).get("exact", [])
    near_groups = raw.duplicates.get("items", {}).get("near", [])
    if exact_groups or near_groups:
        findings.append(
            Reportable(
                report_type="key_value",
                title="Duplicates",
                data={
                    "exact_groups": len(exact_groups),
                    "near_groups": len(near_groups),
                },
                description=(
                    f"{len(exact_groups)} exact duplicate groups, {len(near_groups)} near-duplicate groups found."
                ),
            )
        )

    # Label distribution finding
    if raw.label_stats:
        findings.append(
            Reportable(
                report_type="table",
                title="Label Distribution",
                data=raw.label_stats.get("label_counts_per_class", {}),
                description=(
                    f"{raw.label_stats.get('class_count', '?')} classes, "
                    f"{raw.label_stats.get('item_count', '?')} items."
                ),
            )
        )

    return findings


def _collect_flagged_indices(raw: DataCleaningRawOutputs) -> set[int]:
    """Collect all unique item indices flagged by outlier or duplicate detection."""
    flagged: set[int] = set()

    # Outlier-flagged items
    for issue in raw.img_outliers.get("issues", []):
        flagged.add(issue["item_id"])

    # Duplicate-flagged items (keep first in each group, flag the rest)
    for group in raw.duplicates.get("items", {}).get("exact", []):
        for idx in group[1:]:  # keep first, flag rest
            flagged.add(idx)
    for group in raw.duplicates.get("items", {}).get("near", []):
        for idx in group["indices"][1:]:
            flagged.add(idx)

    return flagged


# ---------------------------------------------------------------------------
# Processor (internal — not exported)
# ---------------------------------------------------------------------------


def _run_cleaning(
    dataset: "MaiteDataset",
    params: DataCleaningParameters,
    embeddings: Embeddings | None = None,
    metadata: Metadata | None = None,
) -> DataCleaningRawOutputs:
    """Run outlier + duplicate detection on dataset."""
    outliers = _build_outliers(params, extractor=embeddings)
    duplicates = _build_duplicates(params, extractor=embeddings)

    # MaiteDataset conforms to DataEval's dataset protocol at runtime (duck typing);
    # pyright can't verify cross-library structural conformance.
    outliers_result = outliers.evaluate(dataset, per_image=True, per_target=True)  # type: ignore[arg-type]

    # Split image vs target outliers from the issues DataFrame.
    # Note: target_id column is omitted when all outliers are image-level
    # (e.g. classification datasets with no bounding boxes).
    issues_df = outliers_result.issues
    if "target_id" in issues_df.columns:
        img_issues = issues_df.filter(issues_df["target_id"].is_null())
        target_issues = issues_df.filter(issues_df["target_id"].is_not_null())
    else:
        img_issues = issues_df
        target_issues = None

    duplicates_result = duplicates.evaluate(dataset)  # type: ignore[arg-type]  # same reason as above

    # Compute label stats from Metadata (if available)
    # Empty dict is valid at runtime; LabelStatsDict uses total=False so all keys are optional.
    label_stats: LabelStatsDict = _compute_label_stats(metadata) if metadata else {}  # type: ignore[assignment]

    return DataCleaningRawOutputs(
        dataset_size=len(dataset),
        img_outliers=_serialize_outlier_issues(img_issues),
        target_outliers=_serialize_outlier_issues(target_issues)
        if target_issues is not None and len(target_issues) > 0
        else None,
        duplicates=_serialize_duplicates(duplicates_result),
        label_stats=label_stats,
    )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class DataCleaningWorkflow:
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
    ) -> WorkflowResult[DataCleaningMetadata]:
        """Run data cleaning workflow on dataset."""
        from dataeval_app.embeddings import build_embeddings
        from dataeval_app.metadata import build_metadata
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

            # Resolve the single dataset context (cleaning is single-dataset)
            dc = next(iter(context.dataset_contexts.values()))

            # 1. Apply selection if configured
            dataset = dc.dataset
            if dc.selection_steps:
                dataset = build_selection(dataset, dc.selection_steps)  # type: ignore[arg-type]

            # 2. Build embeddings if extractor configured
            embeddings = None
            if dc.extractor:
                embeddings = build_embeddings(
                    dataset,  # type: ignore[arg-type]
                    extractor_config=dc.extractor,
                    transforms=dc.transforms,
                )

            # 3. Build metadata for label stats
            metadata = build_metadata(
                dataset,  # type: ignore[arg-type]
                auto_bin_method=context.metadata_auto_bin_method,
                exclude=context.metadata_exclude or None,
                continuous_factor_bins=context.metadata_continuous_factor_bins,
            )

            # 4. Run cleaning evaluators
            raw = _run_cleaning(dataset, params, embeddings, metadata)  # type: ignore[arg-type]

            # 5. Generate findings from raw results
            findings = _build_findings(raw, metadata)

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
        except Exception as e:  # noqa: BLE001
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
