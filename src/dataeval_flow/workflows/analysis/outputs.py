"""Data analysis workflow outputs."""

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from pydantic import BaseModel, Field
from typing_extensions import TypeIs

from dataeval_flow.config.schemas import ResultMetadata
from dataeval_flow.workflow.base import Reportable, WorkflowOutputsBase, WorkflowReportBase

if TYPE_CHECKING:
    from dataeval_flow.workflow import WorkflowResult

__all__ = [
    "BiasResult",
    "CrossSplitLabelHealth",
    "CrossSplitRedundancy",
    "CrossSplitResult",
    "DataAnalysisMetadata",
    "DataAnalysisOutputs",
    "DataAnalysisRawOutputs",
    "DataAnalysisReport",
    "DataAnalysisResult",
    "DistributionShiftResult",
    "ImageQualityResult",
    "LabelHealthResult",
    "RedundancyResult",
    "SplitResult",
    "is_analysis_result",
]


# ---------------------------------------------------------------------------
# Per-split assessment sub-models
# ---------------------------------------------------------------------------


class ImageQualityResult(BaseModel):
    """Image-level anomaly detection results."""

    outlier_count: int = Field(description="Number of items flagged as outliers")
    outlier_rate: float = Field(description="Fraction of items flagged as outliers")
    outlier_summary: dict[str, int] = Field(description="Count of outlier items per metric name")


class RedundancyResult(BaseModel):
    """Duplicate and near-duplicate detection results."""

    exact_duplicate_groups: int = Field(description="Number of exact duplicate groups found")
    near_duplicate_groups: int = Field(description="Number of near duplicate groups found")
    exact_duplicates_count: int = Field(description="Total items across all exact duplicate groups")
    near_duplicates_count: int = Field(description="Total items across all near duplicate groups")
    exact_groups: list[list[int]] = Field(default_factory=list, description="Indices per exact duplicate group")
    near_groups: list[list[int]] = Field(default_factory=list, description="Indices per near duplicate group")


class LabelHealthResult(BaseModel):
    """Label completeness and distribution results."""

    num_classes: int = Field(description="Number of unique class labels")
    class_distribution: dict[str, int] = Field(description="Mapping of class name to label count")
    empty_images: list[int] = Field(description="Indices of images with no labels")


class BiasResult(BaseModel):
    """Metadata bias analysis results."""

    metadata_factors: list[str] = Field(description="Names of metadata factors present")
    metadata_summary: dict[str, dict[str, Any]] = Field(description="Per-factor summary statistics")
    balance_summary: dict[str, Any] | None = Field(default=None, description="Balance (MI) analysis results")
    diversity_summary: dict[str, Any] | None = Field(default=None, description="Diversity analysis results")


# ---------------------------------------------------------------------------
# Per-split result (composed from assessment sub-models)
# ---------------------------------------------------------------------------


class SplitResult(BaseModel):
    """Per-split summary statistics grouped by assessment area."""

    num_samples: int = Field(description="Total number of images in the split")
    image_quality: ImageQualityResult = Field(description="Image-level anomaly detection")
    redundancy: RedundancyResult = Field(description="Duplicate detection")
    label_health: LabelHealthResult = Field(description="Label completeness and distribution")
    bias: BiasResult = Field(description="Metadata bias analysis")


# ---------------------------------------------------------------------------
# Cross-split assessment sub-models
# ---------------------------------------------------------------------------


class CrossSplitRedundancy(BaseModel):
    """Cross-split duplicate leakage detection."""

    duplicate_leakage: dict[str, Any] = Field(
        default_factory=lambda: {"exact_count": 0, "near_count": 0, "exact_groups": [], "near_groups": []},
        description="Cross-split duplicate detection results",
    )


class CrossSplitLabelHealth(BaseModel):
    """Cross-split label comparison results."""

    label_overlap: dict[str, Any] = Field(
        description="Class-level comparison including shared classes and proportion differences"
    )
    label_parity: dict[str, Any] | None = Field(
        default=None,
        description="Chi-squared label distribution parity test between splits",
    )


class DistributionShiftResult(BaseModel):
    """Cross-split embedding divergence results."""

    divergence: float | None = Field(
        default=None,
        description="Embedding-space divergence between splits, or None if no extractor was provided",
    )
    divergence_method: str | None = Field(
        default=None,
        description='Method used for divergence computation ("mst" or "fnn")',
    )


# ---------------------------------------------------------------------------
# Cross-split result (composed from assessment sub-models)
# ---------------------------------------------------------------------------


class CrossSplitResult(BaseModel):
    """Comparison results between two dataset splits."""

    redundancy: CrossSplitRedundancy = Field(description="Cross-split duplicate leakage")
    label_health: CrossSplitLabelHealth = Field(description="Label distribution comparison")
    distribution_shift: DistributionShiftResult = Field(description="Embedding divergence")


# ---------------------------------------------------------------------------
# Workflow output models
# ---------------------------------------------------------------------------


class DataAnalysisRawOutputs(WorkflowOutputsBase):
    """Machine-readable results from data analysis workflow.

    ``dataset_size`` is the total number of items across all splits.
    """

    splits: dict[str, SplitResult] = Field(
        default_factory=dict,
        description="Per-split quality summaries keyed by split name",
    )
    cross_split: dict[str, CrossSplitResult] = Field(
        default_factory=dict,
        description='Pairwise cross-split comparisons keyed by "splitA_vs_splitB"',
    )


class DataAnalysisReport(WorkflowReportBase):
    """Human-readable report for data analysis workflow."""

    findings: list[Reportable] = Field(default_factory=list)


class DataAnalysisOutputs(BaseModel):
    """Complete data analysis workflow output."""

    raw: DataAnalysisRawOutputs
    report: DataAnalysisReport


class DataAnalysisMetadata(ResultMetadata):
    """Metadata for the data-analysis workflow."""

    mode: Literal["advisory", "preparatory"] = "advisory"
    split_names: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Type alias and TypeIs guard for type narrowing
# ---------------------------------------------------------------------------

#: Fully typed result alias for the data-analysis workflow.
DataAnalysisResult: TypeAlias = "WorkflowResult[DataAnalysisMetadata, DataAnalysisOutputs]"


def is_analysis_result(
    result: "WorkflowResult[Any, Any]",
) -> TypeIs["WorkflowResult[DataAnalysisMetadata, DataAnalysisOutputs]"]:
    """Narrow a generic ``WorkflowResult`` to a data-analysis result.

    Useful in the CLI loop or any code that receives a generic result::

        [result] = run_tasks(config, "my_task")
        if is_analysis_result(result):
            result.metadata.split_names   # ✓ typed
            result.data.raw.splits        # ✓ typed
    """
    return isinstance(result.metadata, DataAnalysisMetadata)
