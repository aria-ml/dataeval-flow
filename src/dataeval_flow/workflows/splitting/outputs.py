"""Dataset splitting workflow outputs."""

from typing import TYPE_CHECKING, Any, TypeAlias

from pydantic import BaseModel, Field
from typing_extensions import TypeIs

from dataeval_flow.config.schemas import ResultMetadata
from dataeval_flow.workflow.base import Reportable, WorkflowOutputsBase, WorkflowReportBase

if TYPE_CHECKING:
    from dataeval_flow.workflow import WorkflowResult

__all__ = [
    "DataSplittingMetadata",
    "DataSplittingOutputs",
    "DataSplittingRawOutputs",
    "DataSplittingReport",
    "DataSplittingResult",
    "SplitInfo",
    "is_splitting_result",
]


# ---------------------------------------------------------------------------
# Raw outputs
# ---------------------------------------------------------------------------


class SplitInfo(BaseModel):
    """Per-fold split information."""

    fold: int
    train_indices: list[int]
    val_indices: list[int]
    label_stats_train: dict[str, Any] = Field(default_factory=dict)
    label_stats_val: dict[str, Any] = Field(default_factory=dict)
    coverage_train: dict[str, Any] | None = None
    coverage_val: dict[str, Any] | None = None


class DataSplittingRawOutputs(WorkflowOutputsBase):
    """Machine-readable splitting results."""

    pre_split_balance: dict[str, Any] = Field(default_factory=dict)
    pre_split_diversity: dict[str, Any] = Field(default_factory=dict)
    label_stats_full: dict[str, Any] = Field(default_factory=dict)
    test_indices: list[int] = Field(default_factory=list)
    label_stats_test: dict[str, Any] = Field(default_factory=dict)
    coverage_test: dict[str, Any] | None = None
    folds: list[SplitInfo] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class DataSplittingReport(WorkflowReportBase):
    """Human-readable splitting report."""

    findings: list[Reportable] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Composite output
# ---------------------------------------------------------------------------


class DataSplittingOutputs(BaseModel):
    """Composite output: raw results + human-readable report."""

    raw: DataSplittingRawOutputs
    report: DataSplittingReport


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class DataSplittingMetadata(ResultMetadata):
    """Splitting-specific metadata extending the JATIC envelope."""

    num_folds: int = 1
    stratified: bool = True
    split_on: list[str] | None = None
    rebalance_method: str | None = None
    split_sizes: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result alias and guard
# ---------------------------------------------------------------------------

DataSplittingResult: TypeAlias = "WorkflowResult[DataSplittingMetadata, DataSplittingOutputs]"


def is_splitting_result(result: "WorkflowResult[Any, Any]") -> TypeIs["DataSplittingResult"]:
    """Type guard for splitting workflow results."""
    return isinstance(result.metadata, DataSplittingMetadata)
