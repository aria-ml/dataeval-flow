"""Data prioritization workflow outputs."""

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from pydantic import BaseModel, Field
from typing_extensions import TypedDict, TypeIs

from dataeval_flow.config import ResultMetadata
from dataeval_flow.workflow.base import Reportable, WorkflowOutputsBase, WorkflowReportBase

if TYPE_CHECKING:
    from dataeval_flow.workflow import WorkflowResult

__all__ = [
    "CleaningSummaryDict",
    "DataPrioritizationMetadata",
    "DataPrioritizationOutputs",
    "DataPrioritizationRawOutputs",
    "DataPrioritizationReport",
    "DataPrioritizationResult",
    "PerDatasetPrioritizationDict",
    "is_prioritization_result",
]


# ---------------------------------------------------------------------------
# TypedDicts for serialized outputs
# ---------------------------------------------------------------------------


class PerDatasetPrioritizationDict(TypedDict):
    """Prioritization results for a single additional dataset."""

    source_name: str
    original_size: int
    cleaned_size: int
    prioritized_indices: list[int]
    scores: list[float] | None


class CleaningSummaryDict(TypedDict):
    """Summary of the cleaning step (combined across all datasets)."""

    total_combined: int
    outliers_flagged: int
    duplicates_flagged: int
    total_removed: int


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------


class DataPrioritizationRawOutputs(WorkflowOutputsBase):
    """Machine-readable results from the data-prioritization workflow."""

    reference_size: int = Field(
        default=0,
        description="Number of items in the reference dataset.",
    )
    method: str = Field(
        default="",
        description="Ranking method used.",
    )
    order: str = Field(
        default="",
        description="Sort direction used.",
    )
    policy: str = Field(
        default="",
        description="Selection policy used.",
    )
    cleaning_summary: CleaningSummaryDict | None = Field(
        default=None,
        description="Cleaning step summary. None if cleaning was skipped.",
    )
    prioritizations: list[PerDatasetPrioritizationDict] = Field(
        default_factory=list,
        description="Per-dataset prioritization results.",
    )


class DataPrioritizationReport(WorkflowReportBase):
    """Human-readable report for the data-prioritization workflow."""

    findings: list[Reportable] = Field(default_factory=list)


class DataPrioritizationOutputs(BaseModel):
    """Complete data-prioritization workflow output."""

    raw: DataPrioritizationRawOutputs
    report: DataPrioritizationReport


class DataPrioritizationMetadata(ResultMetadata):
    """Metadata for the data-prioritization workflow."""

    mode: Literal["advisory", "preparatory"] = "advisory"
    method: str = ""
    order: str = ""
    policy: str = ""
    cleaning_enabled: bool = False
    items_removed_by_cleaning: int = 0
    per_source_clean_indices: dict[str, list[int]] = Field(default_factory=dict)
    per_source_prioritized_indices: dict[str, list[int]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Type alias and TypeIs guard for type narrowing
# ---------------------------------------------------------------------------

#: Fully typed result alias for the data-prioritization workflow.
DataPrioritizationResult: TypeAlias = "WorkflowResult[DataPrioritizationMetadata, DataPrioritizationOutputs]"


def is_prioritization_result(
    result: "WorkflowResult[Any, Any]",
) -> TypeIs["WorkflowResult[DataPrioritizationMetadata, DataPrioritizationOutputs]"]:
    """Narrow a generic ``WorkflowResult`` to a data-prioritization result."""
    return isinstance(result.metadata, DataPrioritizationMetadata)
