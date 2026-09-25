"""Data prioritization workflow outputs."""

from typing import Literal

from pydantic import Field
from typing_extensions import TypedDict

from dataeval_flow._result import ResultMetadata
from dataeval_flow.workflows._base import WorkflowOutput, WorkflowRawOutput, WorkflowReport
from dataeval_flow.workflows._result import WorkflowResult

__all__ = [
    "CleaningSummaryDict",
    "DataPrioritizationMetadata",
    "DataPrioritizationOutput",
    "DataPrioritizationRawOutput",
    "DataPrioritizationReport",
    "DataPrioritizationResult",
    "PerDatasetPrioritizationDict",
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


class DataPrioritizationRawOutput(WorkflowRawOutput):
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


class DataPrioritizationReport(WorkflowReport):
    """Human-readable report for the data-prioritization workflow."""


class DataPrioritizationOutput(WorkflowOutput[DataPrioritizationRawOutput, DataPrioritizationReport]):
    """Complete data-prioritization workflow output."""


class DataPrioritizationMetadata(ResultMetadata):
    """Metadata for the data-prioritization workflow."""

    mode: Literal["advisory", "preparatory"] = Field(
        default="advisory", description="The `mode` the workflow ran in, as configured: `advisory` or `preparatory`."
    )
    method: str = Field(default="", description="Ranking method used.")
    order: str = Field(default="", description="Sort direction used.")
    policy: str = Field(default="", description="Selection policy used.")
    cleaning_enabled: bool = Field(default=False, description="Whether the task cleaned the sources before ranking.")
    items_removed_by_cleaning: int = Field(
        default=0, description="How many items cleaning removed, across every source."
    )
    per_source_clean_indices: dict[str, list[int]] = Field(
        default_factory=dict,
        description=(
            "Per source, the indices of the items cleaning kept, the reference under `__reference__`. Empty unless "
            "`mode` is `preparatory`."
        ),
    )
    per_source_prioritized_indices: dict[str, list[int]] = Field(
        default_factory=dict,
        description=(
            "Per source ranked against the reference, its item indices in priority order. Empty unless `mode` is "
            "`preparatory`."
        ),
    )


class DataPrioritizationResult(WorkflowResult[DataPrioritizationMetadata, DataPrioritizationOutput]):
    """The result of a ``data-prioritization`` run.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. Every workflow result also carries
    ``output.raw.dataset_size`` (:class:`~dataeval_flow.workflows.WorkflowRawOutput`), ``output.report.summary`` and
    ``output.report.findings`` (:class:`~dataeval_flow.workflows.WorkflowReport`), and the envelope fields of
    :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output.raw.reference_size
        Number of items in the reference dataset.
    output.raw.method
        Ranking method used.
    output.raw.order
        Sort direction used.
    output.raw.policy
        Selection policy used.
    output.raw.cleaning_summary
        Cleaning step summary. None if cleaning was skipped.
    output.raw.prioritizations
        Per-dataset prioritization results.
    metadata.mode
        The ``mode`` the workflow ran in, as configured: ``advisory`` or ``preparatory``.
    metadata.method
        Ranking method used.
    metadata.order
        Sort direction used.
    metadata.policy
        Selection policy used.
    metadata.cleaning_enabled
        Whether the task cleaned the sources before ranking.
    metadata.items_removed_by_cleaning
        How many items cleaning removed, across every source.
    metadata.per_source_clean_indices
        Per source, the indices of the items cleaning kept, the reference under ``__reference__``. Empty unless ``mode``
        is ``preparatory``.
    metadata.per_source_prioritized_indices
        Per source ranked against the reference, its item indices in priority order. Empty unless ``mode`` is
        ``preparatory``.
    """
