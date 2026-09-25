"""Dataset splitting workflow outputs."""

from typing import Any

from pydantic import BaseModel, Field

from dataeval_flow._result import ResultMetadata
from dataeval_flow.workflows._base import WorkflowOutput, WorkflowRawOutput, WorkflowReport
from dataeval_flow.workflows._result import WorkflowResult

__all__ = [
    "DataSplittingMetadata",
    "DataSplittingOutput",
    "DataSplittingRawOutput",
    "DataSplittingReport",
    "DataSplittingResult",
    "SplitInfo",
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


class DataSplittingRawOutput(WorkflowRawOutput):
    """Machine-readable splitting results."""

    pre_split_balance: dict[str, Any] = Field(
        default_factory=dict,
        description="DataEval's `Balance` over the whole dataset: its `balance`, `factors` and `classwise` rows.",
    )
    pre_split_diversity: dict[str, Any] = Field(
        default_factory=dict,
        description="DataEval's `Diversity` over the whole dataset: its `factors` and `classwise` rows.",
    )
    label_stats_full: dict[str, Any] = Field(
        default_factory=dict,
        description="Label statistics of the whole dataset: per-class label and image counts, totals, `index2label`.",
    )
    test_indices: list[int] = Field(
        default_factory=list, description="Indices of the items held out for the test split."
    )
    label_stats_test: dict[str, Any] = Field(
        default_factory=dict,
        description="Label statistics of the test split, shaped as `label_stats_full`. Empty without a test split.",
    )
    coverage_test: dict[str, Any] | None = Field(
        default=None, description="Coverage of the test split. None unless the task names an extractor."
    )
    folds: list[SplitInfo] = Field(
        default_factory=list,
        description=(
            "One entry per fold: its `train_indices` and `val_indices`, their label statistics and, with an "
            "extractor, their coverage."
        ),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class DataSplittingReport(WorkflowReport):
    """Human-readable splitting report."""


# ---------------------------------------------------------------------------
# Composite output
# ---------------------------------------------------------------------------


class DataSplittingOutput(WorkflowOutput[DataSplittingRawOutput, DataSplittingReport]):
    """Composite output: raw results + human-readable report."""


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class DataSplittingMetadata(ResultMetadata):
    """Splitting-specific metadata extending the JATIC envelope."""

    num_folds: int = Field(default=1, description="How many train/validation folds the run made.")
    stratified: bool = Field(default=True, description="Whether each split preserves the class distribution.")
    split_on: list[str] | None = Field(
        default=None, description="Metadata keys grouped on so no group spans two splits. None where there were none."
    )
    rebalance_method: str | None = Field(
        default=None, description="How each fold's train split was class-rebalanced. None where it was not."
    )
    split_sizes: dict[str, int] = Field(
        default_factory=dict, description="Item counts of the first fold's `train` and `val` splits and of `test`."
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


class DataSplittingResult(WorkflowResult[DataSplittingMetadata, DataSplittingOutput]):
    """The result of a ``data-splitting`` run.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. Every workflow result also carries
    ``output.raw.dataset_size`` (:class:`~dataeval_flow.workflows.WorkflowRawOutput`), ``output.report.summary`` and
    ``output.report.findings`` (:class:`~dataeval_flow.workflows.WorkflowReport`), and the envelope fields of
    :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output.raw.pre_split_balance
        DataEval's ``Balance`` over the whole dataset: its ``balance``, ``factors`` and ``classwise`` rows.
    output.raw.pre_split_diversity
        DataEval's ``Diversity`` over the whole dataset: its ``factors`` and ``classwise`` rows.
    output.raw.label_stats_full
        Label statistics of the whole dataset: per-class label and image counts, totals, ``index2label``.
    output.raw.test_indices
        Indices of the items held out for the test split.
    output.raw.label_stats_test
        Label statistics of the test split, shaped as ``label_stats_full``. Empty without a test split.
    output.raw.coverage_test
        Coverage of the test split. None unless the task names an extractor.
    output.raw.folds
        One entry per fold: its ``train_indices`` and ``val_indices``, their label statistics and, with an extractor,
        their coverage.
    metadata.num_folds
        How many train/validation folds the run made.
    metadata.stratified
        Whether each split preserves the class distribution.
    metadata.split_on
        Metadata keys grouped on so no group spans two splits. None where there were none.
    metadata.rebalance_method
        How each fold's train split was class-rebalanced. None where it was not.
    metadata.split_sizes
        Item counts of the first fold's ``train`` and ``val`` splits and of ``test``.
    """
