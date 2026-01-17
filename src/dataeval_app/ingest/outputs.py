"""Workflow outputs - base classes and typed results."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Utility classes
# -----------------------------------------------------------------------------


class Reportable(BaseModel):
    """Human-readable report item."""

    report_type: Literal["table", "key_value", "image", "text"]
    title: str
    data: dict[str, Any] | list[dict[str, Any]] | str
    description: str | None = None


# -----------------------------------------------------------------------------
# Base classes
# -----------------------------------------------------------------------------


class WorkflowOutputsBase(BaseModel):
    """Base class for all workflow outputs."""

    dataset_size: int = Field(description="Number of items in dataset")


class WorkflowReportBase(BaseModel):
    """Base class for all workflow reports."""

    summary: str


# -----------------------------------------------------------------------------
# Data Cleaning workflow
# -----------------------------------------------------------------------------


class DataCleaningRawOutputs(WorkflowOutputsBase):
    """Machine-readable results from data cleaning workflow.

    Matches DataEval's data cleaning workflow outputs.
    """

    # Duplicate detection (always runs)
    duplicates: dict[str, Any] = Field(
        default_factory=dict,
        description="DuplicatesOutput from DataEval",
    )

    # Image outlier detection (always runs)
    img_outliers: dict[str, Any] = Field(
        default_factory=dict,
        description="OutliersOutput for images from DataEval",
    )

    # Label statistics (always runs)
    label_stats: dict[str, Any] = Field(
        default_factory=dict,
        description="LabelStatsOutput from DataEval",
    )

    # Target outliers (object detection only)
    target_outliers: dict[str, Any] | None = Field(
        default=None,
        description="OutliersOutput for bounding boxes (OD datasets only)",
    )


class DataCleaningReport(WorkflowReportBase):
    """Human-readable report for data cleaning workflow."""

    findings: list[Reportable] = Field(default_factory=list)


class DataCleaningOutputs(BaseModel):
    """Complete data cleaning workflow output."""

    raw: DataCleaningRawOutputs
    report: DataCleaningReport
