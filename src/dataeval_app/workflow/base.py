"""Workflow base classes - shared by all concrete workflows."""

from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = [
    "Reportable",
    "WorkflowOutputsBase",
    "WorkflowParametersBase",
    "WorkflowReportBase",
]

# --- Parameter base ---


class WorkflowParametersBase(BaseModel):
    """Base class for all workflow parameters."""

    mode: Literal["advisory", "preparatory"] = Field(
        default="advisory",
        description="advisory: report only, preparatory: modify dataset",
    )


# --- Output bases ---


class Reportable(BaseModel):
    """Human-readable report item."""

    report_type: Literal["table", "key_value", "image", "text", "pivot_table"]
    severity: Literal["ok", "info", "warning"] = "info"
    title: str
    data: dict[str, Any] | list[dict[str, Any]] | str
    description: str | None = None


class WorkflowOutputsBase(BaseModel):
    """Base class for all workflow outputs."""

    dataset_size: int = Field(description="Number of items in dataset")


class WorkflowReportBase(BaseModel):
    """Base class for all workflow reports."""

    summary: str
