"""Workflow base classes - shared by all concrete workflows."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from dataeval_flow.config.schemas import AutoBinMethod

__all__ = [
    "MetadataConfigMixin",
    "Reportable",
    "WorkflowOutputsBase",
    "WorkflowParametersBase",
    "WorkflowReportBase",
]


# --- Metadata configuration mixin ---


class MetadataConfigMixin(BaseModel):
    """Mixin for workflows that need dataset metadata configuration.

    Provides binning and exclusion settings for metadata analysis.
    Mix into workflow parameter classes that require metadata processing
    (e.g. data cleaning).
    """

    metadata_auto_bin_method: AutoBinMethod | None = None
    metadata_exclude: Sequence[str] = Field(default_factory=list)
    metadata_continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = None


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

    report_type: Literal["table", "key_value", "image", "text", "pivot_table", "chunk_table", "classwise_table"]
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
