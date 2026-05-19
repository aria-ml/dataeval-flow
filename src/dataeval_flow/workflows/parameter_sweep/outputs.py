"""Parameter sweep workflow outputs."""

from typing import Any

from pydantic import BaseModel, Field

from dataeval_flow.config.schemas import ResultMetadata
from dataeval_flow.workflow.base import WorkflowOutputsBase, WorkflowReportBase

__all__ = [
    "ParameterSweepMetadata",
    "ParameterSweepOutputs",
    "ParameterSweepRawOutputs",
    "ParameterSweepReport",
    "SweepRunResult",
]


class SweepRunResult(BaseModel):
    """Result of a single run within the parameter sweep."""

    params: dict[str, Any] = Field(description="Parameters used for this run")
    outlier_count: int = Field(description="Number of outliers detected")
    exact_duplicate_groups: int = Field(description="Number of exact duplicate groups detected")
    near_duplicate_groups: int = Field(description="Number of near duplicate groups detected")


class ParameterSweepRawOutputs(BaseModel):
    """Raw outputs for parameter sweep workflow."""

    results: list[SweepRunResult] = Field(default_factory=list)


class ParameterSweepReport(WorkflowReportBase):
    """Report for parameter sweep workflow."""

    findings: list[Any] = Field(default_factory=list)


class ParameterSweepOutputs(WorkflowOutputsBase):
    """Outputs for parameter sweep workflow."""

    raw: ParameterSweepRawOutputs
    report: ParameterSweepReport


class ParameterSweepMetadata(ResultMetadata):
    """Metadata for parameter sweep workflow."""

    sweep_parameters: list[str] = Field(default_factory=list, description="List of parameters that were swept")
