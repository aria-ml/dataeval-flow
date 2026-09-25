"""Parameter sweep workflow outputs."""

from typing import Any

from pydantic import BaseModel, Field

from dataeval_flow._result import ResultMetadata
from dataeval_flow.workflows._base import WorkflowOutput, WorkflowRawOutput, WorkflowReport
from dataeval_flow.workflows._result import WorkflowResult

__all__ = [
    "ParameterSweepMetadata",
    "ParameterSweepOutput",
    "ParameterSweepRawOutput",
    "ParameterSweepReport",
    "ParameterSweepResult",
    "SweepRunResult",
]


class SweepRunResult(BaseModel):
    """Result of a single run within the parameter sweep."""

    params: dict[str, Any] = Field(description="Parameters used for this run")
    outlier_count: int = Field(description="Number of outliers detected")
    exact_duplicate_groups: int = Field(description="Number of exact duplicate groups detected")
    near_duplicate_groups: int = Field(description="Number of near duplicate groups detected")


class ParameterSweepRawOutput(WorkflowRawOutput):
    """Raw outputs for parameter sweep workflow."""

    results: list[SweepRunResult] = Field(
        default_factory=list,
        description="One entry per combination tried: its `params` and the outlier and duplicate group counts found.",
    )


class ParameterSweepReport(WorkflowReport):
    """Report for parameter sweep workflow."""


class ParameterSweepOutput(WorkflowOutput[ParameterSweepRawOutput, ParameterSweepReport]):
    """Outputs for parameter sweep workflow."""


class ParameterSweepMetadata(ResultMetadata):
    """Metadata for parameter sweep workflow."""

    sweep_parameters: list[str] = Field(default_factory=list, description="List of parameters that were swept")


class ParameterSweepResult(WorkflowResult[ParameterSweepMetadata, ParameterSweepOutput]):
    """The result of a ``parameter-sweep`` run.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. Every workflow result also carries
    ``output.raw.dataset_size`` (:class:`~dataeval_flow.workflows.WorkflowRawOutput`), ``output.report.summary`` and
    ``output.report.findings`` (:class:`~dataeval_flow.workflows.WorkflowReport`), and the envelope fields of
    :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output.raw.results
        One entry per combination tried: its ``params`` and the outlier and duplicate group counts found.
    metadata.sweep_parameters
        List of parameters that were swept
    """
