"""OOD detection workflow outputs."""

from typing import Literal, NotRequired

from pydantic import Field
from typing_extensions import TypedDict

from dataeval_flow._result import ResultMetadata
from dataeval_flow.workflows._base import WorkflowOutput, WorkflowRawOutput, WorkflowReport
from dataeval_flow.workflows._result import WorkflowResult

__all__ = [
    "DetectorOODResultDict",
    "FactorDeviationDict",
    "OODDetectionMetadata",
    "OODDetectionOutput",
    "OODDetectionRawOutput",
    "OODDetectionReport",
    "OODDetectionResult",
    "OODSampleDict",
]


# ---------------------------------------------------------------------------
# TypedDicts for serialized detector outputs
# ---------------------------------------------------------------------------


class OODSampleDict(TypedDict):
    """Per-sample OOD result."""

    index: int
    score: float
    is_ood: bool


class DetectorOODResultDict(TypedDict):
    """Serialized result from a single OOD detector.

    ``samples`` contains per-sample scores and OOD flags for all test
    samples.
    """

    method: str
    ood_count: int
    total_count: int
    ood_percentage: float
    threshold_score: float
    samples: NotRequired[list[OODSampleDict]]


class FactorDeviationDict(TypedDict):
    """Per-sample metadata factor deviations for an OOD sample."""

    index: int
    deviations: dict[str, float]


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------


class OODDetectionRawOutput(WorkflowRawOutput):
    """Machine-readable results from OOD detection workflow."""

    reference_size: int = Field(
        default=0,
        description="Number of items in the reference dataset.",
    )
    test_size: int = Field(
        default=0,
        description="Number of items in the test dataset(s).",
    )
    detectors: dict[str, DetectorOODResultDict] = Field(
        default_factory=dict,
        description="Per-detector results keyed by method name.",
    )
    ood_indices: list[int] = Field(
        default_factory=list,
        description="Union of OOD sample indices across all detectors.",
    )
    factor_deviations: list[FactorDeviationDict] | None = Field(
        default=None,
        description="Per-OOD-sample metadata factor deviations. None if metadata insights disabled.",
    )
    factor_predictors: dict[str, float] | None = Field(
        default=None,
        description="Mutual information (bits) per metadata factor with OOD status. None if insights disabled.",
    )


class OODDetectionReport(WorkflowReport):
    """Human-readable report for OOD detection workflow."""


class OODDetectionOutput(WorkflowOutput[OODDetectionRawOutput, OODDetectionReport]):
    """Complete OOD detection workflow output."""


class OODDetectionMetadata(ResultMetadata):
    """Metadata for the ood-detection workflow."""

    mode: Literal["advisory", "preparatory"] = Field(
        default="advisory", description="The `mode` the workflow ran in, as configured: `advisory` or `preparatory`."
    )
    detectors_used: list[str] = Field(
        default_factory=list, description="The detectors that produced results, as keyed in `detectors`."
    )
    metadata_insights_enabled: bool = Field(
        default=False, description="Whether metadata insights ran: configured, with at least one sample flagged OOD."
    )


class OODDetectionResult(WorkflowResult[OODDetectionMetadata, OODDetectionOutput]):
    """The result of an ``ood-detection`` run.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. Every workflow result also carries
    ``output.raw.dataset_size`` (:class:`~dataeval_flow.workflows.WorkflowRawOutput`), ``output.report.summary`` and
    ``output.report.findings`` (:class:`~dataeval_flow.workflows.WorkflowReport`), and the envelope fields of
    :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output.raw.reference_size
        Number of items in the reference dataset.
    output.raw.test_size
        Number of items in the test dataset(s).
    output.raw.detectors
        Per-detector results keyed by method name.
    output.raw.ood_indices
        Union of OOD sample indices across all detectors.
    output.raw.factor_deviations
        Per-OOD-sample metadata factor deviations. None if metadata insights disabled.
    output.raw.factor_predictors
        Mutual information (bits) per metadata factor with OOD status. None if insights disabled.
    metadata.mode
        The ``mode`` the workflow ran in, as configured: ``advisory`` or ``preparatory``.
    metadata.detectors_used
        The detectors that produced results, as keyed in ``detectors``.
    metadata.metadata_insights_enabled
        Whether metadata insights ran: configured, with at least one sample flagged OOD.
    """
