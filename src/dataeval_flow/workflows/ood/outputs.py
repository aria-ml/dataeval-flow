"""OOD detection workflow outputs."""

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from pydantic import BaseModel, Field
from typing_extensions import TypedDict, TypeIs

from dataeval_flow.config import ResultMetadata
from dataeval_flow.workflow.base import Reportable, WorkflowOutputsBase, WorkflowReportBase

if TYPE_CHECKING:
    from dataeval_flow.workflow import WorkflowResult

__all__ = [
    "DetectorOODResultDict",
    "FactorDeviationDict",
    "OODDetectionMetadata",
    "OODDetectionOutputs",
    "OODDetectionRawOutputs",
    "OODDetectionReport",
    "OODDetectionResult",
    "OODSampleDict",
    "is_ood_result",
]


# ---------------------------------------------------------------------------
# TypedDicts for serialized detector outputs
# ---------------------------------------------------------------------------


class OODSampleDict(TypedDict):
    """Per-sample OOD result."""

    index: int
    score: float
    is_ood: bool


class _DetectorOODResultRequired(TypedDict):
    """Required fields for an OOD detector result."""

    method: str
    ood_count: int
    total_count: int
    ood_percentage: float
    threshold_score: float


class DetectorOODResultDict(_DetectorOODResultRequired, total=False):
    """Serialized result from a single OOD detector.

    ``samples`` contains per-sample scores and OOD flags for all test
    samples.
    """

    samples: list[OODSampleDict]


class FactorDeviationDict(TypedDict):
    """Per-sample metadata factor deviations for an OOD sample."""

    index: int
    deviations: dict[str, float]


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------


class OODDetectionRawOutputs(WorkflowOutputsBase):
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


class OODDetectionReport(WorkflowReportBase):
    """Human-readable report for OOD detection workflow."""

    findings: list[Reportable] = Field(default_factory=list)


class OODDetectionOutputs(BaseModel):
    """Complete OOD detection workflow output."""

    raw: OODDetectionRawOutputs
    report: OODDetectionReport


class OODDetectionMetadata(ResultMetadata):
    """Metadata for the ood-detection workflow."""

    mode: Literal["advisory", "preparatory"] = "advisory"
    detectors_used: list[str] = Field(default_factory=list)
    metadata_insights_enabled: bool = False


# ---------------------------------------------------------------------------
# Type alias and TypeIs guard for type narrowing
# ---------------------------------------------------------------------------

#: Fully typed result alias for the ood-detection workflow.
OODDetectionResult: TypeAlias = "WorkflowResult[OODDetectionMetadata, OODDetectionOutputs]"


def is_ood_result(
    result: "WorkflowResult[Any, Any]",
) -> TypeIs["WorkflowResult[OODDetectionMetadata, OODDetectionOutputs]"]:
    """Narrow a generic ``WorkflowResult`` to an OOD detection result."""
    return isinstance(result.metadata, OODDetectionMetadata)
