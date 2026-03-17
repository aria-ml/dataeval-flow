"""Drift monitoring workflow outputs."""

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from pydantic import BaseModel, Field
from typing_extensions import TypedDict, TypeIs

from dataeval_flow.config.schemas import ResultMetadata
from dataeval_flow.workflow.base import Reportable, WorkflowOutputsBase, WorkflowReportBase

if TYPE_CHECKING:
    from dataeval_flow.workflow import WorkflowResult

__all__ = [
    "ChunkResultDict",
    "ClasswiseDriftDict",
    "ClasswiseDriftRowDict",
    "DetectorResultDict",
    "DriftMonitoringMetadata",
    "DriftMonitoringOutputs",
    "DriftMonitoringRawOutputs",
    "DriftMonitoringReport",
    "DriftMonitoringResult",
    "is_drift_result",
]


# ---------------------------------------------------------------------------
# TypedDicts for serialized detector outputs
# ---------------------------------------------------------------------------


class ChunkResultDict(TypedDict):
    """Single chunk result from chunked drift analysis."""

    key: str  # e.g. "[0:100]"
    index: int
    start_index: int
    end_index: int
    value: float
    upper_threshold: float | None
    lower_threshold: float | None
    drifted: bool


class _DetectorResultRequired(TypedDict):
    """Required fields for a detector result."""

    method: str
    drifted: bool
    distance: float
    threshold: float
    metric_name: str


class DetectorResultDict(_DetectorResultRequired, total=False):
    """Serialized result from a single drift detector.

    ``details`` contains detector-specific statistics (p-values, per-feature
    flags, fold AUROCs, etc.).  ``chunks`` is present only when chunked
    analysis is enabled.
    """

    details: dict[str, Any]
    chunks: list[ChunkResultDict]


class ClasswiseDriftRowDict(TypedDict):
    """Per-class drift result for a single detector."""

    class_name: str
    drifted: bool
    distance: float
    p_val: float | None


class ClasswiseDriftDict(TypedDict):
    """Classwise drift results for one detector."""

    detector: str
    rows: list[ClasswiseDriftRowDict]


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------


class DriftMonitoringRawOutputs(WorkflowOutputsBase):
    """Machine-readable results from drift monitoring workflow."""

    reference_size: int = Field(
        default=0,
        description="Number of items in the reference dataset.",
    )
    test_size: int = Field(
        default=0,
        description="Number of items in the test dataset(s).",
    )
    detectors: dict[str, DetectorResultDict] = Field(
        default_factory=dict,
        description="Per-detector results keyed by method name.",
    )
    classwise: list[ClasswiseDriftDict] | None = Field(
        default=None,
        description="Per-class drift results (one entry per detector). None if classwise disabled.",
    )


class DriftMonitoringReport(WorkflowReportBase):
    """Human-readable report for drift monitoring workflow."""

    findings: list[Reportable] = Field(default_factory=list)


class DriftMonitoringOutputs(BaseModel):
    """Complete drift monitoring workflow output."""

    raw: DriftMonitoringRawOutputs
    report: DriftMonitoringReport


class DriftMonitoringMetadata(ResultMetadata):
    """Metadata for the drift-monitoring workflow."""

    mode: Literal["advisory", "preparatory"] = "advisory"
    detectors_used: list[str] = Field(default_factory=list)
    chunking_enabled: bool = False
    classwise_enabled: bool = False


# ---------------------------------------------------------------------------
# Type alias and TypeIs guard for type narrowing
# ---------------------------------------------------------------------------

#: Fully typed result alias for the drift-monitoring workflow.
DriftMonitoringResult: TypeAlias = "WorkflowResult[DriftMonitoringMetadata, DriftMonitoringOutputs]"


def is_drift_result(
    result: "WorkflowResult[Any, Any]",
) -> TypeIs["WorkflowResult[DriftMonitoringMetadata, DriftMonitoringOutputs]"]:
    """Narrow a generic ``WorkflowResult`` to a drift-monitoring result."""
    return isinstance(result.metadata, DriftMonitoringMetadata)
