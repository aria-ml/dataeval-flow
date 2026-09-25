"""Drift monitoring workflow outputs."""

from typing import Any, Literal, NotRequired

from pydantic import Field
from typing_extensions import TypedDict

from dataeval_flow._result import ResultMetadata
from dataeval_flow.workflows._base import WorkflowOutput, WorkflowRawOutput, WorkflowReport
from dataeval_flow.workflows._result import WorkflowResult

__all__ = [
    "ChunkResultDict",
    "ClasswiseDriftDict",
    "ClasswiseDriftRowDict",
    "DetectorResultDict",
    "DriftMonitoringMetadata",
    "DriftMonitoringOutput",
    "DriftMonitoringRawOutput",
    "DriftMonitoringReport",
    "DriftMonitoringResult",
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


class DetectorResultDict(TypedDict):
    """Serialized result from a single drift detector.

    ``details`` contains detector-specific statistics (p-values, per-feature
    flags, fold AUROCs, etc.).  ``chunks`` is present only when chunked
    analysis is enabled.
    """

    method: str
    drifted: bool
    distance: float
    threshold: float
    metric_name: str
    details: NotRequired[dict[str, Any]]
    chunks: NotRequired[list[ChunkResultDict]]


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


class DriftMonitoringRawOutput(WorkflowRawOutput):
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


class DriftMonitoringReport(WorkflowReport):
    """Human-readable report for drift monitoring workflow."""


class DriftMonitoringOutput(WorkflowOutput[DriftMonitoringRawOutput, DriftMonitoringReport]):
    """Complete drift monitoring workflow output."""


class DriftMonitoringMetadata(ResultMetadata):
    """Metadata for the drift-monitoring workflow."""

    mode: Literal["advisory", "preparatory"] = Field(
        default="advisory", description="The `mode` the workflow ran in, as configured: `advisory` or `preparatory`."
    )
    detectors_used: list[str] = Field(
        default_factory=list, description="The detectors that produced results, as keyed in `detectors`."
    )
    chunking_enabled: bool = Field(default=False, description="Whether any detector ran chunked.")
    classwise_enabled: bool = Field(default=False, description="Whether any detector also ran per class.")


class DriftMonitoringResult(WorkflowResult[DriftMonitoringMetadata, DriftMonitoringOutput]):
    """The result of a ``drift-monitoring`` run.

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
    output.raw.classwise
        Per-class drift results (one entry per detector). None if classwise disabled.
    metadata.mode
        The ``mode`` the workflow ran in, as configured: ``advisory`` or ``preparatory``.
    metadata.detectors_used
        The detectors that produced results, as keyed in ``detectors``.
    metadata.chunking_enabled
        Whether any detector ran chunked.
    metadata.classwise_enabled
        Whether any detector also ran per class.
    """
