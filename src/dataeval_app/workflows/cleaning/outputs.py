"""Data cleaning workflow outputs."""

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from dataeval_app.workflow.base import Reportable, WorkflowOutputsBase, WorkflowReportBase

__all__ = [
    "DataCleaningOutputs",
    "DataCleaningRawOutputs",
    "DataCleaningReport",
    "DetectionDict",
    "DuplicatesDict",
    "LabelStatsDict",
    "NearDuplicateGroupDict",
    "OutlierIssueRecord",
    "OutlierIssuesDict",
]


# ---------------------------------------------------------------------------
# TypedDicts for serialized evaluator outputs
# ---------------------------------------------------------------------------


class OutlierIssueRecord(TypedDict):
    """Single outlier issue from DataEval OutliersOutput."""

    item_id: int
    metric_name: str
    metric_value: float


class OutlierIssuesDict(TypedDict):
    """Serialized outlier issues (image or target level)."""

    issues: list[OutlierIssueRecord]
    count: int


class NearDuplicateGroupDict(TypedDict):
    """Serialized near-duplicate group."""

    indices: list[int]
    methods: list[str]
    orientation: str | None


class DetectionDict(TypedDict, total=False):
    """Serialized duplicate detection result (exact + near groups)."""

    exact: list[list[int]]
    near: list[NearDuplicateGroupDict]


class DuplicatesDict(TypedDict):
    """Serialized DuplicatesOutput (items + targets)."""

    items: DetectionDict
    targets: DetectionDict


class LabelStatsDict(TypedDict, total=False):
    """Label statistics derived from Metadata."""

    item_count: int
    class_count: int
    index2label: dict[int, str]
    label_counts_per_class: dict[str, int]


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------


class DataCleaningRawOutputs(WorkflowOutputsBase):
    """Machine-readable results from data cleaning workflow."""

    duplicates: DuplicatesDict = Field(
        default_factory=lambda: {"items": {}, "targets": {}},
        description="DuplicatesOutput from DataEval",
    )
    img_outliers: OutlierIssuesDict = Field(
        default_factory=lambda: {"issues": [], "count": 0},
        description="OutliersOutput for images from DataEval",
    )
    label_stats: LabelStatsDict = Field(
        default_factory=dict,  # type: ignore[assignment]  # empty dict valid; all LabelStatsDict keys are optional (total=False)
        description="Label statistics derived from Metadata (class_labels, index2label, item_count)",
    )
    target_outliers: OutlierIssuesDict | None = Field(
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
