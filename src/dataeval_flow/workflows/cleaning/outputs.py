"""Data cleaning workflow outputs."""

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from pydantic import BaseModel, Field
from typing_extensions import TypedDict, TypeIs

from dataeval_flow.config.schemas import ResultMetadata
from dataeval_flow.workflow.base import Reportable, WorkflowOutputsBase, WorkflowReportBase

if TYPE_CHECKING:
    from dataeval_flow.workflow import WorkflowResult

__all__ = [
    "ClasswisePivotDict",
    "DataCleaningMetadata",
    "DataCleaningOutputs",
    "DataCleaningRawOutputs",
    "DataCleaningReport",
    "DataCleaningResult",
    "DetectionDict",
    "DuplicatesDict",
    "LabelStatsDict",
    "NearDuplicateGroupDict",
    "OutlierIssueRecord",
    "OutlierIssuesDict",
    "SourceIndexDict",
    "is_cleaning_result",
]


# ---------------------------------------------------------------------------
# TypedDicts for serialized evaluator outputs
# ---------------------------------------------------------------------------


class _OutlierIssueRecordRequired(TypedDict):
    """Required fields for an outlier issue record."""

    item_index: int
    metric_name: str
    metric_value: float


class OutlierIssueRecord(_OutlierIssueRecordRequired, total=False):
    """Single outlier issue from DataEval OutliersOutput.

    ``target_index`` is present for target-level outliers (object detection datasets)
    and absent or ``None`` for image-level outliers.
    """

    target_index: int | None


class OutlierIssuesDict(TypedDict):
    """Serialized outlier issues (image or target level)."""

    issues: list[OutlierIssueRecord]
    count: int


class SourceIndexDict(TypedDict):
    """Serialized SourceIndex from DataEval — identifies an item, target, and channel."""

    item: int
    target: int | None
    channel: int | None


#: An index value is either a plain ``int`` (image-level) or a
#: :class:`SourceIndexDict` (target/channel-level).
IndexValue = int | SourceIndexDict


class NearDuplicateGroupDict(TypedDict):
    """Serialized near-duplicate group."""

    indices: list[IndexValue]
    methods: list[str]
    orientation: str | None


class DetectionDict(TypedDict, total=False):
    """Serialized duplicate detection result (exact + near groups)."""

    exact: list[list[IndexValue]]
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


class ClasswiseRowDict(TypedDict):
    """Single row in the classwise outlier summary."""

    class_name: str
    count: int
    pct: float  # percentage of that class's labels flagged


class ClasswisePivotDict(TypedDict, total=False):
    """Classwise outlier summary — count and % of labels flagged per class.

    For classification datasets this summarises image outliers per class;
    for object-detection datasets this summarises target-level outliers per
    class (image-level entries are excluded since they cannot be attributed
    to a single class).
    """

    level: str  # "image" or "target"
    rows: list[ClasswiseRowDict]  # one per class + Total row


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
    classwise_outliers: ClasswisePivotDict | None = Field(
        default=None,
        description="Classwise outlier pivot — image-level for classification, target-level for OD",
    )


class DataCleaningReport(WorkflowReportBase):
    """Human-readable report for data cleaning workflow."""

    findings: list[Reportable] = Field(default_factory=list)


class DataCleaningOutputs(BaseModel):
    """Complete data cleaning workflow output."""

    raw: DataCleaningRawOutputs
    report: DataCleaningReport


class DataCleaningMetadata(ResultMetadata):
    """Metadata for the data-cleaning workflow."""

    mode: Literal["advisory", "preparatory"] = "advisory"
    evaluators: list[str] = Field(default_factory=list)
    flagged_indices: list[int] = Field(default_factory=list)
    clean_indices: list[int] = Field(default_factory=list)
    removed_count: int = 0


# ---------------------------------------------------------------------------
# Type alias and TypeIs guard for type narrowing
# ---------------------------------------------------------------------------

#: Fully typed result alias for the data-cleaning workflow.
DataCleaningResult: TypeAlias = "WorkflowResult[DataCleaningMetadata, DataCleaningOutputs]"


def is_cleaning_result(
    result: "WorkflowResult[Any, Any]",
) -> TypeIs["WorkflowResult[DataCleaningMetadata, DataCleaningOutputs]"]:
    """Narrow a generic ``WorkflowResult`` to a data-cleaning result.

    Useful in the CLI loop or any code that receives a generic result::

        [result] = run_tasks(config, "my_task")
        if is_cleaning_result(result):
            result.metadata.flagged_indices  # ✓ typed
            result.data.raw.img_outliers     # ✓ typed
    """
    return isinstance(result.metadata, DataCleaningMetadata)
