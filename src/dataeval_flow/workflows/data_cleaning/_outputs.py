"""Data cleaning workflow outputs."""

from typing import Literal, NotRequired

from pydantic import Field
from typing_extensions import TypedDict

from dataeval_flow._result import ResultMetadata
from dataeval_flow.workflows._base import WorkflowOutput, WorkflowRawOutput, WorkflowReport
from dataeval_flow.workflows._result import WorkflowResult

__all__ = [
    "ClasswisePivotDict",
    "DataCleaningMetadata",
    "DataCleaningOutput",
    "DataCleaningRawOutput",
    "DataCleaningReport",
    "DataCleaningResult",
    "DetectionDict",
    "DuplicatesDict",
    "LabelStatsDict",
    "NearDuplicateGroupDict",
    "OutlierIssueRecord",
    "OutlierIssuesDict",
    "SourceIndexDict",
]


# ---------------------------------------------------------------------------
# TypedDicts for serialized evaluator outputs
# ---------------------------------------------------------------------------


class OutlierIssueRecord(TypedDict):
    """Single outlier issue from DataEval OutliersOutput.

    ``target_index`` is present for target-level outliers (object detection datasets)
    and absent or ``None`` for image-level outliers.
    """

    item_index: int
    metric_name: str
    metric_value: float
    target_index: NotRequired[int | None]


class OutlierIssuesDict(TypedDict):
    """Serialized outlier issues (image or target level)."""

    issues: list[OutlierIssueRecord]
    count: int


class SourceIndexDict(TypedDict):
    """Serialized SourceIndex from DataEval — identifies an item, target, and channel."""

    item: int
    target: int | None
    channel: int | None


# An index value is either a plain ``int`` (image-level) or a
# :class:`SourceIndexDict` (target/channel-level).
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

    # What a row counts.  Deliberately not named "level": DataEval uses that
    # key for metadata levels (unit, instance, track, sequence) and for
    # duplicate levels (item, target), and three meanings under one key is a
    # trap for anyone reading an envelope.
    count_basis: str  # "image" or "annotation"
    rows: list[ClasswiseRowDict]  # one per class + Total row


# ---------------------------------------------------------------------------
# Pydantic output models
# ---------------------------------------------------------------------------


class DataCleaningRawOutput(WorkflowRawOutput):
    """Machine-readable results from data cleaning workflow."""

    duplicates: DuplicatesDict = Field(
        default_factory=lambda: {"items": {}, "targets": {}},
        description=(
            "Duplicate groups from DataEval's `Duplicates`: `exact` and `near` groups of items under `items`, and "
            "of detection targets under `targets`."
        ),
    )
    img_outliers: OutlierIssuesDict = Field(
        default_factory=lambda: {"issues": [], "count": 0},
        description=(
            "Image outliers from DataEval's `Outliers`: one `issues` row per flagged image and metric, naming its "
            "`item_index`, `metric_name` and `metric_value`, and their `count`."
        ),
    )
    label_stats: LabelStatsDict = Field(
        default_factory=dict,  # type: ignore[assignment]  # empty dict valid; all LabelStatsDict keys are optional (total=False)
        description="Label statistics: `item_count`, `class_count`, `index2label` and `label_counts_per_class`.",
    )
    target_outliers: OutlierIssuesDict | None = Field(
        default=None,
        description="Detection-target outliers, as `img_outliers` with a `target_index` per row. None if none.",
    )
    classwise_outliers: ClasswisePivotDict | None = Field(
        default=None,
        description=(
            "Outliers per class, as `rows` of `count` and `pct` flagged: of images for classification, of targets "
            "for detection, as `count_basis` says. None where nothing was flagged."
        ),
    )


class DataCleaningReport(WorkflowReport):
    """Human-readable report for data cleaning workflow."""


class DataCleaningOutput(WorkflowOutput[DataCleaningRawOutput, DataCleaningReport]):
    """Complete data cleaning workflow output."""


class DataCleaningMetadata(ResultMetadata):
    """Metadata for the data-cleaning workflow."""

    mode: Literal["advisory", "preparatory"] = Field(
        default="advisory", description="The `mode` the workflow ran in, as configured: `advisory` or `preparatory`."
    )
    evaluators: list[str] = Field(default_factory=list, description="The DataEval evaluators the run used.")
    flagged_indices: list[int] = Field(
        default_factory=list,
        description=(
            "Images flagged as outliers, or as duplicates after the first of their group, sorted. Empty unless "
            "`mode` is `preparatory`."
        ),
    )
    clean_indices: list[int] = Field(
        default_factory=list,
        description="Every image not in `flagged_indices`, sorted. Empty unless `mode` is `preparatory`.",
    )
    removed_count: int = Field(
        default=0, description="How many images `flagged_indices` holds. Zero unless `mode` is `preparatory`."
    )


class DataCleaningResult(WorkflowResult[DataCleaningMetadata, DataCleaningOutput]):
    """The result of a ``data-cleaning`` run.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. Every workflow result also carries
    ``output.raw.dataset_size`` (:class:`~dataeval_flow.workflows.WorkflowRawOutput`), ``output.report.summary`` and
    ``output.report.findings`` (:class:`~dataeval_flow.workflows.WorkflowReport`), and the envelope fields of
    :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output.raw.duplicates
        Duplicate groups from DataEval's ``Duplicates``: ``exact`` and ``near`` groups of items under ``items``, and of
        detection targets under ``targets``.
    output.raw.img_outliers
        Image outliers from DataEval's ``Outliers``: one ``issues`` row per flagged image and metric, naming its
        ``item_index``, ``metric_name`` and ``metric_value``, and their ``count``.
    output.raw.label_stats
        Label statistics: ``item_count``, ``class_count``, ``index2label`` and ``label_counts_per_class``.
    output.raw.target_outliers
        Detection-target outliers, as ``img_outliers`` with a ``target_index`` per row. None if none.
    output.raw.classwise_outliers
        Outliers per class, as ``rows`` of ``count`` and ``pct`` flagged: of images for classification, of targets for
        detection, as ``count_basis`` says. None where nothing was flagged.
    metadata.mode
        The ``mode`` the workflow ran in, as configured: ``advisory`` or ``preparatory``.
    metadata.evaluators
        The DataEval evaluators the run used.
    metadata.flagged_indices
        Images flagged as outliers, or as duplicates after the first of their group, sorted. Empty unless ``mode`` is
        ``preparatory``.
    metadata.clean_indices
        Every image not in ``flagged_indices``, sorted. Empty unless ``mode`` is ``preparatory``.
    metadata.removed_count
        How many images ``flagged_indices`` holds. Zero unless ``mode`` is ``preparatory``.
    """
