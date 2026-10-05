"""The ``data-cleaning`` workflow's config and check settings."""

from collections.abc import Mapping, Sequence
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin, StatsConfigMixin
from dataeval_flow.evaluators._threshold import ThresholdSpec
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.workflows._base import WorkflowConfig

__all__ = [
    "ClasswiseOutliersSettings",
    "DataCleaningChecks",
    "DataCleaningClassImbalanceSettings",
    "DataCleaningConfig",
    "DuplicatesSettings",
    "ImageDuplicatesSettings",
    "ImageOutliersSettings",
    "OutliersSettings",
    "TargetOutliersSettings",
]


class ImageOutliersSettings(BaseModel):
    """The `image-outliers` check's settings in data-cleaning."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: float | None = Field(
        default=3.0,
        ge=0.0,
        le=100.0,
        description=(
            "Most images, as a percentage of the Dataset, that may be flagged as statistical outliers (unusual "
            "dimensions, brightness, entropy, or visual statistics) before the finding warns; `null` judges nothing. "
            "Lower to 1% for safety-critical datasets; raise to 5-10% for diverse real-world collections."
        ),
    )


class TargetOutliersSettings(BaseModel):
    """The `target-outliers` check's settings in data-cleaning."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: float | None = Field(
        default=3.0,
        ge=0.0,
        le=100.0,
        description=(
            "Most targets (boxes), as a percentage of all, that may be flagged as outliers (unusual box sizes, aspect "
            "ratios, or annotation counts) before the finding warns; `null` judges nothing. Lower to 1% for "
            "annotation-quality audits; raise to 5-10% for dense object detection."
        ),
    )


class ClasswiseOutliersSettings(BaseModel):
    """The `classwise-outliers` check's settings in data-cleaning."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: float | None = Field(
        default=3.0,
        ge=0.0,
        le=100.0,
        description=(
            "Most items, as a percentage of those in any one class, that may be outliers before the finding warns; "
            "`null` judges nothing. A class over it may point to labelling errors or a loose class definition."
        ),
    )


class ImageDuplicatesSettings(BaseModel):
    """The `image-duplicates` check's settings in data-cleaning."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    exact: float | None = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description=(
            "Most images, as a percentage of the Dataset, that may sit in exact-duplicate groups before the finding "
            "warns; `null` judges nothing. 0 warns on any. Raise it only where repeated images are intended."
        ),
    )

    near: float | None = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description=(
            "Most images, as a percentage of the Dataset, that may sit in near-duplicate groups before the finding "
            "warns; `null` judges nothing. Lower to 1-2% for curated benchmarks; raise to 10-15% for web-scraped data."
        ),
    )


class DataCleaningClassImbalanceSettings(BaseModel):
    """The `class-imbalance` check's settings in data-cleaning."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: float | None = Field(
        default=5.0,
        ge=1.0,
        description=(
            "Largest class count over smallest past which the finding warns; `null` judges nothing but an empty "
            "class. 3:1 is a common bar for two classes; raise to 10-20:1 for 25 or more classes, whose long tail "
            "raises it naturally."
        ),
    )


class DataCleaningChecks(BaseModel):
    """When data-cleaning's findings warn: each check's settings, keyed by check type."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    image_outliers: ImageOutliersSettings = Field(
        default_factory=ImageOutliersSettings,
        alias="image-outliers",
        description="The `image-outliers` check's settings.",
    )
    target_outliers: TargetOutliersSettings = Field(
        default_factory=TargetOutliersSettings,
        alias="target-outliers",
        description="The `target-outliers` check's settings.",
    )
    classwise_outliers: ClasswiseOutliersSettings = Field(
        default_factory=ClasswiseOutliersSettings,
        alias="classwise-outliers",
        description="The `classwise-outliers` check's settings.",
    )
    image_duplicates: ImageDuplicatesSettings = Field(
        default_factory=ImageDuplicatesSettings,
        alias="image-duplicates",
        description="The `image-duplicates` check's settings.",
    )
    class_imbalance: DataCleaningClassImbalanceSettings = Field(
        default_factory=DataCleaningClassImbalanceSettings,
        alias="class-imbalance",
        description="The `class-imbalance` check's settings.",
    )


class OutliersSettings(BaseModel):
    """The `outliers` step's settings: which statistics, and how far out an outlier sits. data-prioritization's
    `cleaning:` takes the same block."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    flags: Sequence[Literal["dimension", "pixel", "visual"]] = Field(
        min_length=1, description="Image statistics groups to judge. At least one."
    )
    outlier_threshold: ThresholdSpec | Mapping[str, ThresholdSpec] = Field(
        description=(
            "The method, such as `zscore`, `modzscore`, `iqr` or `adaptive`, alone for its default bound or as "
            "`[method, bound]`; or a mapping from flag to either."
        ),
    )
    cluster_threshold: float | None = Field(
        default=None,
        description="Standard deviations from a cluster's center past which an item is an outlier; needs an extractor. "
        "Unset skips cluster detection.",
    )
    cluster_algorithm: Literal["kmeans", "hdbscan"] | None = Field(
        default=None, description="The clustering algorithm cluster detection uses."
    )
    n_clusters: int | None = Field(default=None, description="Expected number of clusters; unset detects it.")


class DuplicatesSettings(BaseModel):
    """The `duplicates` step's settings. data-prioritization's `cleaning:` takes the same block."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    flags: Sequence[Literal["hash_basic", "hash_d4"]] | None = Field(
        default=None, description="Hash groups to compare; unset is DataEval's default, `hash_basic`."
    )
    merge_near_duplicates: bool = Field(
        default=True, description="Merge overlapping near-duplicate groups found by different methods."
    )
    cluster_sensitivity: float | None = Field(
        default=None, description="Cluster-based near-duplicate threshold; needs an extractor. Unset skips it."
    )
    cluster_algorithm: Literal["kmeans", "hdbscan"] | None = Field(
        default=None, description="The clustering algorithm cluster detection uses."
    )
    n_clusters: int | None = Field(default=None, description="Expected number of clusters; unset detects it.")


class DataCleaningConfig(WorkflowConfig[ChainResult], MetadataConfigMixin, StatsConfigMixin):
    """The settings of one ``data-cleaning`` entry: how outliers and duplicates are detected, and when a finding warns.

    The type is a preset: these settings expand to a chain of steps, whose checks make the findings and whose
    ``clean`` step removes each flagged item and each duplicate but the first (see ``DataCleaningWorkflow``).

    Required fields have no default and must be set, per CR-4.14-G-1
    (avoid application-specific defaults).

    Example YAML::

        workflows:
          - name: clean_zscore_stats
            type: data-cleaning
            outliers:
              flags: [pixel, visual]
              outlier_threshold: zscore
    """

    type: str = Field(default="data-cleaning", description="The workflow type this entry configures: `data-cleaning`.")

    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.STATS, InputKind.METADATA}),
        optional=frozenset({InputKind.CLUSTERS}),
        sources=SourceCount.ONE,
    )

    outliers: OutliersSettings = Field(description="The `outliers` step's settings.")
    duplicates: DuplicatesSettings = Field(
        default_factory=DuplicatesSettings, description="The `duplicates` step's settings."
    )

    # --- Checks ---
    checks: DataCleaningChecks = Field(
        default_factory=DataCleaningChecks, description="When findings warn, keyed by check type."
    )

    def wanted_kinds(self) -> frozenset[InputKind]:
        """Stats and metadata always, and clusters too when a cluster parameter is set."""
        cluster_fields = (
            self.outliers.cluster_threshold,
            self.outliers.cluster_algorithm,
            self.outliers.n_clusters,
            self.duplicates.cluster_sensitivity,
            self.duplicates.cluster_algorithm,
            self.duplicates.n_clusters,
        )
        clusters = frozenset({InputKind.CLUSTERS}) if any(f is not None for f in cluster_fields) else frozenset()
        return self.inputs.required | clusters
