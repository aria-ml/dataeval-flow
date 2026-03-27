"""Data prioritization workflow parameters."""

from collections.abc import Sequence
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dataeval_flow.workflow.base import WorkflowParametersBase

__all__ = [
    "CleaningConfig",
    "DataPrioritizationHealthThresholds",
    "DataPrioritizationParameters",
]

MethodType = Literal["knn", "kmeans_distance", "kmeans_complexity", "hdbscan_distance", "hdbscan_complexity"]
OrderType = Literal["easy_first", "hard_first"]
PolicyType = Literal["difficulty", "stratified", "class_balanced"]


class CleaningConfig(BaseModel):
    """Optional cleaning sub-config for outlier/duplicate removal before prioritization.

    When provided, outlier and duplicate detection runs across all datasets
    before prioritization.  Flagged items are excluded from the ranking.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    outlier_method: Literal["adaptive", "zscore", "modzscore", "iqr"] = Field(
        description="Statistical method for outlier detection.",
    )
    outlier_flags: Sequence[Literal["dimension", "pixel", "visual"]] = Field(
        min_length=1,
        description="Image statistics groups for outlier detection. At least one required.",
    )
    outlier_threshold: float | None = Field(
        default=None,
        ge=0.0,
        description="Custom threshold (None = use DataEval default for chosen method).",
    )
    duplicate_flags: Sequence[Literal["hash_basic", "hash_d4"]] | None = Field(
        default=None,
        description="Hash flag groups for duplicate detection. None = DataEval default (hash_basic).",
    )
    duplicate_merge_near: bool = Field(
        default=True,
        description="Merge overlapping near-duplicate groups from different detection methods.",
    )
    duplicate_exact_only: bool = Field(
        default=False,
        description="When True, only flag exact duplicates — skip near-duplicate detection.",
    )


class DataPrioritizationHealthThresholds(BaseModel):
    """Thresholds that control finding severity for the prioritization report."""

    cleaning_removed_pct_warning: float = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description="Warn if cleaning removes more than this percentage of the combined dataset.",
    )


class DataPrioritizationParameters(WorkflowParametersBase):
    """Parameters for the data-prioritization workflow.

    Requires at least two sources: the first is the reference (labeled) dataset,
    and subsequent sources are unlabeled data pools to prioritize.
    """

    # --- Prioritization method ---
    method: MethodType = Field(
        default="knn",
        description="Ranking method: knn, kmeans_distance, kmeans_complexity, hdbscan_distance, hdbscan_complexity.",
    )
    k: int | None = Field(
        default=None,
        gt=0,
        description="Number of nearest neighbors for knn method. None = sqrt(n_samples).",
    )
    c: int | None = Field(
        default=None,
        gt=0,
        description="Number of clusters for clustering methods. None = sqrt(n_samples).",
    )
    n_init: int | Literal["auto"] = Field(
        default="auto",
        description="Number of K-means initializations (kmeans methods only).",
    )
    max_cluster_size: int | None = Field(
        default=None,
        gt=0,
        description="Maximum cluster size for HDBSCAN methods.",
    )

    # --- Order and policy ---
    order: OrderType = Field(
        default="hard_first",
        description="Sort direction: easy_first (prototypical first) or hard_first (novel/challenging first).",
    )
    policy: PolicyType = Field(
        default="difficulty",
        description="Selection policy: difficulty (direct ordering), stratified (binned), class_balanced.",
    )
    num_bins: int = Field(
        default=50,
        gt=0,
        description="Number of bins for stratified policy.",
    )

    # --- Optional cleaning ---
    cleaning: CleaningConfig | None = Field(
        default=None,
        description="Optional cleaning config. When set, outlier/duplicate detection runs before prioritization.",
    )

    # --- Health thresholds ---
    health_thresholds: DataPrioritizationHealthThresholds = Field(
        default_factory=DataPrioritizationHealthThresholds,
        description="Warning thresholds for the prioritization report.",
    )
