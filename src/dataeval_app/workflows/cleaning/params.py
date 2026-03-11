"""Data cleaning workflow parameters."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from dataeval_app.workflow.base import WorkflowParametersBase

__all__ = ["DataCleaningParameters", "DataCleaningHealthThresholds", "load_params"]

# Duplicated from config.loader to avoid circular import
# (params → config.loader → config.models → params)
DEFAULT_PARAMS_PATH = Path("/data/config/params.yaml")


class DataCleaningHealthThresholds(BaseModel):
    """Configurable warning thresholds for data cleaning health status.

    Each threshold is a percentage (0–100). When the detected rate exceeds the
    threshold the corresponding finding is elevated to ``severity="warning"``;
    otherwise it stays at ``severity="info"``.

    Set a threshold to ``None`` to disable the warning for that metric.
    """

    exact_duplicates: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description=(
            "Max allowable % of images in exact-duplicate groups. "
            "Exact duplicates are byte-identical images that inflate dataset size without "
            "adding information. Default 0% — any exact duplicates trigger a warning. "
            "Raise above 0 only if your pipeline intentionally includes repeated images "
            "(e.g. augmentation-before-split workflows)."
        ),
    )
    near_duplicates: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description=(
            "Max allowable % of images in near-duplicate groups. "
            "Near duplicates are visually similar images (crops, resizes, minor edits) "
            "that can bias model training toward repeated content. Default 5%. "
            "Lower to 1–2% for curated benchmarks; raise to 10–15% for large-scale "
            "web-scraped datasets where some redundancy is expected."
        ),
    )
    image_outliers: float = Field(
        default=3.0,
        ge=0.0,
        le=100.0,
        description=(
            "Max allowable % of images flagged as statistical outliers "
            "(unusual dimensions, brightness, entropy, or visual statistics). Default 3%. "
            "Lower to 1% for safety-critical datasets; raise to 5–10% for diverse "
            "real-world collections where high visual variance is expected."
        ),
    )
    target_outliers: float = Field(
        default=3.0,
        ge=0.0,
        le=100.0,
        description=(
            "Max allowable % of targets (labels/annotations) flagged as outliers "
            "(unusual bounding-box sizes, aspect ratios, or annotation counts). Default 3%. "
            "Lower to 1% for annotation-quality audits; raise to 5–10% for datasets "
            "with naturally high annotation variance (e.g. dense object detection)."
        ),
    )
    classwise_outliers: float = Field(
        default=3.0,
        ge=0.0,
        le=100.0,
        description=(
            "Max allowable % of items flagged as outliers within any single class. Default 3%. "
            "This catches classes where outlier concentration is disproportionately high, "
            "which may indicate labeling errors or class definition issues. "
            "Lower to 1% for label-quality audits; raise to 5–10% for classes with "
            "inherently high visual diversity."
        ),
    )
    class_label_imbalance: float = Field(
        default=5.0,
        ge=1.0,
        description=(
            "Max allowable ratio between the largest and smallest class counts "
            "(max_class / min_class). Default 5:1. "
            "For binary classification, 3:1 is a common threshold for 'imbalanced'. "
            "For large class hierarchies (25+ classes), the long tail naturally "
            "increases this ratio — raise to 10–20:1 to avoid false warnings. "
            "Set to 1.0 to require perfectly balanced classes."
        ),
    )


class DataCleaningParameters(WorkflowParametersBase):
    """Parameters for data cleaning workflow.

    Required parameters must be explicitly set per CR-4.14-G-1
    (avoid application-specific defaults).
    """

    # --- Outlier detection params ---
    outlier_method: Literal["adaptive", "zscore", "modzscore", "iqr"] = Field(
        description="Statistical method for outlier detection",
    )
    outlier_flags: list[Literal["dimension", "pixel", "visual"]] = Field(
        min_length=1,
        description="Image statistics groups for outlier detection. At least one required.",
    )
    outlier_threshold: float | None = Field(
        default=None,
        ge=0.0,
        description="Custom threshold (None = use DataEval default for chosen method)",
    )
    outlier_cluster_threshold: float | None = Field(
        default=None,
        description=(
            "Std devs from cluster center to flag as outlier (requires extractor). None = skip cluster detection."
        ),
    )
    outlier_cluster_algorithm: Literal["kmeans", "hdbscan"] | None = Field(
        default=None,
        description="Clustering algorithm for cluster-based outlier detection.",
    )
    outlier_n_clusters: int | None = Field(
        default=None,
        description="Expected number of clusters. None = auto-detect.",
    )

    # --- Duplicate detection params ---
    duplicate_flags: list[Literal["hash_basic", "hash_d4"]] | None = Field(
        default=None,
        description=(
            "Hash flag groups for duplicate detection. None = DataEval default (hash_basic: xxhash + phash + dhash)."
        ),
    )
    duplicate_merge_near: bool = Field(
        default=True,
        description="Merge overlapping near-duplicate groups from different detection methods.",
    )
    duplicate_cluster_sensitivity: float | None = Field(
        default=None,
        description=(
            "Threshold for cluster-based near duplicate detection (requires extractor). None = skip cluster detection."
        ),
    )
    duplicate_cluster_algorithm: Literal["kmeans", "hdbscan"] | None = Field(
        default=None,
        description="Clustering algorithm for cluster-based duplicate detection.",
    )
    duplicate_n_clusters: int | None = Field(
        default=None,
        description="Expected number of clusters for duplicate detection. None = auto-detect.",
    )

    # --- Health thresholds ---
    health_thresholds: DataCleaningHealthThresholds = Field(
        default_factory=DataCleaningHealthThresholds,
        description="Warning thresholds for dataset health status. Findings are flagged as warnings.",
    )


def load_params(params_path: Path | None = None) -> DataCleaningParameters:
    """Load data cleaning parameters from 'data_cleaning' section."""
    path = params_path or DEFAULT_PARAMS_PATH

    if not path.exists():
        msg = f"Parameters file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "data_cleaning" not in data:
        msg = (
            f"'data_cleaning' section not found in {path}. "
            f"Config must have 'data_cleaning:' section with required fields."
        )
        raise ValueError(msg)

    return DataCleaningParameters.model_validate(data["data_cleaning"])
