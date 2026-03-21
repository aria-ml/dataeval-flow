"""Data analysis workflow parameters."""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from dataeval_flow.workflow.base import WorkflowParametersBase

__all__ = ["DataAnalysisHealthThresholds", "DataAnalysisParameters"]


class DataAnalysisHealthThresholds(BaseModel):
    """Configurable warning thresholds for data analysis health status.

    Each threshold is a percentage (0–100) unless otherwise noted.
    When the detected rate exceeds the threshold the corresponding
    finding is elevated to ``severity="warning"``; otherwise it stays
    at ``severity="info"``.

    Set a threshold to ``None`` to disable the warning for that metric.
    """

    image_outliers: float = Field(
        default=3.0,
        ge=0.0,
        le=100.0,
        description=(
            "Max allowable % of images flagged as statistical outliers. Default 3%. "
            "Lower to 1% for safety-critical datasets; raise to 5–10% for diverse "
            "real-world collections where high visual variance is expected."
        ),
    )
    exact_duplicates: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description=(
            "Max allowable % of images in exact-duplicate groups. Default 0% — any exact duplicates trigger a warning."
        ),
    )
    near_duplicates: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description=(
            "Max allowable % of images in near-duplicate groups. Default 5%. "
            "Lower to 1–2% for curated benchmarks; raise to 10–15% for large-scale "
            "web-scraped datasets where some redundancy is expected."
        ),
    )
    class_label_imbalance: float = Field(
        default=5.0,
        ge=1.0,
        description=(
            "Max allowable ratio between the largest and smallest class counts "
            "(max_class / min_class). Default 5:1. "
            "Set to 1.0 to require perfectly balanced classes."
        ),
    )
    distribution_shift: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "Embedding divergence above which cross-split shift is flagged as warning. "
            "Default 0.5. Values above 0.2 are reported as 'moderate'."
        ),
    )


class DataAnalysisParameters(WorkflowParametersBase):
    """Parameters for data analysis workflow.

    Required parameters must be explicitly set per CR-4.14-G-1
    (avoid application-specific defaults).
    """

    # --- Outlier detection ---
    outlier_method: Literal["adaptive", "zscore", "modzscore", "iqr"] = Field(
        description="Statistical method for outlier detection",
    )
    outlier_flags: Sequence[Literal["dimension", "pixel", "visual"]] = Field(
        min_length=1,
        description="Image statistics groups for outlier detection. At least one required.",
    )
    outlier_threshold: float | None = Field(
        default=None,
        ge=0.0,
        description="Custom threshold (None = use DataEval default for chosen method)",
    )

    # --- Per-split bias analysis ---
    balance: bool = Field(default=False, description="Run Balance (MI) analysis per split and combined")
    diversity_method: Literal["simpson", "shannon"] | None = Field(
        default=None, description="Diversity method (None = skip)"
    )
    include_image_stats: bool = Field(
        default=False,
        description="Compute ImageStats and inject as metadata factors for bias analysis",
    )

    # --- Cross-split divergence ---
    divergence_method: Literal["mst", "fnn"] | None = Field(
        default=None,
        description=(
            "Method for computing cross-split embedding divergence. Only used when an extractor is configured."
        ),
    )

    # --- Health thresholds ---
    health_thresholds: DataAnalysisHealthThresholds = Field(
        default_factory=DataAnalysisHealthThresholds,
        description="Warning thresholds for dataset health status. Findings are flagged as warnings.",
    )
