"""Data cleaning workflow parameters."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from dataeval_app.workflow.base import WorkflowParametersBase

__all__ = ["DataCleaningParameters", "load_params"]

# Duplicated from config.loader to avoid circular import
# (params → config.loader → config.models → params)
DEFAULT_PARAMS_PATH = Path("/data/config/params.yaml")


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
