"""OOD detection workflow parameters."""

from collections.abc import Sequence
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dataeval_flow.workflow.base import WorkflowParametersBase

__all__ = [
    "OODDetectionParameters",
    "OODDetectorConfig",
    "OODDetectorDomainClassifier",
    "OODDetectorKNeighbors",
    "OODHealthThresholds",
]


# ---------------------------------------------------------------------------
# OOD detector configs — discriminated union on ``method``
# ---------------------------------------------------------------------------


class OODDetectorKNeighbors(BaseModel):
    """K-nearest neighbors OOD detector.

    Uses average distance to k nearest neighbors in embedding space to
    detect OOD samples.  Samples with larger average distances to their
    k nearest neighbors in the reference set are considered more likely OOD.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    method: Literal["kneighbors"] = "kneighbors"
    k: int = Field(
        default=10,
        gt=0,
        description="Number of nearest neighbors to consider.",
    )
    distance_metric: Literal["cosine", "euclidean"] = Field(
        default="cosine",
        description="Distance metric for k-NN computation.",
    )
    threshold_perc: float = Field(
        default=95.0,
        gt=0.0,
        le=100.0,
        description=(
            "Percentage of reference data considered normal (0-100). "
            "Higher values result in more permissive thresholds."
        ),
    )


class OODDetectorDomainClassifier(BaseModel):
    """Domain classifier OOD detector.

    Uses a LightGBM classifier's ability to distinguish test samples from
    reference samples as an OOD signal.  Samples that a classifier can easily
    identify as 'not reference' are likely OOD.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    method: Literal["domain_classifier"] = "domain_classifier"
    n_folds: int = Field(
        default=5,
        ge=2,
        description="Number of cross-validation folds per repeat.",
    )
    n_repeats: int = Field(
        default=5,
        ge=1,
        description="Number of times to repeat the k-fold split.",
    )
    n_std: float = Field(
        default=2.0,
        gt=0.0,
        description="Number of standard deviations above the null mean for threshold.",
    )
    threshold_perc: float = Field(
        default=95.0,
        gt=0.0,
        le=100.0,
        description=(
            "Percentage of reference data considered normal (0-100). "
            "Higher values result in more permissive thresholds."
        ),
    )


# Discriminated union — Pydantic selects the right model based on ``method``.
OODDetectorConfig = Annotated[
    OODDetectorKNeighbors | OODDetectorDomainClassifier,
    Field(discriminator="method"),
]


# ---------------------------------------------------------------------------
# Health thresholds
# ---------------------------------------------------------------------------


class OODHealthThresholds(BaseModel):
    """Configurable thresholds that control finding severity.

    Findings that exceed a threshold are elevated to ``severity="warning"``;
    otherwise they stay at ``severity="info"`` or ``severity="ok"``.
    """

    ood_pct_warning: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Percentage of test samples flagged OOD that triggers a warning.",
    )
    ood_pct_info: float = Field(
        default=1.0,
        ge=0.0,
        le=100.0,
        description=(
            "Percentage of test samples flagged OOD that triggers an info finding. "
            "Below this percentage, severity is 'ok'."
        ),
    )


# ---------------------------------------------------------------------------
# Top-level parameters
# ---------------------------------------------------------------------------


class OODDetectionParameters(WorkflowParametersBase):
    """Parameters for the ood-detection workflow.

    At least one detector must be configured.  Metadata insights are
    enabled by default to explain why samples are flagged OOD.
    """

    detectors: Sequence[OODDetectorConfig] = Field(
        min_length=1,
        description="List of OOD detectors to run. At least one required.",
    )
    health_thresholds: OODHealthThresholds = Field(
        default_factory=OODHealthThresholds,
        description="Warning thresholds for OOD severity classification.",
    )
    metadata_insights: bool = Field(
        default=True,
        description=(
            "Whether to compute factor_deviation and factor_predictors for OOD samples to explain why they are flagged."
        ),
    )
    max_ood_insights: int = Field(
        default=50,
        gt=0,
        description=(
            "Maximum number of OOD samples to compute detailed metadata "
            "deviation for. Caps compute cost on large datasets."
        ),
    )
