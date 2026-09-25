"""The ``ood-detection`` workflow's config: its detectors and health thresholds."""

from collections.abc import Sequence
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin, StatsConfigMixin
from dataeval_flow.workflows._base import WorkflowConfig, _LegacyValueRangeMixin
from dataeval_flow.workflows.ood_detection._outputs import OODDetectionResult

__all__ = [
    "OODDetectionConfig",
    "OODDetectorConfig",
    "OODDetectorDomainClassifier",
    "OODDetectorKNeighbors",
    "OODDetectionHealthThresholds",
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

    method: Literal["kneighbors"] = Field(default="kneighbors", description="Selects this OOD detector: `kneighbors`.")
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
    reference samples as an OOD signal.  Samples that a classifier can identify as 'not reference' are likely OOD.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    method: Literal["domain_classifier"] = Field(
        default="domain_classifier", description="Selects this OOD detector: `domain_classifier`."
    )
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


class OODDetectionHealthThresholds(BaseModel):
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


class OODDetectionConfig(
    WorkflowConfig[OODDetectionResult], MetadataConfigMixin, _LegacyValueRangeMixin, StatsConfigMixin
):
    """The settings of one ``ood-detection`` entry: the OOD detectors scored against the reference, and when they warn.

    At least one detector must be configured.  Metadata insights are
    enabled by default to explain why samples are flagged OOD.

    Example YAML::

        workflows:
          - name: ood_knn
            type: ood-detection
            detectors:
              - method: kneighbors
                k: 10
    """

    type: str = Field(default="ood-detection", description="The workflow type this entry configures: `ood-detection`.")

    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.EMBEDDINGS, InputKind.STATS, InputKind.METADATA}),
        sources=SourceCount.TWO_OR_MORE,
    )

    detectors: Sequence[OODDetectorConfig] = Field(
        min_length=1,
        description="List of OOD detectors to run. At least one required.",
    )
    health_thresholds: OODDetectionHealthThresholds = Field(
        default_factory=OODDetectionHealthThresholds,
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
