"""Drift monitoring workflow parameters."""

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataeval_app.workflow.base import WorkflowParametersBase

__all__ = [
    "ChunkingConfig",
    "DriftDetectorConfig",
    "DriftDetectorDomainClassifier",
    "DriftDetectorKNeighbors",
    "DriftDetectorMMD",
    "DriftDetectorUnivariate",
    "DriftHealthThresholds",
    "DriftMonitoringParameters",
    "UpdateStrategyConfig",
]


# ---------------------------------------------------------------------------
# Chunking configuration
# ---------------------------------------------------------------------------


class ChunkingConfig(BaseModel):
    """Configuration for chunked (temporal) drift analysis.

    When present, the test data is split into sequential chunks and each
    chunk is tested for drift independently.  This answers "when did drift
    start?" and "did it recover?".
    """

    chunk_size: int | None = Field(
        default=None,
        gt=0,
        description="Fixed number of samples per chunk. Mutually exclusive with chunk_count.",
    )
    chunk_count: int | None = Field(
        default=None,
        gt=0,
        description="Split test data into this many equal chunks. Mutually exclusive with chunk_size.",
    )
    incomplete: Literal["keep", "drop", "append"] = Field(
        default="keep",
        description=(
            "How to handle a final chunk smaller than chunk_size. "
            "'keep': retain as-is, 'drop': discard, 'append': merge into last full chunk."
        ),
    )
    threshold_multiplier: float = Field(
        default=3.0,
        gt=0.0,
        description=(
            "Z-score multiplier for the chunk-level drift threshold. "
            "Lower values are more sensitive (e.g. 2.0), higher values more conservative."
        ),
    )

    @model_validator(mode="after")
    def _validate_chunking(self) -> "ChunkingConfig":
        if self.chunk_size is None and self.chunk_count is None:
            raise ValueError("Either chunk_size or chunk_count must be set.")
        if self.chunk_size is not None and self.chunk_count is not None:
            raise ValueError("chunk_size and chunk_count are mutually exclusive.")
        return self


# ---------------------------------------------------------------------------
# Drift detector configs — discriminated union on ``method``
# ---------------------------------------------------------------------------


class DriftDetectorUnivariate(BaseModel):
    """Univariate statistical test per feature.

    Applies a chosen statistical test independently to each feature dimension
    and flags drift when enough features reject the null hypothesis (after
    multiple-testing correction).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    method: Literal["univariate"] = "univariate"
    test: Literal["ks", "cvm", "mwu", "anderson", "bws"] = Field(
        default="ks",
        description=(
            "Statistical test to apply per feature. "
            "'ks' (Kolmogorov-Smirnov) is a good general-purpose default. "
            "'cvm' (Cramér-von Mises) integrates squared CDF distance. "
            "'mwu' (Mann-Whitney U) is rank-based and outlier-robust. "
            "'anderson' (Anderson-Darling) is tail-sensitive. "
            "'bws' (Baumgartner-Weiss-Schindler) has high power with tail sensitivity."
        ),
    )
    p_val: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Significance level for the test.",
    )
    correction: Literal["bonferroni", "fdr"] = Field(
        default="bonferroni",
        description="Multiple-testing correction method across features.",
    )
    alternative: Literal["two-sided", "less", "greater"] = Field(
        default="two-sided",
        description="Alternative hypothesis direction.",
    )
    n_features: int | None = Field(
        default=None,
        gt=0,
        description="Expected number of features. None = infer from reference data.",
    )
    classwise: bool = Field(
        default=False,
        description=(
            "Run drift detection per class for this detector. Requires labelled datasets. Runs non-chunked only."
        ),
    )
    chunking: ChunkingConfig | None = Field(
        default=None,
        description="Chunked (temporal) drift analysis for this detector. None disables chunking.",
    )


class DriftDetectorMMD(BaseModel):
    """Maximum Mean Discrepancy with permutation test.

    Computes the MMD² statistic between reference and test distributions
    using an RBF kernel, then estimates a p-value via permutation testing.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    method: Literal["mmd"] = "mmd"
    p_val: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Significance level for the permutation test.",
    )
    n_permutations: int = Field(
        default=100,
        gt=0,
        description="Number of permutations for the test.",
    )
    device: str | None = Field(
        default=None,
        description="PyTorch device for kernel computation (e.g. 'cpu', 'cuda:0'). None = auto.",
    )
    classwise: bool = Field(
        default=False,
        description=(
            "Run drift detection per class for this detector. Requires labelled datasets. Runs non-chunked only."
        ),
    )
    chunking: ChunkingConfig | None = Field(
        default=None,
        description="Chunked (temporal) drift analysis for this detector. None disables chunking.",
    )


class DriftDetectorDomainClassifier(BaseModel):
    """Binary domain classifier (LightGBM) approach.

    Trains a binary classifier to distinguish reference from test data.
    An AUROC significantly above 0.5 indicates the distributions are
    distinguishable — i.e. drift has occurred.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    method: Literal["domain_classifier"] = "domain_classifier"
    n_folds: int = Field(
        default=5,
        ge=2,
        description="Number of cross-validation folds.",
    )
    threshold: float = Field(
        default=0.55,
        gt=0.5,
        le=1.0,
        description="AUROC threshold above which drift is declared.",
    )
    classwise: bool = Field(
        default=False,
        description=(
            "Run drift detection per class for this detector. Requires labelled datasets. Runs non-chunked only."
        ),
    )
    chunking: ChunkingConfig | None = Field(
        default=None,
        description="Chunked (temporal) drift analysis for this detector. None disables chunking.",
    )


class DriftDetectorKNeighbors(BaseModel):
    """K-nearest neighbors distance-based drift detection.

    Computes per-sample k-NN distances for test data against the reference
    set. A Mann-Whitney U test determines whether test distances are
    stochastically larger than reference self-distances.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    method: Literal["kneighbors"] = "kneighbors"
    k: int = Field(
        default=10,
        gt=0,
        description="Number of nearest neighbors.",
    )
    distance_metric: Literal["cosine", "euclidean"] = Field(
        default="euclidean",
        description="Distance metric for k-NN computation.",
    )
    p_val: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Significance level for the Mann-Whitney U test.",
    )
    classwise: bool = Field(
        default=False,
        description=(
            "Run drift detection per class for this detector. Requires labelled datasets. Runs non-chunked only."
        ),
    )
    chunking: ChunkingConfig | None = Field(
        default=None,
        description="Chunked (temporal) drift analysis for this detector. None disables chunking.",
    )


# Discriminated union — Pydantic selects the right model based on ``method``.
DriftDetectorConfig = Annotated[
    DriftDetectorUnivariate | DriftDetectorMMD | DriftDetectorDomainClassifier | DriftDetectorKNeighbors,
    Field(discriminator="method"),
]


# ---------------------------------------------------------------------------
# Update strategy (stubbed for future online detection)
# ---------------------------------------------------------------------------


class UpdateStrategyConfig(BaseModel):
    """Reference-set update strategy — **stubbed for future use**.

    When online drift detection is implemented, this will control how the
    reference distribution evolves after each prediction step.  Currently
    accepted in configuration but not applied at runtime.
    """

    type: Literal["last_seen", "reservoir_sampling"] = Field(
        description="Update algorithm. 'last_seen': sliding window. 'reservoir_sampling': uniform random sample.",
    )
    n: int = Field(
        gt=0,
        description="Number of samples to retain in the updated reference set.",
    )


# ---------------------------------------------------------------------------
# Health thresholds
# ---------------------------------------------------------------------------


class DriftHealthThresholds(BaseModel):
    """Configurable thresholds that control finding severity.

    Findings that exceed a threshold are elevated to ``severity="warning"``;
    otherwise they stay at ``severity="info"``.
    """

    any_drift_is_warning: bool = Field(
        default=True,
        description=(
            "Non-chunked mode: if any detector flags drift, the finding is a warning. "
            "Set to False to treat all non-chunked results as informational."
        ),
    )
    chunk_drift_pct_warning: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Chunked mode: percentage of chunks that must drift to trigger a warning.",
    )
    consecutive_chunks_warning: int = Field(
        default=3,
        ge=1,
        description=(
            "Chunked mode: number of consecutive drifted chunks that triggers a warning. "
            "Sustained drift is typically more concerning than sporadic drift."
        ),
    )
    classwise_any_drift_is_warning: bool = Field(
        default=True,
        description="Classwise mode: if any class drifts for any detector, the finding is a warning.",
    )


# ---------------------------------------------------------------------------
# Top-level parameters
# ---------------------------------------------------------------------------


class DriftMonitoringParameters(WorkflowParametersBase):
    """Parameters for the drift-monitoring workflow.

    At least one detector must be configured.  Chunking and classwise
    analysis are optional extensions.
    """

    detectors: list[DriftDetectorConfig] = Field(
        min_length=1,
        description="List of drift detectors to run. At least one required.",
    )
    update_strategy: UpdateStrategyConfig | None = Field(
        default=None,
        description=(
            "Reference-set update strategy (stubbed — not yet applied at runtime). "
            "Accepted in config to preserve forward compatibility."
        ),
    )
    health_thresholds: DriftHealthThresholds = Field(
        default_factory=DriftHealthThresholds,
        description="Warning thresholds for drift severity classification.",
    )
