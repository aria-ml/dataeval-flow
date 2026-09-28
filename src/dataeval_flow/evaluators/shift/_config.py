"""Configs for the ``shift`` evaluators: DataEval's drift and out-of-distribution detectors.

Field names are DataEval's argument names. An unset field is not passed, so DataEval's own default applies. Each
reads the embeddings of the task's sources, the reference first.
"""

__all__ = [
    "ChunkedDriftConfig",
    "DriftDomainClassifierConfig",
    "DriftKNeighborsConfig",
    "DriftMMDConfig",
    "DriftUnivariateConfig",
    "DriftWassersteinConfig",
]

from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.evaluators._base import EvaluatorConfig
from dataeval_flow.evaluators._threshold import ThresholdSpec
from dataeval_flow.evaluators.shift._result import (
    DriftDomainClassifierResult,
    DriftKNeighborsResult,
    DriftMMDResult,
    DriftUnivariateResult,
    DriftWassersteinResult,
)

_CHUNKING_DESCRIPTION = (
    "Split the data into chunks, and test each against the spread of the reference's chunks. Unset tests the data "
    "whole."
)


class ChunkedDriftConfig(BaseModel):
    """How a drift detector splits its data into chunks: the arguments of DataEval's ``chunked()``.

    Named for DataEval's ``ChunkedDrift``, which ``chunked()`` returns. Set it as a drift entry's ``chunking:`` to
    test each chunk of the data against the spread of the reference's chunks, rather than the data as a whole. Give
    ``chunk_size`` or ``chunk_count``.

    Example YAML::

        evaluators:
          - name: mmd_chunked
            type: shift.drift-mmd
            chunking:
              chunk_count: 10
              threshold: [zscore, 2.5]
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    chunk_size: int | None = Field(
        default=None, gt=0, description="Items per chunk. Set this or `chunk_count`; given both, DataEval uses this."
    )
    chunk_count: int | None = Field(default=None, gt=0, description="Number of equal chunks. Set this or `chunk_size`.")
    threshold: ThresholdSpec | None = Field(
        default=None,
        description=(
            "How far a chunk's drift metric must sit from the reference chunks' to count as drift: a method "
            "(`zscore`, `modzscore`, `iqr`, `adaptive`, `constant`), bounds, or `[method, bounds]`. Unset uses the "
            "detector's default."
        ),
    )
    incomplete: Literal["keep", "drop", "append"] | None = Field(
        default=None,
        description=(
            "What `chunk_size` does with the reference's final chunk when it falls short: `keep` scores it as a chunk "
            "of its own, `drop` leaves it out of the baseline, and `append` merges it into the chunk before. The data "
            "to test always merges its remainder into its last chunk. Needs `chunk_size`. Unset uses DataEval's "
            "default (`keep`)."
        ),
    )

    @model_validator(mode="after")
    def _sized(self) -> Self:
        if self.chunk_size is None and self.chunk_count is None:
            raise ValueError("Set `chunk_size` or `chunk_count`: DataEval needs one to cut the data into chunks.")
        if self.incomplete is not None and self.chunk_size is None:
            raise ValueError("`incomplete` applies only to `chunk_size`: set `chunk_size`, or remove `incomplete`.")
        return self


class DriftUnivariateConfig(EvaluatorConfig[DriftUnivariateResult]):
    """Config for ``shift.drift-univariate``, DataEval's DriftUnivariate.

    Tests each embedding dimension of the second source against the first's with a univariate statistical test, and
    declares drift when any dimension drifts after a multiple-testing correction. Needs an extractor on the task, and
    two sources: the reference, then the data to test.

    Every parameter, its DataEval argument and its unset behaviour is listed in the Evaluator Catalog
    (``reference/evaluators``), and ``dataeval-flow evaluators shift.drift-univariate`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: ks
            type: shift.drift-univariate
            p_val: 0.01

        tasks:
          - name: drift_check
            evaluator: ks
            sources: [train, operational]
            extractor: bovw_ext
    """

    type: str = Field(
        default="shift.drift-univariate",
        description="The evaluator type this entry configures: `shift.drift-univariate`.",
    )
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.EMBEDDINGS}), sources=SourceCount.TWO)

    method: Literal["ks", "cvm", "mwu", "anderson", "bws"] | None = Field(
        default=None,
        description=(
            "The statistical test: `ks` (Kolmogorov-Smirnov), `cvm` (Cramér-von Mises), `mwu` (Mann-Whitney U), "
            "`anderson` (Anderson-Darling) or `bws` (Baumgartner-Weiss-Schindler). Unset uses DataEval's default "
            "(`ks`)."
        ),
    )
    p_val: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Significance threshold for drift. Unset uses DataEval's default (0.05).",
    )
    correction: Literal["bonferroni", "fdr"] | None = Field(
        default=None,
        description=(
            "Multiple-testing correction across dimensions: `bonferroni`, conservative, or `fdr` (Benjamini-Hochberg). "
            "Unset uses DataEval's default (`bonferroni`)."
        ),
    )
    alternative: Literal["two-sided", "less", "greater"] | None = Field(
        default=None,
        description=(
            "Alternative hypothesis, for the `ks`, `mwu` and `bws` tests; `cvm` and `anderson` are always two-sided. "
            "Unset uses DataEval's default (`two-sided`)."
        ),
    )
    n_features: int | None = Field(
        default=None, gt=0, description="Embedding dimensions to test. Unset infers them from the embeddings."
    )
    chunking: ChunkedDriftConfig | None = Field(default=None, description=_CHUNKING_DESCRIPTION)


class DriftMMDConfig(EvaluatorConfig[DriftMMDResult]):
    """Config for ``shift.drift-mmd``, DataEval's DriftMMD.

    Measures the maximum mean discrepancy between the two sources' embeddings, and tests it against a permutation
    estimate of its no-drift distribution. Needs an extractor on the task, and two sources: the reference, then the
    data to test.

    Every parameter, its DataEval argument and its unset behaviour is listed in the Evaluator Catalog
    (``reference/evaluators``), and ``dataeval-flow evaluators shift.drift-mmd`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: mmd
            type: shift.drift-mmd
            n_permutations: 500
    """

    type: str = Field(
        default="shift.drift-mmd", description="The evaluator type this entry configures: `shift.drift-mmd`."
    )
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.EMBEDDINGS}), sources=SourceCount.TWO)

    p_val: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Significance threshold for drift. Unset uses DataEval's default (0.05).",
    )
    n_permutations: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Permutations in the test that estimates the no-drift distribution; more is more accurate and slower. "
            "Unset uses DataEval's default (100)."
        ),
    )
    permutation_batch_size: int | Literal["auto"] | None = Field(
        default=None,
        description=(
            "Permutations computed per batch, to bound memory: a count, or `auto`, which sizes it from free GPU memory "
            "on CUDA and computes all at once on CPU. Unset uses DataEval's default (`auto`)."
        ),
    )
    chunking: ChunkedDriftConfig | None = Field(default=None, description=_CHUNKING_DESCRIPTION)


class DriftKNeighborsConfig(EvaluatorConfig[DriftKNeighborsResult]):
    """Config for ``shift.drift-kneighbors``, DataEval's DriftKNeighbors.

    Compares the test data's distances to their nearest reference neighbors with the reference's own. Needs an
    extractor on the task, and two sources: the reference, then the data to test.

    Every parameter, its DataEval argument and its unset behaviour is listed in the Evaluator Catalog
    (``reference/evaluators``), and ``dataeval-flow evaluators shift.drift-kneighbors`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: knn_drift
            type: shift.drift-kneighbors
            k: 5
    """

    type: str = Field(
        default="shift.drift-kneighbors",
        description="The evaluator type this entry configures: `shift.drift-kneighbors`.",
    )
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.EMBEDDINGS}), sources=SourceCount.TWO)

    k: int | None = Field(default=None, gt=0, description="Nearest neighbors. Unset uses DataEval's default (10).")
    distance_metric: Literal["cosine", "euclidean"] | None = Field(
        default=None, description="Distance between embeddings. Unset uses DataEval's default (`euclidean`)."
    )
    p_val: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Significance threshold for drift, without chunking. Unset uses DataEval's default (0.05).",
    )
    chunking: ChunkedDriftConfig | None = Field(default=None, description=_CHUNKING_DESCRIPTION)


class DriftWassersteinConfig(EvaluatorConfig[DriftWassersteinResult]):
    """Config for ``shift.drift-wasserstein``, DataEval's DriftWasserstein.

    Compares each embedding dimension's Wasserstein distance from the reference to the test data with its distance to
    an in-distribution validation set, and declares drift where the ratio passes ``ratio_threshold``. Needs an
    extractor on the task, and three sources: the reference, the validation set, then the data to test.

    Every parameter, its DataEval argument and its unset behaviour is listed in the Evaluator Catalog
    (``reference/evaluators``), and ``dataeval-flow evaluators shift.drift-wasserstein`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: wasserstein
            type: shift.drift-wasserstein

        tasks:
          - name: drift_check
            evaluator: wasserstein
            sources: [train, validation, operational]
            extractor: bovw_ext
    """

    type: str = Field(
        default="shift.drift-wasserstein",
        description="The evaluator type this entry configures: `shift.drift-wasserstein`.",
    )
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.EMBEDDINGS}), sources=SourceCount.THREE)

    ratio_threshold: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Ratio of a dimension's distance to the test data over its distance to the validation set, above which it "
            "drifts: 1.4 flags distances 40% above the baseline. Unset uses DataEval's default (1.4)."
        ),
    )
    n_features: int | None = Field(
        default=None, gt=0, description="Embedding dimensions to compare. Unset infers them from the embeddings."
    )
    chunking: ChunkedDriftConfig | None = Field(default=None, description=_CHUNKING_DESCRIPTION)


class DriftDomainClassifierConfig(EvaluatorConfig[DriftDomainClassifierResult]):
    """Config for ``shift.drift-domain-classifier``, DataEval's DriftDomainClassifier.

    Trains a classifier to tell the reference from the test data under cross-validation, and declares drift when it
    tells them apart better than ``threshold`` (AUROC). Needs an extractor on the task, and two sources: the
    reference, then the data to test.

    Every parameter, its DataEval argument and its unset behaviour is listed in the Evaluator Catalog
    (``reference/evaluators``), and ``dataeval-flow evaluators shift.drift-domain-classifier`` prints the JSON
    Schema.

    Example YAML::

        evaluators:
          - name: classifier_drift
            type: shift.drift-domain-classifier
            threshold: 0.6
    """

    type: str = Field(
        default="shift.drift-domain-classifier",
        description="The evaluator type this entry configures: `shift.drift-domain-classifier`.",
    )
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.EMBEDDINGS}), sources=SourceCount.TWO)

    n_folds: int | None = Field(
        default=None, ge=2, description="Cross-validation folds. Unset uses DataEval's default (5)."
    )
    threshold: float | tuple[float, float] | None = Field(
        default=None,
        description=(
            "AUROC above which drift is declared; with `chunking`, a (lower, upper) pair of AUROC bounds. Unset uses "
            "DataEval's default (0.55)."
        ),
    )
    chunking: ChunkedDriftConfig | None = Field(default=None, description=_CHUNKING_DESCRIPTION)
