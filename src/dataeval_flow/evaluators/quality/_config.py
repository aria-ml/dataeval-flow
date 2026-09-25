"""Configs for the ``quality`` evaluators: DataEval's Duplicates and Outliers.

Field names are DataEval's argument names. An unset field is not passed, so DataEval's own
default applies. Each model constructs its DataEval evaluator when it validates, so an
argument DataEval refuses fails the config load with DataEval's own message.
"""

__all__ = ["DuplicatesConfig", "OutliersConfig", "ThresholdSpec"]

import functools
import operator
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, Self, TypeVar

from pydantic import Field, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config._schemas._mixins import StatsConfigMixin
from dataeval_flow.evaluators._base import EvaluatorConfig
from dataeval_flow.evaluators.quality._result import DuplicatesResult, OutliersResult

if TYPE_CHECKING:
    from dataeval.flags import ImageStats
    from dataeval.quality import Duplicates, Outliers

# One bound, or a (lower, upper) pair.
Bounds = float | tuple[float | None, float | None]
# Limits a bound is clipped to, as (lower, upper).
Limits = tuple[float | None, float | None]
# DataEval's ``ThresholdLike``, less the ``Threshold`` objects a config cannot build: a
# method name, bounds, or ``[method, bounds]``, ``[method, bounds, limits]``, ``[bounds, limits]``.
ThresholdSpec = (
    str | Bounds | tuple[str, Bounds | None] | tuple[str, Bounds | None, Limits] | tuple[Bounds | None, Limits]
)

_QUALITY_INPUTS = InputSpec(
    required=frozenset({InputKind.STATS}),
    optional=frozenset({InputKind.CLUSTERS}),
    sources=SourceCount.ONE_OR_MORE,
)


R = TypeVar("R")


class _QualityConfig(EvaluatorConfig[R], StatsConfigMixin, Generic[R]):
    """What both quality evaluators share: the stats policy, clustering, and ``from_stats``'s scope."""

    inputs: ClassVar[InputSpec] = _QUALITY_INPUTS

    cluster_algorithm: Literal["kmeans", "hdbscan"] | None = Field(
        default=None,
        description="Clustering algorithm for cluster mode. Unset uses DataEval's default (`hdbscan`).",
    )
    n_clusters: int | None = Field(
        default=None,
        gt=0,
        description="Expected number of clusters for cluster mode. Unset lets DataEval choose.",
    )
    per_image: bool | None = Field(
        default=None,
        description="Passed to `from_stats`: evaluate whole images. Unset uses DataEval's default.",
    )
    per_target: bool | None = Field(
        default=None,
        description=(
            "Passed to `from_stats`: evaluate individual targets (detections). Unset uses DataEval's default."
        ),
    )

    @abstractmethod
    def cluster_mode(self) -> bool:
        """Whether these values ask for cluster-based detection."""

    @abstractmethod
    def stats_flags(self) -> "ImageStats":
        """The statistics families this run reads: the configured ones, else DataEval's default."""

    @abstractmethod
    def stats_request(self) -> dict[str, Any]:
        """Keyword arguments for ``stats_policy_for``: the families, under the consumer that reads them."""

    @abstractmethod
    def constructor_kwargs(self) -> dict[str, Any]:
        """Constructor arguments for the DataEval evaluator, leaving unset ones to DataEval."""

    @abstractmethod
    def dataeval_evaluator(self) -> "Duplicates | Outliers":
        """The DataEval evaluator these values configure."""

    def call_kwargs(self) -> dict[str, bool]:
        """Keyword arguments for ``from_stats``, leaving unset ones to DataEval."""
        values = {"per_image": self.per_image, "per_target": self.per_target}
        return {key: value for key, value in values.items() if value is not None}

    def wanted_kinds(self) -> frozenset[InputKind]:
        """Stats always, and clusters too in cluster mode."""
        clusters = frozenset({InputKind.CLUSTERS}) if self.cluster_mode() else frozenset()
        return self.inputs.required | clusters

    def check_inputs(self, count: int) -> str | None:
        """Cluster mode reads one source, because DataEval's ``from_clusters`` takes one cluster result."""
        if self.cluster_mode() and count > 1:
            return "reads exactly one source in cluster mode; name one source, or leave the cluster parameters unset."
        return None

    @model_validator(mode="after")
    def _dataeval_accepts(self) -> Self:
        try:
            self.dataeval_evaluator()
        except (TypeError, ValueError) as e:
            raise ValueError(f"DataEval rejected these parameters: {e}") from e
        return self


class DuplicatesConfig(_QualityConfig[DuplicatesResult]):
    """Config for ``quality.duplicates``, DataEval's Duplicates.

    Determines which images are exact or near duplicates of each other, across every source
    the task names. Setting ``cluster_sensitivity`` adds embedding-space near duplicates,
    which needs an extractor and a single source.

    Every parameter, its DataEval argument and its unset behaviour is listed in the
    Evaluator Catalog (``reference/evaluators``), and
    ``dataeval-flow evaluators quality.duplicates`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: dupes
            type: quality.duplicates
            flags: [hash_basic, hash_d4]
    """

    type: str = Field(
        default="quality.duplicates", description="The evaluator type this entry configures: `quality.duplicates`."
    )
    flags: Sequence[Literal["hash_basic", "hash_d4"]] | None = Field(
        default=None,
        min_length=1,
        description="Hash families to compare. Unset uses DataEval's default (`hash_basic`).",
    )
    merge_near_duplicates: bool | None = Field(
        default=None,
        description=(
            "Merge overlapping near-duplicate groups found by different methods. Unset uses DataEval's default."
        ),
    )
    hash_radius: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Maximum Hamming distance, in bits, for two perceptual hashes to be treated as near duplicates. "
            "Unset uses DataEval's default (0, documented to change in a future major release)."
        ),
    )
    cluster_sensitivity: float | None = Field(
        default=None,
        description=(
            "Enables cluster mode: embedding-space near duplicates at this sensitivity, merged with the hash "
            "groups. Needs an extractor on the task, and reads one source."
        ),
    )

    def cluster_mode(self) -> bool:
        """Whether ``cluster_sensitivity`` is set."""
        return self.cluster_sensitivity is not None

    def stats_flags(self) -> "ImageStats":
        """The hash families to compute: the configured ones, else ``Duplicates.Config().flags``."""
        from dataeval.quality import Duplicates

        from dataeval_flow._stats import HASH_FLAG_MAP

        if self.flags is None:
            return Duplicates.Config().flags
        return functools.reduce(operator.or_, (HASH_FLAG_MAP[name] for name in self.flags))

    def stats_request(self) -> dict[str, Any]:
        """The hash families, requested as the duplicate consumer and blamed on ``flags``."""
        return {"duplicate_flags": self.stats_flags(), "duplicate_declaration": "`flags`"}

    def constructor_kwargs(self) -> dict[str, Any]:
        """Arguments for ``Duplicates(...)``, leaving unset ones to DataEval."""
        values = {
            "flags": self.stats_flags() if self.flags is not None else None,
            "merge_near_duplicates": self.merge_near_duplicates,
            "hash_radius": self.hash_radius,
            "cluster_sensitivity": self.cluster_sensitivity,
            "cluster_algorithm": self.cluster_algorithm,
            "n_clusters": self.n_clusters,
        }
        return {key: value for key, value in values.items() if value is not None}

    def dataeval_evaluator(self) -> "Duplicates":
        """``Duplicates`` configured by these values."""
        from dataeval.quality import Duplicates

        return Duplicates(**self.constructor_kwargs())


class OutliersConfig(_QualityConfig[OutliersResult]):
    """Config for ``quality.outliers``, DataEval's Outliers.

    Determines which images' statistics sit outside ``outlier_threshold``, across every
    source the task names. Setting ``cluster_threshold`` adds embedding-space outliers,
    which needs an extractor and a single source.

    Every parameter, its DataEval argument and its unset behaviour is listed in the
    Evaluator Catalog (``reference/evaluators``), and
    ``dataeval-flow evaluators quality.outliers`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: outliers
            type: quality.outliers
            flags: [pixel, visual]
            outlier_threshold: [zscore, 3.0]
    """

    type: str = Field(
        default="quality.outliers", description="The evaluator type this entry configures: `quality.outliers`."
    )
    flags: Sequence[Literal["dimension", "pixel", "visual"]] | None = Field(
        default=None,
        min_length=1,
        description="Statistics families to test. Unset uses DataEval's default.",
    )
    outlier_threshold: ThresholdSpec | Mapping[str, ThresholdSpec] | None = Field(
        default=None,
        description=(
            "How far a statistic must sit from the rest to be an outlier: a method (`zscore`, `modzscore`, "
            "`iqr`, `adaptive`, `constant`), `[method, bounds]`, or a mapping from metric name to either. "
            "Unset uses DataEval's default."
        ),
    )
    cluster_threshold: ThresholdSpec | None = Field(
        default=None,
        description=(
            "Enables cluster mode: embedding-space outliers at this threshold, merged with the statistical "
            "ones. Needs an extractor on the task, and reads one source."
        ),
    )

    def cluster_mode(self) -> bool:
        """Whether ``cluster_threshold`` is set."""
        return self.cluster_threshold is not None

    def stats_flags(self) -> "ImageStats":
        """The statistics families to test: the configured ones, else ``Outliers.Config().flags``."""
        from dataeval.quality import Outliers

        from dataeval_flow._stats import OUTLIER_FLAG_MAP

        if self.flags is None:
            return Outliers.Config().flags
        return functools.reduce(operator.or_, (OUTLIER_FLAG_MAP[name] for name in self.flags))

    def stats_request(self) -> dict[str, Any]:
        """The families, requested as the outlier consumer."""
        return {"outlier_flags": self.stats_flags()}

    def constructor_kwargs(self) -> dict[str, Any]:
        """Arguments for ``Outliers(...)``, leaving unset ones to DataEval."""
        values = {
            "flags": self.stats_flags() if self.flags is not None else None,
            "outlier_threshold": self.outlier_threshold,
            "cluster_threshold": self.cluster_threshold,
            "cluster_algorithm": self.cluster_algorithm,
            "n_clusters": self.n_clusters,
        }
        return {key: value for key, value in values.items() if value is not None}

    def dataeval_evaluator(self) -> "Outliers":
        """``Outliers`` configured by these values."""
        from dataeval.quality import Outliers

        return Outliers(**self.constructor_kwargs())
