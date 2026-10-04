"""Configs for the ``quality`` evaluators: DataEval's Duplicates and Outliers.

Field names are DataEval's argument names. An unset field is not passed, so DataEval's own
default applies. Each model constructs its DataEval evaluator when it validates, so an
argument DataEval refuses fails the config load with DataEval's own message.
"""

__all__ = ["ContentDigestConfig", "DuplicatesConfig", "FactorTriageConfig", "LabelHealthConfig", "OutliersConfig"]

import functools
import operator
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Generic, Literal, Self, TypeVar

from pydantic import Field, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin, StatsConfigMixin
from dataeval_flow.evaluators._base import EvaluatorConfig
from dataeval_flow.evaluators._threshold import ThresholdSpec
from dataeval_flow.evaluators.quality._result import (
    ContentDigestResult,
    DuplicatesResult,
    FactorTriageResult,
    LabelHealthResult,
    OutliersResult,
)

if TYPE_CHECKING:
    from dataeval.flags import ImageStats
    from dataeval.quality import Duplicates, Outliers

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
    """Config for ``duplicates``, DataEval's Duplicates.

    Determines which images are exact or near duplicates of each other, across every source
    the task names. Setting ``cluster_sensitivity`` adds embedding-space near duplicates,
    which needs an extractor and a single source.

    Every parameter, its DataEval argument and its unset behaviour is listed in the
    Evaluator Catalog (``reference/evaluators``), and
    ``dataeval-flow evaluators duplicates`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: dupes
            type: duplicates
            flags: [hash_basic, hash_d4]
    """

    type: str = Field(default="duplicates", description="The evaluator type this entry configures: `duplicates`.")
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
    redundancy_radius: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Maximum Hamming distance, in bits, for a video frame to count as carrying nothing new over the frame "
            "before it; such stretches are reported as `redundant` groups. No effect on image datasets. Unset uses "
            "DataEval's default (4)."
        ),
    )
    min_segment_frames: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Shortest stretch, in frames, two videos may share and still be reported as a `segment` row. No effect "
            "on image datasets. Unset uses DataEval's default (30)."
        ),
    )
    max_segment_gap: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Frames a shared stretch may skip and still count as continuous; too large a value bridges a cut. No "
            "effect on image datasets. Unset uses DataEval's default (5)."
        ),
    )
    segment_offset_tolerance: int | None = Field(
        default=None,
        ge=0,
        description=(
            "How far two stretches may differ in offset and still be joined into one; raise it where two videos were "
            "sampled at slightly different rates. No effect on image datasets. Unset uses DataEval's default (0)."
        ),
    )
    verify_alignment: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Mean bits per frame two videos may differ by along a warped alignment and still be reported as an "
            "`aligned` row; around 8 is a reasonable start, and the cost is quadratic in the videos' lengths. No "
            "effect on image datasets. Unset skips warped matching."
        ),
    )
    min_track_frames: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Shortest stretch, in detections, two tracks may share and still be reported at `level: track`. Read only "
            "with `per_target: true`. No effect on image datasets. Unset uses DataEval's default (5)."
        ),
    )
    frame_sample: Annotated[int, Field(ge=1)] | Annotated[float, Field(gt=0)] | None = Field(
        default=None,
        description=(
            "How much of each video to read: an integer is a stride in frames (`5` keeps every fifth), a float a "
            "target rate in frames per second. No effect on image datasets. Unset reads every frame."
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
            "redundancy_radius": self.redundancy_radius,
            "min_segment_frames": self.min_segment_frames,
            "max_segment_gap": self.max_segment_gap,
            "segment_offset_tolerance": self.segment_offset_tolerance,
            "verify_alignment": self.verify_alignment,
            "min_track_frames": self.min_track_frames,
            "frame_sample": self.frame_sample,
        }
        return {key: value for key, value in values.items() if value is not None}

    def dataeval_evaluator(self) -> "Duplicates":
        """``Duplicates`` configured by these values."""
        from dataeval.quality import Duplicates

        return Duplicates(**self.constructor_kwargs())


class OutliersConfig(_QualityConfig[OutliersResult]):
    """Config for ``outliers``, DataEval's Outliers.

    Determines which images' statistics sit outside ``outlier_threshold``, across every
    source the task names. Setting ``cluster_threshold`` adds embedding-space outliers,
    which needs an extractor and a single source.

    Every parameter, its DataEval argument and its unset behaviour is listed in the
    Evaluator Catalog (``reference/evaluators``), and
    ``dataeval-flow evaluators outliers`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: outliers
            type: outliers
            flags: [pixel, visual]
            outlier_threshold: [zscore, 3.0]
    """

    type: str = Field(default="outliers", description="The evaluator type this entry configures: `outliers`.")
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


class LabelHealthConfig(EvaluatorConfig[LabelHealthResult], MetadataConfigMixin):
    """Config for ``label-health``: how a Dataset's labels spread over its classes.

    Wraps ``dataeval.core.label_stats`` over the Dataset's metadata. It adds the number of classes the Dataset
    declares, seen or not, and where its labels came from. The ``class-imbalance`` and ``target-outlier-rate`` checks
    read it.

    Example YAML::

        evaluators:
          - name: labels
            type: label-health
    """

    type: str = Field(default="label-health", description="The evaluator type this entry configures: `label-health`.")
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.METADATA}), sources=SourceCount.ONE)


class ContentDigestConfig(EvaluatorConfig[ContentDigestResult]):
    """Config for ``content-digest``: SHA-256 digests of every item a Dataset holds.

    Reads each item from the Dataset itself, never through a cache, and digests its image and labels (the content
    digest, with the class names) and its metadata (the metadata digest), whatever the items' order.
    :func:`~dataeval_flow.dataset_digest` computes the same values, so a training job can check it holds the data a
    run read. It takes no settings.

    Example YAML::

        evaluators:
          - name: digest
            type: content-digest
    """

    type: str = Field(
        default="content-digest", description="The evaluator type this entry configures: `content-digest`."
    )
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.DATASET}), sources=SourceCount.ONE)


class FactorTriageConfig(EvaluatorConfig[FactorTriageResult], MetadataConfigMixin):
    """Config for ``factor-triage``: what a Dataset's metadata failed to read, and a policy that repairs it.

    Reads the Dataset's metadata under its policy and finds each factor the run could not read as configured. It
    suggests a correction or a bin count where one repairs it, and, with ``verify``, reads the metadata back under the
    suggestions. The ``metadata-issues`` check reads its output; ``metadata-triage`` runs both.

    Example YAML::

        evaluators:
          - name: triage
            type: factor-triage
            metadata: standard
    """

    type: str = Field(default="factor-triage", description="The evaluator type this entry configures: `factor-triage`.")
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.METADATA}), sources=SourceCount.ONE)

    verify: bool = Field(
        default=True,
        description=(
            "Re-read the metadata under the complete suggestions and report what they "
            "recover. Costs no second dataset walk: `repair` returns a copy sharing the store."
        ),
    )
    default_bins: int = Field(
        default=10,
        ge=2,
        description=(
            "Bin count a suggestion falls back to where the run left no fit to read. Where "
            "there is one, the populated bins of the derived cut are carried forward instead, "
            "which pins the cut the run used rather than substituting a different one."
        ),
    )
    min_missing_fraction: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Share of rows recording no value above which a factor is called degenerate.",
    )
