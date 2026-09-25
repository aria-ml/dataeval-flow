"""The ``parameter-sweep`` workflow's config."""

from collections.abc import Sequence
from typing import ClassVar, Literal

from pydantic import Field

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config._schemas._mixins import StatsConfigMixin
from dataeval_flow.workflows._base import WorkflowConfig, _LegacyValueRangeMixin
from dataeval_flow.workflows.parameter_sweep._outputs import ParameterSweepResult

__all__ = ["ParameterSweepConfig"]


class ParameterSweepConfig(WorkflowConfig[ParameterSweepResult], _LegacyValueRangeMixin, StatsConfigMixin):
    """The settings of one ``parameter-sweep`` entry: the ``data-cleaning`` values to try, each field a list of them.

    Every combination runs once, so the report shows how each outcome moves with the
    values that feed it.

    Example YAML::

        workflows:
          - name: sweep_outlier_threshold
            type: parameter-sweep
            outlier_method: [adaptive]
            outlier_threshold: [null, 1.0, 2.0, 3.0]
    """

    type: str = Field(
        default="parameter-sweep", description="The workflow type this entry configures: `parameter-sweep`."
    )

    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.STATS}),
        optional=frozenset({InputKind.CLUSTERS}),
        sources=SourceCount.ONE,
    )

    # --- Static params (shared across all runs in the sweep) ---
    outlier_flags: Sequence[Literal["dimension", "pixel", "visual"]] = Field(
        default=("dimension", "pixel", "visual"),
        min_length=1,
        description="Image statistics groups for outlier detection. At least one required.",
    )
    duplicate_flags: Sequence[Literal["hash_basic", "hash_d4"]] | None = Field(
        default=None,
        description=(
            "Hash flag groups for duplicate detection. None = DataEval default (hash_basic: xxhash + phash + dhash)."
        ),
    )
    duplicate_merge_near: bool = Field(
        default=True,
        description="Merge overlapping near-duplicate groups from different detection methods.",
    )

    # --- Sweep params (can be single values or sequences) ---
    # We use Sequence[T] for these to allow multiple values.
    # If a single value is desired, pass a sequence of length 1.

    outlier_method: Sequence[Literal["adaptive", "zscore", "modzscore", "iqr"]] = Field(
        default=("adaptive",),
        description="Statistical method(s) for outlier detection.",
    )
    outlier_threshold: Sequence[float | None] = Field(
        default=(None,),
        description="Custom threshold(s) (None = use DataEval default for chosen method).",
    )
    outlier_cluster_threshold: Sequence[float | None] = Field(
        default=(None,),
        description=("Std devs from cluster center to flag as outlier. None = skip cluster detection."),
    )
    outlier_cluster_algorithm: Sequence[Literal["kmeans", "hdbscan"] | None] = Field(
        default=(None,),
        description="Clustering algorithm(s) for cluster-based outlier detection.",
    )
    duplicate_cluster_sensitivity: Sequence[float | None] = Field(
        default=(None,),
        description=("Threshold(s) for cluster-based near duplicate detection. None = skip cluster detection."),
    )
    duplicate_cluster_algorithm: Sequence[Literal["kmeans", "hdbscan"] | None] = Field(
        default=(None,),
        description="Clustering algorithm(s) for cluster-based duplicate detection.",
    )
