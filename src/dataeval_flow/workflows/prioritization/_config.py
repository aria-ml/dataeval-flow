"""The ``prioritization`` preset's config."""

from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.workflows._base import WorkflowConfig

__all__ = ["PrioritizationSettings", "PrioritizationWorkflowConfig", "SelectSettings"]

MethodType = Literal["knn", "kmeans_distance", "kmeans_complexity", "hdbscan_distance", "hdbscan_complexity"]
OrderType = Literal["easy_first", "hard_first"]
PolicyType = Literal["difficulty", "stratified", "class_balanced"]


class PrioritizationSettings(BaseModel):
    """The `prioritization` step's settings: how each pool is ranked against the reference."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    method: MethodType | None = Field(
        default=None,
        description=(
            "Ranking method: knn, kmeans_distance, kmeans_complexity, hdbscan_distance, hdbscan_complexity. "
            "Unset is DataEval's default, knn."
        ),
    )
    k: int | None = Field(
        default=None,
        gt=0,
        description="Number of nearest neighbors for knn method. None = sqrt(n_samples).",
    )
    c: int | None = Field(
        default=None,
        gt=0,
        description="Number of clusters for clustering methods. None = sqrt(n_samples).",
    )
    n_init: int | Literal["auto"] | None = Field(
        default=None,
        description="Number of K-means initializations (kmeans methods only). Unset is DataEval's default, auto.",
    )
    max_cluster_size: int | None = Field(
        default=None,
        gt=0,
        description="Maximum cluster size for HDBSCAN methods.",
    )
    order: OrderType = Field(
        default="hard_first",
        description=(
            "Sort direction: easy_first (prototypical first) or hard_first (novel/challenging first). "
            "`hard_first` unless set; DataEval's default is `easy_first`."
        ),
    )
    policy: PolicyType | None = Field(
        default=None,
        description=(
            "Selection policy: difficulty (direct ordering), stratified (binned), class_balanced. "
            "Unset is DataEval's default, difficulty."
        ),
    )
    num_bins: int | None = Field(
        default=None,
        gt=0,
        description="Number of bins for stratified policy. Unset is DataEval's default, 50.",
    )


class SelectSettings(BaseModel):
    """The `select` step's settings: how much of each pool's ranking is kept."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    n: int | None = Field(
        default=None,
        ge=1,
        description=(
            "How many of each pool's ranked items `selected` keeps. With `fraction` also unset, it keeps them all."
        ),
    )
    fraction: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "The share of each pool's ranked items `selected` keeps, rounded up. "
            "With `n` also unset, it keeps them all."
        ),
    )

    @model_validator(mode="after")
    def _one_amount(self) -> Self:
        if self.n is not None and self.fraction is not None:
            raise ValueError("`select` takes `n:` or `fraction:`, not both.")
        return self


class PrioritizationWorkflowConfig(WorkflowConfig[ChainResult]):
    """The settings of one ``prioritization`` entry: how data is ranked against a reference.

    Requires at least two sources: the first is the reference (labeled) dataset,
    and each later source is a pool, ranked against it on its own.

    Example YAML::

        workflows:
          - name: prioritize_knn
            type: prioritization
            prioritization:
              method: knn
              k: 10
              order: hard_first
            select:
              n: 200

    To clean the data first, run the preset as a step of a custom workflow, after a ``quality`` step or the
    ``outliers``, ``duplicates`` and ``remove`` steps.
    """

    type: str = Field(
        default="prioritization", description="The workflow type this entry configures: `prioritization`."
    )

    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.EMBEDDINGS}),
        sources=SourceCount.TWO_OR_MORE,
    )

    prioritization: PrioritizationSettings = Field(
        default_factory=PrioritizationSettings, description="The `prioritization` step's settings."
    )
    # --- Selection ---
    select: SelectSettings = Field(default_factory=SelectSettings, description="The `select` step's settings.")
