"""The ``data-prioritization`` preset's config."""

from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.workflows._base import WorkflowConfig

__all__ = ["DataPrioritizationConfig", "SelectSettings"]

MethodType = Literal["knn", "kmeans_distance", "kmeans_complexity", "hdbscan_distance", "hdbscan_complexity"]
OrderType = Literal["easy_first", "hard_first"]
PolicyType = Literal["difficulty", "stratified", "class_balanced"]


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


class DataPrioritizationConfig(WorkflowConfig[ChainResult]):
    """The settings of one ``data-prioritization`` entry: how data is ranked against a reference.

    Requires at least two sources: the first is the reference (labeled) dataset,
    and each later source is a pool, ranked against it on its own.

    Example YAML::

        workflows:
          - name: prioritize_knn
            type: data-prioritization
            method: knn
            k: 10
            order: hard_first
            select:
              n: 200

    To clean the data first, run the preset as a step of a custom workflow, after a ``data-cleaning`` step or the
    ``outliers``, ``duplicates`` and ``remove`` steps.
    """

    type: str = Field(
        default="data-prioritization", description="The workflow type this entry configures: `data-prioritization`."
    )

    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.EMBEDDINGS}),
        sources=SourceCount.TWO_OR_MORE,
    )

    # --- Prioritization method ---
    method: MethodType = Field(
        default="knn",
        description="Ranking method: knn, kmeans_distance, kmeans_complexity, hdbscan_distance, hdbscan_complexity.",
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
    n_init: int | Literal["auto"] = Field(
        default="auto",
        description="Number of K-means initializations (kmeans methods only).",
    )
    max_cluster_size: int | None = Field(
        default=None,
        gt=0,
        description="Maximum cluster size for HDBSCAN methods.",
    )

    # --- Order and policy ---
    order: OrderType = Field(
        default="hard_first",
        description="Sort direction: easy_first (prototypical first) or hard_first (novel/challenging first).",
    )
    policy: PolicyType = Field(
        default="difficulty",
        description="Selection policy: difficulty (direct ordering), stratified (binned), class_balanced.",
    )
    num_bins: int = Field(
        default=50,
        gt=0,
        description="Number of bins for stratified policy.",
    )

    # --- Selection ---
    select: SelectSettings = Field(default_factory=SelectSettings, description="The `select` step's settings.")
