"""The ``data-splitting`` preset's config: the split, its rebalancing, its coverage, and when a finding warns."""

__all__ = ["DataSplittingConfig", "DataSplittingChecks", "DataSplittingCoverageSettings"]

from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps.checks._stratification import StratificationThresholds
from dataeval_flow.workflows._base import WorkflowConfig


class DataSplittingCoverageSettings(BaseModel):
    """The `coverage` steps' settings, as a `coverage` evaluator entry reads them, with legacy data-splitting's
    defaults; each keeps its default when `coverage:` is written partly."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    method: Literal["naive", "adaptive"] | None = Field(
        default=None,
        description=(
            "How the coverage radius is set: `naive`, a fixed analytic radius, or `adaptive`, a cutoff on the "
            "`percent` most sparsely neighbored items. Unset uses DataEval's default (`adaptive`). Only `naive` "
            "coverage is judged, by `uncovered-items` steps. DataEval's naive radius overflows past about 340 "
            "embedding dimensions, so `naive` suits low-dimensional embeddings; with a wide extractor its coverage "
            "steps are skipped with `failed: OverflowError`."
        ),
    )
    num_observations: int = Field(
        default=50,
        gt=0,
        description=(
            "Neighbors an item needs within the radius to count as covered, fewer than the smallest part's items."
        ),
    )
    percent: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description=("Fraction of each part's items flagged as uncovered, for `adaptive` only."),
    )


class DataSplittingClassImbalanceSettings(BaseModel):
    """The `class-imbalance` check's field, with legacy data-splitting's default."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: float | None = Field(
        default=10.0,
        ge=1.0,
        description=(
            "Largest class count over smallest that may hold before the whole set's Class Imbalance finding warns; "
            "`null` judges nothing but an empty class."
        ),
    )


class DataSplittingUncoveredItemsSettings(BaseModel):
    """The `uncovered-items` check's field, with legacy data-splitting's default."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: float | None = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description=(
            "The percent of a Dataset's items uncovered past which its Uncovered Items finding warns, under `naive` "
            "coverage only; `null` judges nothing."
        ),
    )


class DataSplittingChecks(BaseModel):
    """When data-splitting's findings warn: each check's fields, keyed by check type."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    class_imbalance: DataSplittingClassImbalanceSettings = Field(
        default_factory=DataSplittingClassImbalanceSettings,
        alias="class-imbalance",
        description="The `class-imbalance` check's thresholds, on the whole set.",
    )
    stratification: StratificationThresholds = Field(
        default_factory=StratificationThresholds, description="The `stratification` check's thresholds, on each fold."
    )
    uncovered_items: DataSplittingUncoveredItemsSettings = Field(
        default_factory=DataSplittingUncoveredItemsSettings,
        alias="uncovered-items",
        description="The `uncovered-items` check's thresholds, on the whole set and each part, under `naive` coverage.",
    )


class DataSplittingConfig(WorkflowConfig[ChainResult], MetadataConfigMixin):
    """The settings of one ``data-splitting`` entry: the split or folds, rebalancing, coverage, and when a finding
    warns.

    Example YAML::

        workflows:
          - name: splits
            type: data-splitting
            folds: 3
            test_frac: 0.2
            rebalance: interclass
    """

    type: str = Field(
        default="data-splitting", description="The workflow type this entry configures: `data-splitting`."
    )
    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.METADATA}),
        optional=frozenset({InputKind.EMBEDDINGS}),
        sources=SourceCount.ONE,
    )

    folds: int = Field(
        default=1,
        ge=1,
        description=(
            "`1` splits once into train, val and test; 2 or more make that many train and val folds and one test."
        ),
    )
    test_frac: float = Field(
        default=0.2, ge=0.0, lt=1.0, description="The share held out as `test`; `0` holds out none."
    )
    val_frac: float | None = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description=(
            "The share held out as `val` with `folds: 1`; unset is 0.1. With `folds` 2 or more, each fold's val is "
            "its 1/k, and setting this is refused."
        ),
    )
    stratify: bool = Field(default=True, description="Whether each part keeps the whole's class proportions.")
    split_on: list[str] | None = Field(
        default=None,
        description=(
            "Metadata factors whose values never straddle parts, such as a scene or site. Classification data only: "
            "DataEval ignores it on detection data, with a warning in the log."
        ),
    )
    rebalance: Literal["global", "interclass"] | None = Field(
        default=None,
        description="DataEval's `ClassBalance` method applied to each train; unset rebalances nothing.",
    )
    coverage: DataSplittingCoverageSettings = Field(
        default_factory=DataSplittingCoverageSettings,
        description=(
            "The `coverage` steps' settings, run on the whole set and each part when the task names an extractor."
        ),
    )
    checks: DataSplittingChecks = Field(
        default_factory=DataSplittingChecks, description="When findings warn, keyed by check type."
    )

    @model_validator(mode="after")
    def _val_frac_with_one_fold(self) -> Self:
        if self.val_frac is not None and self.folds >= 2:
            raise ValueError(
                f"Entry '{self.name}' sets `val_frac`, which applies with `folds: 1` only: with `folds: {self.folds}`, "
                f"`kfold` holds out each fold's 1/{self.folds} as its val. Remove `val_frac`."
            )
        return self
