"""The ``data-splitting`` workflow's config."""

from typing import ClassVar, Literal

from pydantic import Field

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.workflows._base import WorkflowConfig
from dataeval_flow.workflows.data_splitting._outputs import DataSplittingResult

__all__ = ["DataSplittingConfig"]


class DataSplittingConfig(WorkflowConfig[DataSplittingResult]):
    """The settings of one ``data-splitting`` entry: the folds and fractions, stratification and rebalancing.

    Controls how the dataset is split into train/val/test partitions and
    what assessments are run on each split.

    Example YAML::

        workflows:
          - name: split_stratified
            type: data-splitting
            test_frac: 0.2
            val_frac: 0.1
            stratify: true
    """

    type: str = Field(
        default="data-splitting", description="The workflow type this entry configures: `data-splitting`."
    )

    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.METADATA}),
        optional=frozenset({InputKind.EMBEDDINGS}),
        sources=SourceCount.ONE,
    )

    test_frac: float = Field(
        default=0.2,
        ge=0.0,
        lt=1.0,
        description="Fraction of the dataset held out for the test set.",
    )
    val_frac: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description="Fraction of training data reserved for validation (single-fold).",
    )
    num_folds: int = Field(
        default=1,
        ge=1,
        description="Number of train/val folds. If 1, val_frac must be > 0.",
    )
    stratify: bool = Field(
        default=True,
        description="Preserve class distribution within each partition.",
    )
    split_on: list[str] | None = Field(
        default=None,
        description="Metadata keys to group on so no group spans train/val.",
    )
    rebalance_method: Literal["global", "interclass"] | None = Field(
        default=None,
        description="ClassBalance method for train split. None = no rebalancing.",
    )
    coverage_percent: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description=(
            "Proportion of observations considered uncovered for coverage_adaptive "
            "(when model provided). Per h2_detect_undersampling tutorial."
        ),
    )
    num_observations: int = Field(
        default=50,
        ge=1,
        description="Number of neighbors for coverage_adaptive (when model provided).",
    )
