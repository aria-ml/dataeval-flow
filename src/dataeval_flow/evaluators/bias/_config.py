"""Configs for the ``bias`` evaluators: DataEval's Balance, Diversity and Parity.

Field names are DataEval's argument names. An unset field is not passed, so DataEval's own default applies. Each
reads the source's metadata under the metadata policy ``metadata:`` names.
"""

__all__ = ["BalanceConfig", "DiversityConfig", "MetadataSummaryConfig", "ParityConfig"]

from typing import ClassVar, Literal

from pydantic import Field

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin
from dataeval_flow.evaluators._base import EvaluatorConfig
from dataeval_flow.evaluators.bias._result import (
    BalanceResult,
    DiversityResult,
    MetadataSummaryResult,
    ParityResult,
)


class BalanceConfig(EvaluatorConfig[BalanceResult], MetadataConfigMixin):
    """Config for ``balance``, DataEval's Balance.

    Measures the mutual information between each metadata factor and the class labels, and between pairs of factors.
    A factor that predicts the class is a shortcut a model can learn.

    Every parameter, its DataEval argument and its unset behaviour is listed in the Evaluator Catalog
    (``reference/evaluators``), and ``dataeval-flow evaluators balance`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: balance
            type: balance
            metadata: standard
            class_imbalance_threshold: 0.2
    """

    type: str = Field(default="balance", description="The evaluator type this entry configures: `balance`.")
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.METADATA}), sources=SourceCount.ONE)

    num_neighbors: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Neighborhood size for the estimator that scores measured factors; no effect on factors read as codes. "
            "Unset uses DataEval's default (5)."
        ),
    )
    class_imbalance_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Normalized mutual information above which a class counts as imbalanced against a factor. Unset uses "
            "DataEval's default (0.3)."
        ),
    )
    factor_correlation_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Normalized mutual information above which two factors count as highly correlated. Unset uses "
            "DataEval's default (0.5)."
        ),
    )
    label: str | list[str] | None = Field(
        default=None,
        description=(
            "Factor, or factors, to condition on instead of the class labels; several are combined into one "
            "composite axis. Unset reads the class labels."
        ),
    )
    factor_source: Literal["coded", "values", "auto"] | None = Field(
        default=None,
        description=(
            "Which representation of each factor to score: `coded` reads the codes binning produced, `values` the "
            "measurements, and `auto` decides per factor, reading codes wherever a cut was declared or accepted. "
            "Unset reads the metadata policy's `factor_source` where `metadata:` names a policy that sets one, else "
            "DataEval's default (`auto`)."
        ),
    )


class DiversityConfig(EvaluatorConfig[DiversityResult], MetadataConfigMixin):
    """Config for ``diversity``, DataEval's Diversity.

    Measures how evenly each metadata factor's values are spread, overall and within each class. A factor whose
    values crowd into few bins under-represents the rest.

    Every parameter, its DataEval argument and its unset behaviour is listed in the Evaluator Catalog
    (``reference/evaluators``), and ``dataeval-flow evaluators diversity`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: diversity
            type: diversity
            method: shannon
    """

    type: str = Field(default="diversity", description="The evaluator type this entry configures: `diversity`.")
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.METADATA}), sources=SourceCount.ONE)

    method: Literal["simpson", "shannon"] | None = Field(
        default=None,
        description=(
            "The diversity index: `simpson`, rescaled so 1 is an even spread and 0 all values in one bin, or "
            "`shannon`. Unset uses DataEval's default (`simpson`)."
        ),
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Diversity at or below which a factor counts as low. Unset uses DataEval's default (0.5).",
    )
    label: str | list[str] | None = Field(
        default=None,
        description=(
            "Factor, or factors, to condition on instead of the class labels; several are combined into one "
            "composite axis. Unset reads the class labels."
        ),
    )


class ParityConfig(EvaluatorConfig[ParityResult], MetadataConfigMixin):
    """Config for ``parity``, DataEval's Parity.

    Tests each metadata factor for association with the class labels (chi-squared, reported as Cramér's V). A factor
    significantly associated with the class is a shortcut a model can learn.

    Every parameter, its DataEval argument and its unset behaviour is listed in the Evaluator Catalog
    (``reference/evaluators``), and ``dataeval-flow evaluators parity`` prints the JSON Schema.

    Example YAML::

        evaluators:
          - name: parity
            type: parity
            p_value_threshold: 0.01
    """

    type: str = Field(default="parity", description="The evaluator type this entry configures: `parity`.")
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.METADATA}), sources=SourceCount.ONE)

    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Cramér's V above which a factor significant under `p_value_threshold` counts as highly correlated with "
            "the class labels. Unset uses DataEval's default (0.3)."
        ),
    )
    p_value_threshold: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="p-value below which a factor's association is significant. Unset uses DataEval's default (0.05).",
    )
    label: str | list[str] | None = Field(
        default=None,
        description=(
            "Factor, or factors, to condition on instead of the class labels; several are combined into one "
            "composite axis. Unset reads the class labels."
        ),
    )


class MetadataSummaryConfig(EvaluatorConfig[MetadataSummaryResult], MetadataConfigMixin):
    """Config for ``metadata-summary``: each metadata factor's type, binning, nulls, and range or top values.

    A Flow-only evaluator over DataEval's ``Metadata``, as legacy data-coverage's Metadata Distribution was; audit
    reuses it.

    Example YAML::

        evaluators:
          - name: summary
            type: metadata-summary
            metadata: standard
    """

    type: str = Field(
        default="metadata-summary", description="The evaluator type this entry configures: `metadata-summary`."
    )
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.METADATA}), sources=SourceCount.ONE)
