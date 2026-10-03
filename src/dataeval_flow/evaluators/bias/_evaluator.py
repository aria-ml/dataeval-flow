"""The bias evaluators: DataEval's Balance, Diversity and Parity, each over one source's metadata.

``run`` is the only code here that calls DataEval.
"""

__all__ = ["BalanceEvaluator", "DiversityEvaluator", "MetadataSummaryEvaluator", "ParityEvaluator", "balance_arguments"]

import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from dataeval import Metadata
from dataeval.bias import Balance, BalanceOutput, Diversity, DiversityOutput, Parity, ParityOutput

from dataeval_flow._input_spec import InputKind
from dataeval_flow.evaluators._core import execution
from dataeval_flow.evaluators._evaluator import Evaluator
from dataeval_flow.evaluators._fields import dataeval_arguments, require
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators.bias._config import (
    BalanceConfig,
    DiversityConfig,
    MetadataSummaryConfig,
    ParityConfig,
)
from dataeval_flow.evaluators.bias._result import MetadataSummaryOutput

if TYPE_CHECKING:
    from dataeval_flow._policy import ResolvedPolicy

_EVALUATE: Mapping[InputKind, str] = {InputKind.METADATA: "evaluate"}


def balance_arguments(config: BalanceConfig, policy: "ResolvedPolicy | None") -> dict[str, Any]:
    """``Balance``'s arguments: the config's, with the metadata policy's ``factor_source`` where the config sets none.

    A shared policy then governs the evaluator's numbers as it governs ``data-analysis``'s, which passes the policy's
    ``factor_source`` to ``Balance``.
    """
    arguments = dataeval_arguments(config)
    if "factor_source" not in arguments and policy is not None and policy.factor_source is not None:
        arguments["factor_source"] = policy.factor_source
    return arguments


class BalanceEvaluator(Evaluator[BalanceConfig, BalanceOutput]):
    """``balance``: how much each metadata factor says about the class label, per DataEval's Balance."""

    name: ClassVar[str] = "balance"
    title: ClassVar[str] = "Balance"
    description: ClassVar[str] = "Mutual information between metadata factors and class labels (DataEval Balance)"
    dataeval_class: ClassVar[type] = Balance
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _EVALUATE

    def run(self, config: BalanceConfig, inputs: Sequence[EvaluatorInputs]) -> BalanceOutput:
        """Score the source's factors against its class labels."""
        (source,) = inputs
        metadata = require(source.metadata, "metadata", source.source)
        return Balance(**balance_arguments(config, source.metadata_policy)).evaluate(metadata)


class DiversityEvaluator(Evaluator[DiversityConfig, DiversityOutput]):
    """``diversity``: how evenly each metadata factor's values are spread, per DataEval's Diversity."""

    name: ClassVar[str] = "diversity"
    title: ClassVar[str] = "Diversity"
    description: ClassVar[str] = "How evenly metadata factor values are spread (DataEval Diversity)"
    dataeval_class: ClassVar[type] = Diversity
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _EVALUATE

    def run(self, config: DiversityConfig, inputs: Sequence[EvaluatorInputs]) -> DiversityOutput:
        """Measure each factor's spread, overall and per class."""
        (source,) = inputs
        metadata = require(source.metadata, "metadata", source.source)
        return Diversity(**dataeval_arguments(config)).evaluate(metadata)


class ParityEvaluator(Evaluator[ParityConfig, ParityOutput]):
    """``parity``: which metadata factors are associated with the class label, per DataEval's Parity."""

    name: ClassVar[str] = "parity"
    title: ClassVar[str] = "Parity"
    description: ClassVar[str] = "Association between metadata factors and class labels (DataEval Parity)"
    dataeval_class: ClassVar[type] = Parity
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = _EVALUATE

    def run(self, config: ParityConfig, inputs: Sequence[EvaluatorInputs]) -> ParityOutput:
        """Test each factor for association with the class labels."""
        (source,) = inputs
        metadata = require(source.metadata, "metadata", source.source)
        return Parity(**dataeval_arguments(config)).evaluate(metadata)


class MetadataSummaryEvaluator(Evaluator[MetadataSummaryConfig, MetadataSummaryOutput]):
    """``metadata-summary``: each factor of the source's Metadata summarized, as legacy data-coverage did."""

    name: ClassVar[str] = "metadata-summary"
    title: ClassVar[str] = "Metadata Summary"
    description: ClassVar[str] = "Each metadata factor's type, binning, nulls, and range or top values"
    dataeval_class: ClassVar[Any] = Metadata
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.METADATA: "rows_at"}

    def run(self, config: MetadataSummaryConfig, inputs: Sequence[EvaluatorInputs]) -> MetadataSummaryOutput:  # noqa: ARG002
        """Summarize the source's Metadata under the task's policy (coverage spec §6.1)."""
        from dataeval_flow.workflows._common import compute_metadata_summary

        (source,) = inputs
        metadata = require(source.metadata, "metadata", source.source)
        started, clock = datetime.now(UTC), time.monotonic()
        summary = compute_metadata_summary(metadata)
        meta = execution("dataeval_flow.compute_metadata_summary", started, time.monotonic() - clock, {})
        return MetadataSummaryOutput({"factors": list(metadata.factor_names), "summary": summary}, meta)
