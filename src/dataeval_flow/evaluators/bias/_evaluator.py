"""The bias evaluators: DataEval's Balance, Diversity and Parity, each over one source's metadata.

``run`` is the only code here that calls DataEval.
"""

__all__ = ["BalanceEvaluator", "DiversityEvaluator", "ParityEvaluator", "balance_arguments"]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from dataeval.bias import Balance, BalanceOutput, Diversity, DiversityOutput, Parity, ParityOutput

from dataeval_flow._input_spec import InputKind
from dataeval_flow.evaluators._evaluator import Evaluator
from dataeval_flow.evaluators._fields import dataeval_arguments, require
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators.bias._config import BalanceConfig, DiversityConfig, ParityConfig

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
