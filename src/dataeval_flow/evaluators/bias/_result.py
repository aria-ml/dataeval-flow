"""The bias evaluators' results: DataEval's own output objects, typed per evaluator."""

from collections.abc import Mapping, Sequence
from typing import Any

from dataeval.bias import BalanceOutput, DiversityOutput, ParityOutput

from dataeval_flow._blocks import Block
from dataeval_flow.evaluators._core import CoreOutput
from dataeval_flow.evaluators._result import EvaluatorResult

__all__ = [
    "BalanceResult",
    "DiversityResult",
    "FactorSummaryOutput",
    "FactorSummaryResult",
    "ParityResult",
]


class BalanceResult(EvaluatorResult[BalanceOutput]):
    """The result of a ``balance`` run: ``output`` is DataEval's ``BalanceOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``BalanceOutput``: ``balance``, each factor's mutual information with the class labels;
        ``factors``, each factor pair's; and ``classwise``, each class's against each factor. ``to_dict()`` writes
        them as tables under ``data``.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """

    def _section(self, output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block] | None:  # noqa: ARG002
        """Each factor ranked by its mutual information with the class, then the rest."""
        from dataeval_flow.evaluators.bias._report import balance_section

        return balance_section(output, detailed=detailed)


class DiversityResult(EvaluatorResult[DiversityOutput]):
    """The result of a ``diversity`` run: ``output`` is DataEval's ``DiversityOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``DiversityOutput``: ``factors``, each factor's diversity index and whether it is low, and
        ``classwise``, each class's per factor. ``to_dict()`` writes them as tables under ``data``.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """


class ParityResult(EvaluatorResult[ParityOutput]):
    """The result of a ``parity`` run: ``output`` is DataEval's ``ParityOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``ParityOutput``: ``factors``, each factor's Cramér's V against the class labels with its p-value
        and whether it is significant, and ``insufficient_data``, the factor values with too few samples per class.
        ``to_dict()`` writes them under ``data``.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """


class FactorSummaryOutput(CoreOutput):
    """``factor-summary``'s output: each metadata factor's type, binning, nulls and range or top values.

    ``data()`` holds ``factors``, the kept factor names, and ``summary``, each factor's type, level, binning, nulls, and
    its range or top values, and each dropped factor with its reasons.
    """


class FactorSummaryResult(EvaluatorResult[FactorSummaryOutput]):
    """The result of a ``factor-summary`` run; ``output`` is a
    :class:`~dataeval_flow.evaluators.bias.FactorSummaryOutput`.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        ``data()`` holds ``factors``, the kept factor names, and ``summary``, each factor's statistics by name.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """

    def _section(self, output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block] | None:  # noqa: ARG002
        """The kept factors' table: type, unique values or mean, nulls."""
        from dataeval_flow.evaluators.bias._report import metadata_summary_section

        return metadata_summary_section(output)
