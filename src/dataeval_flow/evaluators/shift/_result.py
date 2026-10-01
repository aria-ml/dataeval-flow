"""The shift evaluators' results: DataEval's own output objects, typed per evaluator."""

from collections.abc import Mapping, Sequence
from typing import Any

from dataeval.shift import DriftOutput, OODOutput

from dataeval_flow._blocks import Block
from dataeval_flow.evaluators._result import EvaluatorResult

__all__ = [
    "DriftDomainClassifierResult",
    "DriftKNeighborsResult",
    "DriftMMDResult",
    "DriftUnivariateResult",
    "DriftWassersteinResult",
    "OODDomainClassifierResult",
    "OODKNeighborsResult",
]


class _DriftSection:
    """The report section the five drift results share."""

    def _section(self, output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block] | None:  # noqa: ARG002
        """The verdict and statistics as fields, or one row per chunk."""
        from dataeval_flow.evaluators.shift._report import drift_section

        return drift_section(output)


class DriftUnivariateResult(_DriftSection, EvaluatorResult[DriftOutput[Any]]):
    """The result of a ``drift-univariate`` run: ``output`` is DataEval's ``DriftOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``DriftOutput``: ``drifted``, ``distance``, ``threshold``, ``metric_name``, ``feature_names``, and
        ``details``, the test's own statistics or, with ``chunking``, a table with one row per chunk. ``to_dict()``
        writes them under ``data``.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """


class DriftMMDResult(_DriftSection, EvaluatorResult[DriftOutput[Any]]):
    """The result of a ``drift-mmd`` run: ``output`` is DataEval's ``DriftOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``DriftOutput``: ``drifted``, ``distance``, ``threshold``, ``metric_name``, ``feature_names``, and
        ``details``, the test's own statistics or, with ``chunking``, a table with one row per chunk. ``to_dict()``
        writes them under ``data``.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """


class DriftKNeighborsResult(_DriftSection, EvaluatorResult[DriftOutput[Any]]):
    """The result of a ``drift-kneighbors`` run: ``output`` is DataEval's ``DriftOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``DriftOutput``: ``drifted``, ``distance``, ``threshold``, ``metric_name``, ``feature_names``, and
        ``details``, the test's own statistics or, with ``chunking``, a table with one row per chunk. ``to_dict()``
        writes them under ``data``.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """


class DriftWassersteinResult(_DriftSection, EvaluatorResult[DriftOutput[Any]]):
    """The result of a ``drift-wasserstein`` run: ``output`` is DataEval's ``DriftOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``DriftOutput``: ``drifted``, ``distance``, ``threshold``, ``metric_name``, ``feature_names``, and
        ``details``, the test's own statistics or, with ``chunking``, a table with one row per chunk. ``to_dict()``
        writes them under ``data``.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """


class DriftDomainClassifierResult(_DriftSection, EvaluatorResult[DriftOutput[Any]]):
    """The result of a ``drift-domain-classifier`` run: ``output`` is DataEval's ``DriftOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``DriftOutput``: ``drifted``, ``distance``, ``threshold``, ``metric_name``, ``feature_names``, and
        ``details``, the test's own statistics or, with ``chunking``, a table with one row per chunk. ``to_dict()``
        writes them under ``data``.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """


class OODKNeighborsResult(EvaluatorResult[OODOutput]):
    """The result of an ``ood-kneighbors`` run: ``output`` is DataEval's ``OODOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``OODOutput``: ``is_ood``, whether each item of the data to test is out of distribution;
        ``instance_score``, each item's score; and ``feature_score``, ``None`` for this detector. ``to_dict()``
        writes them under ``data``.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """


class OODDomainClassifierResult(EvaluatorResult[OODOutput]):
    """The result of an ``ood-domain-classifier`` run: ``output`` is DataEval's ``OODOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``OODOutput``: ``is_ood``, whether each item of the data to test is out of distribution;
        ``instance_score``, each item's score; and ``feature_score``, ``None`` for this detector. ``to_dict()``
        writes them under ``data``.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """
