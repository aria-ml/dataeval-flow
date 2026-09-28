"""The shift evaluators' results: DataEval's own output objects, typed per evaluator."""

from typing import Any

from dataeval.shift import DriftOutput

from dataeval_flow.evaluators._result import EvaluatorResult

__all__ = [
    "DriftDomainClassifierResult",
    "DriftKNeighborsResult",
    "DriftMMDResult",
    "DriftUnivariateResult",
    "DriftWassersteinResult",
]


class DriftUnivariateResult(EvaluatorResult[DriftOutput[Any]]):
    """The result of a ``shift.drift-univariate`` run: ``output`` is DataEval's ``DriftOutput``.

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
        The evaluator type, e.g. ``quality.duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """


class DriftMMDResult(EvaluatorResult[DriftOutput[Any]]):
    """The result of a ``shift.drift-mmd`` run: ``output`` is DataEval's ``DriftOutput``.

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
        The evaluator type, e.g. ``quality.duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """


class DriftKNeighborsResult(EvaluatorResult[DriftOutput[Any]]):
    """The result of a ``shift.drift-kneighbors`` run: ``output`` is DataEval's ``DriftOutput``.

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
        The evaluator type, e.g. ``quality.duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """


class DriftWassersteinResult(EvaluatorResult[DriftOutput[Any]]):
    """The result of a ``shift.drift-wasserstein`` run: ``output`` is DataEval's ``DriftOutput``.

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
        The evaluator type, e.g. ``quality.duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """


class DriftDomainClassifierResult(EvaluatorResult[DriftOutput[Any]]):
    """The result of a ``shift.drift-domain-classifier`` run: ``output`` is DataEval's ``DriftOutput``.

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
        The evaluator type, e.g. ``quality.duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """
