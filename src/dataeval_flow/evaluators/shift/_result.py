"""The shift evaluators' results: DataEval's own output objects, typed per evaluator."""

from typing import Any

from dataeval.shift import DriftOutput, OODOutput

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


class DriftUnivariateResult(EvaluatorResult[DriftOutput[Any]]):
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


class DriftMMDResult(EvaluatorResult[DriftOutput[Any]]):
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


class DriftKNeighborsResult(EvaluatorResult[DriftOutput[Any]]):
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


class DriftWassersteinResult(EvaluatorResult[DriftOutput[Any]]):
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


class DriftDomainClassifierResult(EvaluatorResult[DriftOutput[Any]]):
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
