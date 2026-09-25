"""The quality evaluators' results: DataEval's own output objects, typed per evaluator."""

from typing import Any

from dataeval.quality import DuplicatesOutput, OutliersOutput

from dataeval_flow.evaluators._result import EvaluatorResult

__all__ = ["DuplicatesResult", "OutliersResult"]


class DuplicatesResult(EvaluatorResult[DuplicatesOutput[Any, Any]]):
    """The result of a ``quality.duplicates`` run: ``output`` is DataEval's ``DuplicatesOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``DuplicatesOutput``, so its own methods work: ``data()`` for the table of groups,
        ``aggregate_by_image()``, ``aggregate_by_group()`` and the rest. Its JSON form is what ``to_dict()`` writes.
    metadata.evaluator
        The evaluator type, e.g. ``quality.duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """


class OutliersResult(EvaluatorResult[OutliersOutput[Any]]):
    """The result of a ``quality.outliers`` run: ``output`` is DataEval's ``OutliersOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``OutliersOutput``, so its own methods work: ``data()`` for the table of flagged items,
        ``aggregate_by_item()``, ``aggregate_by_metric()`` and the rest. Its JSON form is what ``to_dict()`` writes.
    metadata.evaluator
        The evaluator type, e.g. ``quality.duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """
