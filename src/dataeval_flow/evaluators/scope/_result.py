"""The scope evaluators' results: DataEval's own output objects, typed per evaluator."""

from dataeval.scope import RepresentationOutput

from dataeval_flow.evaluators._result import EvaluatorResult

__all__ = ["RepresentationResult"]


class RepresentationResult(EvaluatorResult[RepresentationOutput]):
    """The result of a ``scope.representation`` run: ``output`` is DataEval's ``RepresentationOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``RepresentationOutput``: ``data()`` is the worklist, one row per concept short of its target
        (``concept``, ``label``, ``parent``, ``action``, ``count``, ``target``, ``deficit``). ``leaf_coverage``,
        ``total_deficit``, ``violations`` and ``dark_branches`` are the summary ``to_dict()`` writes under ``extras``.
    metadata.evaluator
        The evaluator type, e.g. ``quality.duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """
