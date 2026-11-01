"""The ``sufficiency`` evaluator's result: DataEval's own output object, typed."""

__all__ = ["SufficiencyResult"]

from dataeval.performance import SufficiencyOutput

from dataeval_flow.evaluators._result import EvaluatorResult


class SufficiencyResult(EvaluatorResult[SufficiencyOutput]):
    """The result of a ``sufficiency`` run: ``output`` is DataEval's ``SufficiencyOutput``.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. ``metadata`` also carries the envelope
    fields of :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output
        DataEval's ``SufficiencyOutput``: ``steps``, the training-set sizes measured; ``measures``, each metric per run
        and step; ``averaged_measures``, each metric averaged over the runs; and ``unit_interval``. Its ``project`` and
        ``inv_project`` read the fitted power law. ``to_dict()`` writes them under ``data``, and under ``extras`` the
        trainer's ``path``, ``class`` and ``sha256``, and ``params``, the power law's parameters fitted once per
        metric, so a projection can be repeated from the JSON.
    metadata.evaluator
        The evaluator type, e.g. ``duplicates``.
    metadata.dataeval
        DataEval's own record of the call: its ``name``, ``version``, ``execution_time`` and ``execution_duration``. The
        parameters as written are in ``resolved_config``.
    """
