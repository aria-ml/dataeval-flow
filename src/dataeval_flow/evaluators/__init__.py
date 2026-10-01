"""Evaluators: the framework for writing one, and the built-in evaluators.

An evaluator runs one DataEval evaluator and reports its determinations under DataEval's thresholds. It makes
no verdict: health, readiness and ``--fail-on-warning`` belong to workflows. Built-ins live in subpackages named
for DataEval's modules: ``bias``, ``quality``, ``scope`` and ``shift``.
"""

from dataeval_flow.evaluators._base import EvaluatorConfig
from dataeval_flow.evaluators._evaluator import Evaluator
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators._per_class import PerClassOutput
from dataeval_flow.evaluators._registry import get_evaluator, list_evaluators
from dataeval_flow.evaluators._result import EvaluatorResult

__all__ = [
    "Evaluator",
    "EvaluatorConfig",
    "EvaluatorInputs",
    "EvaluatorResult",
    "PerClassOutput",
    "get_evaluator",
    "list_evaluators",
]
