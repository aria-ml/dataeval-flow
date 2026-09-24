"""Evaluator framework: run one DataEval evaluator and report its determinations.

An evaluator applies DataEval's threshold and says what it found. It makes no verdict:
health, readiness and ``--fail-on-warning`` belong to workflows.

Only :mod:`~dataeval_flow.evaluator.base` is imported at module level, because
``config.schemas`` imports every evaluator's parameters, and those import this package.
"""

__all__ = [
    "EvaluatorParametersBase",
    "InputKind",
    "InputSpec",
    "SourceCount",
    "get_evaluator",
    "list_evaluators",
    "task_problem",
]

from typing import TYPE_CHECKING

from dataeval_flow.evaluator.base import EvaluatorParametersBase, InputKind, InputSpec, SourceCount, task_problem

if TYPE_CHECKING:
    from dataeval_flow.evaluator.protocol import EvaluatorBase

_EVALUATORS: "dict[str, EvaluatorBase]" = {}


def _registered() -> "list[type[EvaluatorBase]]":
    """Every evaluator this build provides, imported here because each module pulls in DataEval."""
    from dataeval_flow.evaluators.quality.evaluator import DuplicatesEvaluator, OutliersEvaluator

    return [DuplicatesEvaluator, OutliersEvaluator]


def _ensure_initialized() -> None:
    if not _EVALUATORS:
        # Built locally and assigned once: a dict half-filled by one racing thread must
        # never look "initialized" (truthy) to another thread's `if not _EVALUATORS:` check.
        built = {evaluator.name: evaluator for evaluator in (cls() for cls in _registered())}
        _EVALUATORS.update(built)


def get_evaluator(name: str) -> "EvaluatorBase":
    """Look up an evaluator by type.

    Raises
    ------
    ValueError
        When no evaluator of that type exists; the message lists the ones that do.
    """
    _ensure_initialized()
    if name not in _EVALUATORS:
        raise ValueError(f"Unknown evaluator: '{name}'. Available: {sorted(_EVALUATORS)}")
    return _EVALUATORS[name]


def list_evaluators() -> list[dict[str, str]]:
    """Return every evaluator type with its description, what it consumes, and its source count."""
    _ensure_initialized()
    entries: list[dict[str, str]] = []
    for evaluator in _EVALUATORS.values():
        spec = evaluator.params_schema.inputs
        consumes = [*sorted(spec.required), *(f"{kind} (optional)" for kind in sorted(spec.optional))]
        entries.append(
            {
                "name": evaluator.name,
                "description": evaluator.description,
                "consumes": ", ".join(consumes),
                "sources": spec.sources.value,
            }
        )
    return entries
