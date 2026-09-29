"""The evaluator registry: the built-in table and the `dataeval_flow.evaluators` entry points."""

from typing import Any

from dataeval_flow._kind import config_type_matches
from dataeval_flow._registry import Registry
from dataeval_flow.evaluators._evaluator import Evaluator

__all__ = ["EVALUATORS", "get_evaluator", "list_evaluators"]

_BUILTINS = {
    "bias.balance": "dataeval_flow.evaluators.bias._evaluator:BalanceEvaluator",
    "bias.diversity": "dataeval_flow.evaluators.bias._evaluator:DiversityEvaluator",
    "bias.parity": "dataeval_flow.evaluators.bias._evaluator:ParityEvaluator",
    "quality.duplicates": "dataeval_flow.evaluators.quality._evaluator:DuplicatesEvaluator",
    "quality.label-health": "dataeval_flow.evaluators.quality._evaluator:LabelHealthEvaluator",
    "quality.outliers": "dataeval_flow.evaluators.quality._evaluator:OutliersEvaluator",
    "scope.representation": "dataeval_flow.evaluators.scope._evaluator:RepresentationEvaluator",
    "scope.coverage": "dataeval_flow.evaluators.scope._evaluator:CoverageEvaluator",
    "scope.prioritize": "dataeval_flow.evaluators.scope._evaluator:PrioritizeEvaluator",
    "scope.label-alignment": "dataeval_flow.evaluators.scope._evaluator:LabelAlignmentEvaluator",
    "shift.drift-domain-classifier": "dataeval_flow.evaluators.shift._evaluator:DriftDomainClassifierEvaluator",
    "shift.drift-kneighbors": "dataeval_flow.evaluators.shift._evaluator:DriftKNeighborsEvaluator",
    "shift.drift-mmd": "dataeval_flow.evaluators.shift._evaluator:DriftMMDEvaluator",
    "shift.drift-univariate": "dataeval_flow.evaluators.shift._evaluator:DriftUnivariateEvaluator",
    "shift.drift-wasserstein": "dataeval_flow.evaluators.shift._evaluator:DriftWassersteinEvaluator",
    "shift.ood-domain-classifier": "dataeval_flow.evaluators.shift._evaluator:OODDomainClassifierEvaluator",
    "shift.ood-kneighbors": "dataeval_flow.evaluators.shift._evaluator:OODKNeighborsEvaluator",
}


EVALUATORS: Registry[Evaluator[Any, Any]] = Registry(
    kind="evaluator",
    group="dataeval_flow.evaluators",
    base=lambda: Evaluator,
    builtins=_BUILTINS,
    check=config_type_matches,
)


def get_evaluator(name: str) -> type[Evaluator[Any, Any]]:
    """The evaluator registered as `name`, built-in or plugin.

    Parameters
    ----------
    name : str
        The evaluator's name: its type id, e.g. ``"quality.duplicates"``.

    Returns
    -------
    type[Evaluator]
        The evaluator class, whose ``name``, ``description`` and ``config_type`` describe it. Flow builds its
        instances.

    Raises
    ------
    ValueError
        When nothing is registered under `name`, or the plugin registered under it failed to load.
    """
    return EVALUATORS.get(name)


def list_evaluators() -> list[type[Evaluator[Any, Any]]]:
    """Every installed evaluator, built-in or plugin, sorted by name.

    A plugin that failed to load is left out; :func:`get_evaluator` raises its error.

    Returns
    -------
    list[type[Evaluator]]
        The evaluator classes.
    """
    return EVALUATORS.list()
