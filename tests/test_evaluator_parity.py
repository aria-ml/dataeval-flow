"""Every registered evaluator still matches the DataEval class it wraps.

A DataEval rename then fails CI instead of a user's run.
"""

import inspect

import pytest

from dataeval_flow.evaluator import get_evaluator, list_evaluators

_POLICY_REFS = {"stats", "metadata", "ontology"}
_NAMES = [entry["name"] for entry in list_evaluators()]


@pytest.mark.parametrize("name", _NAMES)
def test_every_entry_point_exists(name: str):
    evaluator = get_evaluator(name)
    for method in evaluator.entry_points.values():
        assert callable(getattr(evaluator.dataeval_class, method, None)), f"{name}: {method} is gone"


@pytest.mark.parametrize("name", _NAMES)
def test_entry_points_cover_exactly_the_consumed_kinds(name: str):
    evaluator = get_evaluator(name)
    assert set(evaluator.entry_points) == set(evaluator.params_schema.inputs.kinds)


@pytest.mark.parametrize("name", _NAMES)
def test_every_field_is_a_dataeval_argument(name: str):
    evaluator = get_evaluator(name)
    accepted = set(inspect.signature(evaluator.dataeval_class.__init__).parameters)
    for method in evaluator.entry_points.values():
        accepted |= set(inspect.signature(getattr(evaluator.dataeval_class, method)).parameters)
    fields = set(evaluator.params_schema.model_fields) - _POLICY_REFS
    assert fields <= accepted, f"{name}: {sorted(fields - accepted)} are not DataEval arguments"
