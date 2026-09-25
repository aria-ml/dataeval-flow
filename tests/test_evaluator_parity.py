"""Every registered evaluator still matches the DataEval class it wraps.

A DataEval rename then fails CI instead of a user's run.
"""

import inspect

import pytest

from dataeval_flow.evaluators import get_evaluator, list_evaluators

# Flow's own fields: the entry's identity and the policies it names, none of them a DataEval argument.
_FLOW_FIELDS = {"name", "type", "stats", "metadata", "ontology"}
_NAMES = [cls.name for cls in list_evaluators()]


@pytest.mark.parametrize("name", _NAMES)
def test_every_dataeval_method_exists(name: str):
    evaluator = get_evaluator(name)
    for method in evaluator.dataeval_methods.values():
        assert callable(getattr(evaluator.dataeval_class, method, None)), f"{name}: {method} is gone"


@pytest.mark.parametrize("name", _NAMES)
def test_dataeval_methods_cover_exactly_the_consumed_kinds(name: str):
    evaluator = get_evaluator(name)
    assert set(evaluator.dataeval_methods) == set(evaluator.config_type.inputs.kinds)


@pytest.mark.parametrize("name", _NAMES)
def test_every_field_is_a_dataeval_argument(name: str):
    evaluator = get_evaluator(name)
    accepted = set(inspect.signature(evaluator.dataeval_class.__init__).parameters)
    for method in evaluator.dataeval_methods.values():
        accepted |= set(inspect.signature(getattr(evaluator.dataeval_class, method)).parameters)
    fields = set(evaluator.config_type.model_fields) - _FLOW_FIELDS
    assert fields <= accepted, f"{name}: {sorted(fields - accepted)} are not DataEval arguments"
