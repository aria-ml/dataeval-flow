"""Every built-in evaluator config mirrors its DataEval class's `Config`, in both directions.

A DataEval release that adds, renames or retypes a parameter fails this test, naming it, instead of a user's run.
"""

import dataclasses
import inspect
import types
import typing
from collections.abc import Callable
from typing import Any, Literal

import pytest
from pydantic import ConfigDict, TypeAdapter

from dataeval_flow.evaluators import get_evaluator, list_evaluators
from dataeval_flow.evaluators._fields import FLOW_FIELDS
from dataeval_flow.evaluators._registry import _BUILTINS

_NAMES = [cls.name for cls in list_evaluators()]

# The DataEval `Config` fields a config file cannot express: objects and runtime choices, which the task supplies
# (the extractor, its batch size) or Flow leaves to DataEval. Kept here rather than on a public class.
_WIRING: dict[str, frozenset[str]] = {
    "duplicates": frozenset({"extractor", "batch_size"}),
    "label-health": frozenset(),
    "outliers": frozenset({"extractor", "batch_size"}),
    "balance": frozenset(),
    "diversity": frozenset(),
    "parity": frozenset(),
    "representation": frozenset(),
    "coverage": frozenset({"extractor", "batch_size"}),
    "prioritize": frozenset({"extractor", "batch_size"}),
    "drift-domain-classifier": frozenset({"extractor", "update_strategy"}),
    "drift-kneighbors": frozenset({"extractor", "update_strategy"}),
    "drift-mmd": frozenset({"extractor", "update_strategy", "sigma", "device"}),
    "drift-univariate": frozenset({"extractor", "update_strategy"}),
    "drift-wasserstein": frozenset({"extractor", "update_strategy"}),
    "ood-domain-classifier": frozenset({"extractor"}),
    "ood-kneighbors": frozenset({"extractor"}),
    # `matchers` holds objects a config file cannot express.
    "label-alignment": frozenset({"matchers"}),
}

# Flow's own evaluators: their settings are Flow's, not a DataEval `Config`'s, so they have no wiring list.
_FLOW_ONLY: frozenset[str] = frozenset({"triage"})

# Fields Flow converts before DataEval sees them, whose types are Flow's by design.
_CONVERTED: set[tuple[str, str]] = {("duplicates", "flags"), ("outliers", "flags")}


def _dataeval_fields(cls: "type | Callable[..., Any]") -> dict[str, Any]:
    """The fields of `cls.Config`, a pydantic model or a dataclass, with their annotations.

    A function-backed evaluator (``cls`` is a ``dataeval.core`` function) has no ``Config``; its keyword-only
    parameters play that role, since those are the ones a config file can set.
    """
    if inspect.isfunction(cls):
        return {
            name: parameter.annotation
            for name, parameter in inspect.signature(cls).parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        }
    config = cls.Config  # type: ignore[attr-defined]
    if dataclasses.is_dataclass(config):
        return {field.name: field.type for field in dataclasses.fields(config)}
    return {name: info.annotation for name, info in config.model_fields.items()}


def _method_parameters(evaluator: Any) -> set[str]:
    """The parameters of the DataEval methods the evaluator calls, such as `from_stats`'s `per_image`.

    A function-backed evaluator's ``__call__`` is the slot every function has, whose own signature is
    ``(*args, **kwargs)``; inspect the function itself instead.
    """
    if inspect.isfunction(evaluator.dataeval_class):
        return set(inspect.signature(evaluator.dataeval_class).parameters) - {"self"}
    return {
        name
        for method in evaluator.dataeval_methods.values()
        for name in inspect.signature(getattr(evaluator.dataeval_class, method)).parameters
    } - {"self"}


def _samples(annotation: Any) -> list[Any]:
    """Every `Literal` choice `annotation` admits, one value per `bool`, `int`, `float`, `str`; none for others."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _samples(typing.get_args(annotation)[0])
    if origin is Literal:
        return list(typing.get_args(annotation))
    if origin in (typing.Union, types.UnionType):
        return [sample for arg in typing.get_args(annotation) if arg is not type(None) for sample in _samples(arg)]
    return {bool: [True], int: [2], float: [0.5], str: ["x"]}.get(annotation, [])


def _refuses(annotation: Any, sample: Any) -> bool:
    try:
        TypeAdapter(annotation, config=ConfigDict(arbitrary_types_allowed=True)).validate_python(sample)
    except (TypeError, ValueError):
        return True
    return False


def test_every_builtin_has_a_wiring_list():
    missing = sorted(set(_BUILTINS) - set(_WIRING) - _FLOW_ONLY)
    assert not missing, f"add {missing} to _WIRING"


@pytest.mark.parametrize("name", _NAMES)
def test_every_dataeval_method_exists(name: str):
    evaluator = get_evaluator(name)
    for method in evaluator.dataeval_methods.values():
        assert callable(getattr(evaluator.dataeval_class, method, None)), f"{name}: {method} is gone"


@pytest.mark.parametrize("name", _NAMES)
def test_dataeval_methods_cover_exactly_the_consumed_kinds(name: str):
    evaluator = get_evaluator(name)
    assert set(evaluator.dataeval_methods) == set(evaluator.config_type.inputs.kinds)


@pytest.mark.parametrize("name", sorted(_WIRING))
def test_every_field_is_a_dataeval_parameter(name: str):
    evaluator = get_evaluator(name)
    accepted = set(_dataeval_fields(evaluator.dataeval_class)) | _method_parameters(evaluator)
    fields = set(evaluator.config_type.model_fields) - FLOW_FIELDS
    assert fields <= accepted, f"{name}: {sorted(fields - accepted)} are not DataEval parameters"


@pytest.mark.parametrize("name", sorted(_WIRING))
def test_every_dataeval_parameter_is_exposed_or_wiring(name: str):
    evaluator = get_evaluator(name)
    exposed = set(evaluator.config_type.model_fields)
    unexposed = set(_dataeval_fields(evaluator.dataeval_class)) - exposed - _WIRING[name]
    assert not unexposed, f"{name}: DataEval's {sorted(unexposed)} are neither fields nor on the wiring list"


@pytest.mark.parametrize("name", sorted(_WIRING))
def test_every_field_takes_values_dataeval_accepts(name: str):
    evaluator = get_evaluator(name)
    dataeval = _dataeval_fields(evaluator.dataeval_class)
    for field, info in evaluator.config_type.model_fields.items():
        if field not in dataeval or (name, field) in _CONVERTED:
            continue
        refused = [sample for sample in _samples(info.annotation) if _refuses(dataeval[field], sample)]
        assert not refused, f"{name}.{field}: DataEval's {dataeval[field]} refuses {refused}"


@pytest.mark.parametrize(
    "detector", ["DriftUnivariate", "DriftMMD", "DriftKNeighbors", "DriftWasserstein", "DriftDomainClassifier"]
)
def test_chunking_mirrors_chunked(detector: str):
    """`chunking:` takes what `chunked()` does, less `chunker`, an object a config file cannot build."""
    import dataeval.shift

    from dataeval_flow.evaluators.shift import ChunkedDriftConfig

    parameters = set(inspect.signature(getattr(dataeval.shift, detector).chunked).parameters) - {"self", "chunker"}
    assert set(ChunkedDriftConfig.model_fields) == parameters, f"chunked() takes {sorted(parameters)}"
