"""Every preset gives its steps their own defaults (preset naming spec R8, R9): a settings block's defaults are its
step's, and a chain passes a step no value other than its default, but for the values a preset fixes by design."""

import importlib
import inspect
import json
import pkgutil
import re
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

import dataeval_flow.workflows
from dataeval_flow.evaluators._registry import EVALUATORS
from dataeval_flow.steps._registry import CHECKS, COMBINES, TRANSFORMS, inline_registry
from dataeval_flow.workflows._registry import WORKFLOWS
from tests.test_naming_conventions import _MINIMAL

# (step type, setting) a preset fixes by design, each with its reason.
_FIXED = {
    ("wrap", "wrapper"),  # coverage crops detections into classification items
    ("wrap", "other_kinds"),  # and passes other kinds of Dataset through unchanged
    ("wrap", "params"),  # with the crop parameters it offers
    ("outliers", "per_target"),  # cleaning judges targets with target-outliers; audit has no per-target check
    ("drift", "subject"),  # each detector's check names it
    ("ood", "subject"),
    ("select", "fraction"),  # prioritization keeps every item unless told otherwise
    ("view", "operations"),  # splitting's `rebalance`
    ("embedding-divergence", "info"),  # audit derives it as the check does
}
# Defaults Task 4 consolidates; empty once it lands.
_PENDING: set[tuple[str, str]] = set()
_CHAIN_WIDE = {"ontology", "metadata", "stats"}
_IDENTITY = {"name", "type", "input", "by", "optional", "pairs", "extractor"}
_STEP_KINDS = ("check", "combine", "transform")


def _default(field: FieldInfo) -> Any:
    if field.default is not PydanticUndefined:
        return field.default
    if field.default_factory is not None:
        return field.default_factory()  # type: ignore[call-arg]
    return PydanticUndefined


def _json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return json.loads(json.dumps(value, default=str))


def _keys(value: Any) -> set[str]:
    """Every key written anywhere in `value`, a minimal config."""
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


def _differences(preset: type) -> list[str]:
    config = preset.config_type.model_validate(_MINIMAL[preset.name])
    chain = preset.chain(config)
    skip = _keys(_MINIMAL[preset.name]) | _CHAIN_WIDE | _IDENTITY
    found: set[str] = set()
    for entry in chain.evaluators or ():
        for name, field in type(entry).model_fields.items():
            if name in skip or (entry.type, name) in _FIXED | _PENDING:
                continue
            if _json(getattr(entry, name)) != _json(_default(field)):
                found.add(f"{entry.type}.{name} = {getattr(entry, name)!r}")
    for step in chain.steps:
        kind = next((kind for kind in _STEP_KINDS if kind in step), None)
        if kind is None:
            continue
        impl = inline_registry(kind).get(step[kind])
        ports = {port.name for port in impl.input_ports()}
        for name, value in step.items():
            if name == kind or name in skip | ports or (step[kind], name) in _FIXED | _PENDING:
                continue
            if _json(value) != _json(_default(impl.config_type.model_fields[name])):
                found.add(f"{step[kind]}.{name} = {value!r}")
    return sorted(found)


@pytest.mark.parametrize("preset", WORKFLOWS.list(plugins=False), ids=lambda cls: cls.name)
def test_a_preset_passes_each_step_its_own_defaults(preset: type) -> None:
    assert _differences(preset) == []


def _squashed(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


_STEPS = {
    _squashed(cls.name): cls
    for registry in (EVALUATORS, TRANSFORMS, COMBINES, CHECKS)
    for cls in registry.list(plugins=False)
}


def _settings_models() -> list[tuple[type, type]]:
    """Each preset `<Type>Settings` model, with the step it configures."""
    pairs: list[tuple[type, type]] = []
    for info in pkgutil.iter_modules(dataeval_flow.workflows.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"dataeval_flow.workflows.{info.name}._config")
        for name, model in inspect.getmembers(module, inspect.isclass):
            step = _STEPS.get(_squashed(name.removesuffix("Settings")))
            if name.endswith("Settings") and model.__module__ == module.__name__ and step is not None:
                pairs.append((model, step))
    return pairs


@pytest.mark.parametrize(("model", "step"), _settings_models(), ids=lambda value: value.__name__)
def test_a_settings_block_takes_its_step_s_defaults(model: type[BaseModel], step: type) -> None:
    own = step.config_type.model_fields
    for name, field in model.model_fields.items():
        default = _default(field)
        if name not in own or default is PydanticUndefined or (step.name, name) in _FIXED | _PENDING:
            continue  # a preset may require what its step leaves optional
        assert _json(default) == _json(_default(own[name])), f"{model.__name__}.{name}"


def test_a_partly_written_coverage_block_leaves_the_rest_to_dataeval() -> None:
    from dataeval_flow.workflows.scope import CoverageSettings

    written = CoverageSettings.model_validate({"method": "naive"})
    assert written.num_observations is None
    assert written.percent is None


def test_a_split_with_no_fractions_holds_out_test_and_val_and_zero_is_still_refused() -> None:
    from pydantic import ValidationError

    from dataeval_flow.steps.transforms._split import KFoldConfig, SplitConfig

    split = SplitConfig(input="data")
    assert (split.test_frac, split.val_frac, split.stratify) == (0.2, 0.1, True)
    assert KFoldConfig(input="data", folds=3).test_frac == 0.2
    with pytest.raises(ValidationError, match="holds nothing out"):
        SplitConfig(input="data", test_frac=0.0, val_frac=0.0)
