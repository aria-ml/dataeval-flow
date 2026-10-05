"""The naming rules every built-in step follows (naming spec §3.1, §4.2, §5, §6). A failure names the rule; the
rules are written out for plugin authors in reference/naming.md."""

import re
from collections import Counter
from typing import Any, get_args

import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from dataeval_flow.evaluators._registry import EVALUATORS
from dataeval_flow.steps._registry import CHECKS, COMBINES, TRANSFORMS
from dataeval_flow.workflows._registry import WORKFLOWS

_STEPS = [
    cls for registry in (EVALUATORS, TRANSFORMS, COMBINES, CHECKS, WORKFLOWS) for cls in registry.list(plugins=False)
]
# data-analysis is deleted by audit MR 3; until then it keeps its legacy surface (naming spec §9.1).
_LEGACY = {"data-analysis"}
# The two step configs whose `<Type>Config` name a pipeline pool entry already holds (`views:`, `exports:`).
_POOL_NAMED = {"view": "ViewTransformConfig", "export": "ExportTransformConfig"}


def _squashed(text: str) -> str:
    """`text` lowercased, with everything but letters and digits dropped: `Drift (K-Neighbors)` → `driftkneighbors`."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _id(cls: type) -> str:
    return f"{cls.kind}:{cls.name}"


@pytest.mark.parametrize("cls", _STEPS, ids=_id)
def test_a_step_type_and_its_title_name_one_thing(cls: type) -> None:
    assert _squashed(cls.title) == _squashed(cls.name), (
        f"{cls.kind} `{cls.name}` is titled {cls.title!r}: a type and its title agree once case, spaces, hyphens and "
        "parentheses are dropped (reference/naming.md)"
    )


def test_no_two_step_types_share_a_name_across_kinds() -> None:
    assert sorted(name for name, n in Counter(cls.name for cls in _STEPS).items() if n > 1) == []


@pytest.mark.parametrize("cls", _STEPS, ids=_id)
def test_a_step_class_is_named_for_its_type_and_kind(cls: type) -> None:
    assert cls.__name__.lower() == _squashed(cls.name) + cls.kind, f"`{cls.__name__}` is not `<Type><Kind>`"


@pytest.mark.parametrize("cls", [cls for cls in _STEPS if cls.name not in _LEGACY], ids=_id)
def test_a_step_config_is_named_for_its_type(cls: type) -> None:
    expected = _POOL_NAMED.get(cls.name)
    if expected is not None:
        assert cls.config_type.__name__ == expected
    else:
        assert cls.config_type.__name__.lower() == _squashed(cls.name) + "config", (
            f"`{cls.config_type.__name__}` is not `<Type>Config`"
        )


def _own_outputs() -> dict[type, list[str]]:
    """Each output class Flow defines, and the types whose output ports carry it."""
    found: dict[type, list[str]] = {}
    for cls in _STEPS:
        if cls.name in _LEGACY:
            continue
        for port in cls.output_ports():
            for output in port.classes:
                if output.__module__.startswith("dataeval_flow"):
                    found.setdefault(output, []).append(cls.name)
    return found


@pytest.mark.parametrize("output", list(_own_outputs()), ids=lambda output: output.__name__)
def test_an_output_flow_defines_is_named_for_its_type(output: type) -> None:
    types = _own_outputs()[output]
    if len(types) == 1:
        assert output.__name__.lower() == _squashed(types[0]) + "output", f"`{output.__name__}` is not `<Type>Output`"


def test_an_old_check_type_is_refused_at_load() -> None:
    from pydantic import ValidationError

    from tests.chain_toys import chain_pipeline

    steps = [{"name": "o", "evaluator": "o", "input": "data"}, {"name": "c", "check": "outlier-rate", "input": "o"}]
    with pytest.raises(ValidationError, match=r"Unknown check: 'outlier-rate'\. Installed: \[.*'image-outliers'"):
        chain_pipeline(
            workflows=[{"name": "w", "inputs": ["data"], "steps": steps}],
            evaluators=[{"name": "o", "type": "outliers", "flags": ["pixel"]}],
            tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        )


# Settings that only shape what a finding shows, and bound nothing.
_DISPLAY = {"max_examples"}
_STATISTIC = re.compile(r"^(rate|count|ratio|total)$|_rate$")


def _settings(cls: type) -> dict[str, FieldInfo]:
    """A step's settings: its config's fields, less its ports and its entry's `name` and `type`."""
    ports = {port.name for port in cls.input_ports()}
    return {
        name: field
        for name, field in cls.config_type.model_fields.items()
        if name not in ports and name not in ("name", "type")
    }


def _numeric(annotation: Any) -> bool:
    kinds = set(get_args(annotation)) or {annotation}
    return bool(kinds & {int, float}) and bool not in kinds


@pytest.mark.parametrize("cls", CHECKS.list(plugins=False), ids=_id)
def test_a_check_names_its_bounds_by_what_they_bound(cls: type) -> None:
    settings = _settings(cls)
    bounds = sorted(name for name, field in settings.items() if _numeric(field.annotation) and name not in _DISPLAY)
    assert [name for name in settings if _STATISTIC.search(name)] == [], "a bound is named for what it bounds"
    if "info" in settings:
        assert "warning" in settings, "`info` comes with `warning`"
    if len(bounds) == 1:
        assert bounds == ["warning"], f"one bound is called `warning`, not `{bounds[0]}`"


_PRESETS = [cls for cls in WORKFLOWS.list(plugins=False) if cls.name not in _LEGACY]


def _models(annotation: Any) -> list[type[BaseModel]]:
    """The pydantic models a field's annotation can hold: `X`, `X | None`, `X | Literal[False]`."""
    kinds = get_args(annotation) or (annotation,)
    return [kind for kind in kinds if isinstance(kind, type) and issubclass(kind, BaseModel)]


@pytest.mark.parametrize("cls", _PRESETS, ids=_id)
def test_a_presets_checks_are_keyed_by_check_type_in_each_checks_own_words(cls: type) -> None:
    fields = cls.config_type.model_fields
    assert "health_thresholds" not in fields, "a preset's check settings sit under `checks:`"
    if "checks" not in fields:
        return
    (model,) = _models(fields["checks"].annotation)
    for name, field in model.model_fields.items():
        key = field.alias or name
        assert key in CHECKS.names(), f"`checks.{key}` is no check type"
        check = CHECKS.get(key)
        for block in _models(field.annotation):
            stray = set(block.model_fields) - set(_settings(check))
            assert not stray, f"`checks.{key}` holds {sorted(stray)}, which `{key}` does not take"


_STEP_TYPES = {name for registry in (EVALUATORS, TRANSFORMS, COMBINES) for name in registry.names()}


def _blocks(model: type[BaseModel], path: str = "") -> list[tuple[str, str, type[BaseModel]]]:
    """Each settings block keyed by a step type in `model`, as (path, type, block), walking blocks keyed by none."""
    found = []
    for name, field in model.model_fields.items():
        key = field.alias or name
        if key == "checks":
            continue  # its own test
        for block in _models(field.annotation):
            if key in _STEP_TYPES:
                found.append((f"{path}{key}", key, block))
            else:
                found += _blocks(block, f"{path}{key}.")
    return found


def _type_settings(type_id: str) -> set[str]:
    registry = next(registry for registry in (EVALUATORS, TRANSFORMS, COMBINES) if type_id in registry.names())
    return set(_settings(registry.get(type_id)))


@pytest.mark.parametrize("cls", _PRESETS, ids=_id)
def test_a_presets_step_block_holds_only_that_steps_own_settings(cls: type) -> None:
    for path, type_id, block in _blocks(cls.config_type):
        stray = set(block.model_fields) - _type_settings(type_id)
        assert not stray, f"`{path}` holds {sorted(stray)}, which `{type_id}` does not take"


@pytest.mark.parametrize("cls", _PRESETS, ids=_id)
def test_a_preset_spells_no_setting_with_a_step_prefix(cls: type) -> None:
    prefixed = [name for name in cls.config_type.model_fields if re.match(r"(outlier|duplicate)_", name)]
    assert prefixed == [], "a setting sits under its step's type, spelled as the step spells it"


# The fewest settings each preset needs to build its chain, with every optional step on.
_MINIMAL = {
    "data-cleaning": {"outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}},
    "data-coverage": {"coverage": {"method": "naive"}},
    "data-prioritization": {"cleaning": {"outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}}},
    "data-splitting": {"rebalance": "interclass", "coverage": {"method": "naive"}},
    "drift-monitoring": {"detectors": [{"name": "mmd", "type": "drift-mmd"}], "classwise": {"mmd": "class"}},
    "label-space": {"ontology": {"animal": {"cat": None}}},
    "metadata-triage": {},
    "ood-detection": {
        "detectors": [
            {"name": "knn", "type": "ood-kneighbors", "distance_metric": "euclidean"},
            {"name": "dc", "type": "ood-domain-classifier"},
        ]
    },
}
_KINDS = ("evaluator", "combine", "check")


def test_every_preset_has_a_minimal_config_here() -> None:
    assert sorted(_MINIMAL) == sorted(cls.name for cls in _PRESETS)


@pytest.mark.parametrize("cls", _PRESETS, ids=_id)
def test_a_presets_steps_are_named_for_their_types(cls: type) -> None:
    config = cls.config_type.model_validate(_MINIMAL[cls.name])
    chain = cls.chain(config)
    entries = {entry.name: entry.type for entry in chain.evaluators or ()}
    detectors = {detector.name for detector in getattr(config, "detectors", ())}
    for name, type_id in entries.items():
        if name not in detectors:
            assert name == type_id, f"evaluator entry `{name}` is not named for its type `{type_id}`"
    for step in chain.steps:
        kind = next((kind for kind in _KINDS if kind in step), None)
        if kind is None:
            continue  # a transform is named for the Dataset it makes
        name = step["name"]
        if any(name == d or name.startswith(f"{d}-") for d in detectors):
            continue  # a user-named detector's steps keep its name
        type_id = entries.get(step[kind], step[kind]) if kind == "evaluator" else step[kind]
        assert name == type_id or name.startswith(f"{type_id}-"), (
            f"step `{name}` is not `{type_id}` or `{type_id}-<role>`"
        )


def test_finding_is_exported_beside_check_and_only_there() -> None:
    import dataeval_flow.steps as steps
    import dataeval_flow.workflows as workflows

    assert "Finding" in steps.__all__
    assert "Finding" not in workflows.__all__
