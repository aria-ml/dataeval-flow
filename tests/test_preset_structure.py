"""Which of a preset's settings decide its steps, declared on the preset (preset decomposition spec §4)."""

from collections.abc import Mapping
from typing import Any

import pytest

from dataeval_flow.workflows._preset import Preset, structural_choices
from dataeval_flow.workflows._registry import get_workflow
from tests.test_audit_preset import _OUTLIERS
from tests.test_naming_conventions import _MINIMAL

_BUILTINS = ("audit", "bias", "prioritization", "quality", "scope", "shift", "splits", "taxonomy", "triage")
_SHIFT = _MINIMAL["shift"]

# (type, settings without the choice, settings with it, the declared setting): flipping the setting changes the
# chain's steps by exactly the steps its declaration names.
_FLIPS: list[tuple[str, dict[str, Any], dict[str, Any], str]] = [
    ("audit", _OUTLIERS, {**_OUTLIERS, "ontology": {"animal": {"cat": None}}}, "ontology"),
    ("audit", _OUTLIERS, {**_OUTLIERS, "factor-leakage": {"factors": ["site"]}}, "factor-leakage"),
    ("audit", _OUTLIERS, {**_OUTLIERS, "factor-gaps": False}, "factor-gaps"),
    ("audit", _OUTLIERS, {**_OUTLIERS, "coverage": {"method": "naive"}}, "coverage.method"),
    ("bias", {}, {"factor-gaps": False}, "factor-gaps"),
    ("scope", {}, {"coverage": {"method": "naive"}}, "coverage.method"),
    ("scope", {}, {"completeness": False}, "completeness"),
    ("splits", {}, {"val_frac": 0}, "val_frac"),
    ("splits", {}, {"test_frac": 0}, "test_frac"),
    ("splits", {}, {"rebalance": "interclass"}, "rebalance"),
    ("shift", _SHIFT, {**_SHIFT, "factor-predictors": False}, "factor-predictors"),
    ("shift", _SHIFT, {**_SHIFT, "factor-deviation": False}, "factor-deviation"),
]


def _config(type_id: str, settings: Mapping[str, Any]) -> Any:
    return get_workflow(type_id).config_type.model_validate({"name": "w", **settings})


def _steps(type_id: str, settings: Mapping[str, Any]) -> set[str]:
    preset = get_workflow(type_id)
    assert issubclass(preset, Preset)
    chain = preset.chain(_config(type_id, settings))
    return {step["name"] if isinstance(step, Mapping) else step.name for step in chain.steps}


@pytest.mark.parametrize("type_id", _BUILTINS)
def test_every_built_in_declares_its_structure(type_id: str) -> None:
    preset = get_workflow(type_id)
    assert issubclass(preset, Preset)
    assert preset.structure is not None
    assert all(declared.rule for declared in preset.structure)


def test_a_preset_that_declares_nothing_has_no_choices_to_report() -> None:
    from tests.example_plugin import CountConfig, CountWorkflow

    assert CountWorkflow.structure is None
    assert structural_choices(CountConfig(name="w"), CountWorkflow) is None


@pytest.mark.parametrize(("type_id", "without", "with_", "setting"), _FLIPS, ids=[f"{t}:{s}" for t, _, _, s in _FLIPS])
def test_flipping_a_structural_setting_changes_exactly_its_steps(
    type_id: str, without: dict[str, Any], with_: dict[str, Any], setting: str
) -> None:
    preset = get_workflow(type_id)
    assert issubclass(preset, Preset)
    (declared,) = [entry for entry in preset.structure or () if entry.setting == setting]
    assert declared.steps, f"{setting} is enumerable, so it names its steps"
    assert _steps(type_id, without) ^ _steps(type_id, with_) == set(declared.steps)


def test_an_omitted_setting_reports_the_value_it_stands_for() -> None:
    preset = get_workflow("audit")
    assert issubclass(preset, Preset)
    choices = {choice.setting: choice for choice in structural_choices(_config("audit", _OUTLIERS), preset) or ()}
    method = choices["coverage.method"]
    assert (method.value, method.written, method.effective) == (None, False, "adaptive")
    assert (method.included, method.excluded) == ((), ("uncovered-items",))
    ontology = choices["ontology"]
    assert (ontology.value, ontology.written, ontology.included) == (None, False, ())
    assert set(ontology.excluded) == {
        "label-reconciliation-train",
        "label-reconciliation-evals",
        "label-conformance-train",
        "label-conformance-evals",
    }


def test_a_written_setting_reports_it_was_written_and_what_it_includes() -> None:
    preset = get_workflow("audit")
    assert issubclass(preset, Preset)
    config = _config("audit", {**_OUTLIERS, "coverage": {"method": "naive"}, "factor-gaps": False})
    choices = {choice.setting: choice for choice in structural_choices(config, preset) or ()}
    assert (choices["coverage.method"].value, choices["coverage.method"].written) == ("naive", True)
    assert choices["coverage.method"].included == ("uncovered-items",)
    assert (choices["factor-gaps"].value, choices["factor-gaps"].written) == (False, True)
    assert choices["factor-gaps"].excluded == ("factor-gaps", "factor-coverage-gaps")


def test_collection_driven_structure_is_a_rule_naming_no_steps() -> None:
    preset = get_workflow("shift")
    assert issubclass(preset, Preset)
    rules = {declared.setting: declared for declared in preset.structure or ()}
    assert rules["detectors"].steps == ()
    assert "ood-agreement" in rules["detectors"].rule
    assert rules["classwise"].steps == ()
    assert "by-class" in rules["classwise"].rule
