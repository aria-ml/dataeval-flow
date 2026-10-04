"""The catalog as the spec states it: each type's sources and extractor rule, and the tasks that break them refused.

A task that breaks the rule costs a config error when the pipeline loads, not a walk over the data.
"""

import re
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import PipelineConfig
from dataeval_flow.evaluators import get_evaluator
from dataeval_flow.evaluators._registry import _BUILTINS
from tests.evaluator_toys import _EXTRA_CONFIG, toy_run, toy_task_run

# Each type's `SourceCount` value, and whether its tasks need, may name, or must not name an extractor (spec §3).
_CATALOG: dict[str, tuple[str, str]] = {
    "duplicates": ("1+", "optional"),
    "label-health": ("1", "refused"),
    "factor-leakage": ("2", "refused"),
    "factor-triage": ("1", "refused"),
    "content-digest": ("1", "refused"),
    "metadata-summary": ("1", "refused"),
    "outliers": ("1+", "optional"),
    "balance": ("1", "refused"),
    "diversity": ("1", "refused"),
    "parity": ("1", "refused"),
    "representation": ("1", "refused"),
    "coverage": ("1", "required"),
    "prioritize": ("1-2", "required"),
    "label-alignment": ("1", "refused"),
    "completeness": ("1", "required"),
    "label-reconciliation": ("1", "refused"),
    "ontology-validation": ("1", "refused"),
    "drift-domain-classifier": ("2", "required"),
    "drift-kneighbors": ("2", "required"),
    "drift-mmd": ("2", "required"),
    "drift-univariate": ("2", "required"),
    "drift-wasserstein": ("3", "required"),
    "ood-domain-classifier": ("2", "required"),
    "ood-kneighbors": ("2", "required"),
    "divergence": ("2", "required"),
}

# A count each rule refuses, where one exists; and a count it allows.
_WRONG_COUNT = {"1": 2, "1-2": 3, "2": 1, "3": 2}
_RIGHT_COUNT = {"1": 1, "1+": 1, "1-2": 1, "2": 2, "3": 3}


def _extractor_rule(name: str) -> str:
    config = get_evaluator(name).config_type(**_EXTRA_CONFIG.get(name, {}))  # type: ignore[call-arg]
    if config.requires_extractor():
        return "required"
    return "optional" if config.inputs.accepts_extractor else "refused"


def _pipeline(name: str, *, sources: int, extractor: bool) -> dict[str, Any]:
    task: dict[str, Any] = {"name": "t", "evaluator": "e", "sources": ["a", "b", "c", "d"][:sources]}
    if extractor:
        task["extractor"] = "flat"
    return {
        "datasets": [{"name": "ds", "format": "huggingface", "path": "./d", "task": "image_classification"}],
        "sources": [{"name": source, "dataset": "ds"} for source in ("a", "b", "c", "d")],
        "extractors": [{"name": "flat", "model": "flatten"}],
        "evaluators": [{"name": "e", "type": name, **_EXTRA_CONFIG.get(name, {})}],
        "tasks": [task],
    }


def _types(rule: str) -> list[str]:
    return sorted(name for name, (_, extractor) in _CATALOG.items() if extractor == rule)


def test_every_builtin_is_in_the_catalog():
    assert set(_BUILTINS) == set(_CATALOG)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("quality.outliers", "outliers"),
        ("bias.balance", "balance"),
        ("scope.coverage", "coverage"),
        ("shift.drift-mmd", "drift-mmd"),
    ],
)
def test_a_family_prefixed_id_is_unknown_and_the_refusal_names_the_plain_one(old: str, new: str):
    with pytest.raises(ValidationError, match=re.escape(f"Unknown evaluator: '{old}'. Installed: ")) as refused:
        PipelineConfig.model_validate({"evaluators": [{"name": "e", "type": old}]})
    installed = str(refused.value).split("Installed: ", 1)[1].split("]", 1)[0]
    assert f"'{new}'" in installed
    assert f"'{old}'" not in installed


@pytest.mark.parametrize("name", sorted(_CATALOG))
def test_each_type_declares_what_the_catalog_says(name: str):
    sources, extractor = _CATALOG[name]
    assert get_evaluator(name).config_type.inputs.sources.value == sources
    assert _extractor_rule(name) == extractor


@pytest.mark.parametrize("name", sorted(_CATALOG))
def test_a_task_the_catalog_allows_loads(name: str):
    sources, extractor = _CATALOG[name]
    PipelineConfig.model_validate(_pipeline(name, sources=_RIGHT_COUNT[sources], extractor=extractor == "required"))


@pytest.mark.parametrize("name", sorted(name for name, (sources, _) in _CATALOG.items() if sources in _WRONG_COUNT))
def test_the_wrong_source_count_is_refused_at_load(name: str):
    sources, extractor = _CATALOG[name]
    with pytest.raises(ValidationError, match="takes"):
        PipelineConfig.model_validate(_pipeline(name, sources=_WRONG_COUNT[sources], extractor=extractor == "required"))


@pytest.mark.parametrize("name", _types("required"))
def test_a_missing_extractor_is_refused_at_load(name: str):
    with pytest.raises(ValidationError, match="needs an extractor"):
        PipelineConfig.model_validate(_pipeline(name, sources=_RIGHT_COUNT[_CATALOG[name][0]], extractor=False))


@pytest.mark.parametrize("name", _types("refused"))
def test_a_named_extractor_is_refused_at_load(name: str):
    with pytest.raises(ValidationError, match="does not use an extractor"):
        PipelineConfig.model_validate(_pipeline(name, sources=_RIGHT_COUNT[_CATALOG[name][0]], extractor=True))


@pytest.mark.parametrize("name", sorted(_CATALOG))
def test_each_type_runs_as_a_task_of_a_pipeline(name: str):
    result = toy_task_run(name)
    assert result.success, result.errors
    assert isinstance(result, get_evaluator(name).config_type.result_type)


@pytest.mark.parametrize("name", sorted(_CATALOG))
def test_an_empty_source_is_a_result_not_an_exception(name: str):
    """Nothing to evaluate is a result, successful or failed with DataEval's message, and it reports."""
    assert toy_run(name, count=0).report()
