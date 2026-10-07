"""The prioritization preset: each pool ranked against the reference, and the top of each ranking kept as
`selected` (spec §10.9). A custom workflow cleans the data first, with steps or a quality step."""

import re
from collections.abc import Mapping
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dataeval_flow import run, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.prioritization import PrioritizationConfig
from tests.chain_toys import FLAT, chain_pipeline
from tests.evaluator_toys import ToyImages

_CLEANING = {"outliers": {"flags": ["pixel", "visual"], "outlier_threshold": "zscore"}}
_KNN = {"method": "knn", "k": 3}


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _task(sources: dict[str, Any], **settings: Any) -> ChainResult:
    config = chain_pipeline(
        workflows=[{"name": "prio", "type": "prioritization", "prioritization": _KNN, **settings}],
        tasks=[{"name": "t", "workflow": "prio", "sources": list(sources), "extractor": "flat"}],
        datasets=sources,
        extractor=True,
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


def _pair() -> dict[str, Any]:
    return {"ref": ToyImages(count=16), "pool": ToyImages(count=20, seed=1)}


def _selected(result: ChainResult) -> dict[str, int]:
    return {key: len(element.output) for key, element in (result.steps["selected"].elements or {}).items()}


def test_the_preset_ranks_and_selects() -> None:
    result = _task(_pair())
    assert result.type == "prioritization"
    assert list(result.steps) == ["prioritization", "selected"]
    assert result.findings == []
    assert _selected(result) == {"pool": 20}


def test_n_keeps_the_top_of_each_ranking() -> None:
    assert _selected(_task(_pair(), select={"n": 5})) == {"pool": 5}


def test_n_larger_than_a_pool_keeps_the_whole_pool() -> None:
    assert _selected(_task(_pair(), select={"n": 100})) == {"pool": 20}


def test_fraction_keeps_its_share_rounded_up() -> None:
    assert _selected(_task(_pair(), select={"fraction": 0.21})) == {"pool": 5}


def test_n_and_fraction_together_are_refused() -> None:
    message = "`select` takes `n:` or `fraction:`, not both."
    with pytest.raises(ValidationError, match=re.escape(message)):
        PrioritizationConfig(select={"n": 5, "fraction": 0.5})  # type: ignore[arg-type]


def test_each_pool_is_ranked_on_its_own_against_the_one_reference() -> None:
    sources = {"ref": ToyImages(count=16), "p1": ToyImages(count=20, seed=1), "p2": ToyImages(count=12, seed=2)}
    result = _task(sources)
    assert result.steps["prioritization"].inputs == ["pools", "reference"]
    assert list(result.steps["prioritization"].elements or {}) == ["p1", "p2"]
    assert list(result.steps["selected"].elements or {}) == ["p1", "p2"]


def test_a_task_naming_one_source_is_refused() -> None:
    message = "Task 't' runs workflow 'prio' (prioritization), which takes two or more sources, but the task names 1."
    with pytest.raises(ValidationError, match=re.escape(message)):
        chain_pipeline(
            workflows=[{"name": "prio", "type": "prioritization"}],
            tasks=[{"name": "t", "workflow": "prio", "sources": ["ref"], "extractor": "flat"}],
            datasets={"ref": ToyImages(count=4)},
            extractor=True,
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("health_thresholds", {}),
        ("value_range", [0, 1]),
        ("cleaning", _CLEANING),
        ("stats", "bands"),
        ("method", "knn"),
        ("order", "easy_first"),
    ],
)
def test_a_config_that_still_writes_a_removed_key_is_refused(key: str, value: object) -> None:
    with pytest.raises(ValidationError) as info:
        PrioritizationConfig.model_validate({"name": "prio", key: value})
    assert [error["loc"] for error in info.value.errors() if error["type"] == "extra_forbidden"] == [(key,)]


def test_run_takes_a_data_prioritization_config() -> None:
    result = run(PrioritizationConfig(prioritization=_KNN), _pair(), extractor=FLAT)  # type: ignore[arg-type]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert _selected(result) == {"pool": 20}
    assert "Highest priority" in result.report()


def test_data_prioritization_runs_as_a_step_after_data_cleaning_over_the_pools() -> None:
    config = chain_pipeline(
        workflows=[
            {"name": "basic_clean", "type": "quality", **_CLEANING},
            {"name": "prio", "type": "prioritization", "prioritization": _KNN},
            {
                "name": "outer",
                "inputs": ["ref", {"name": "pools", "list": True}],
                "steps": [
                    {"name": "cleaning", "workflow": "basic_clean", "input": "pools"},
                    {"name": "ranking", "workflow": "prio", "input": ["ref", "cleaning.clean"]},
                    {"name": "again", "evaluator": "dupes", "input": "ranking.selected"},
                ],
            },
        ],
        evaluators=[{"name": "dupes", "type": "duplicates"}],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["ref", "p1", "p2"], "extractor": "flat"}],
        datasets={"ref": ToyImages(count=16), "p1": ToyImages(count=20, seed=1), "p2": ToyImages(count=12, seed=2)},
        extractor=True,
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert list(result.steps["ranking/selected"].elements or {}) == ["p1", "p2"]
    assert list(result.steps["again"].elements or {}) == ["p1", "p2"]


def test_cleaning_that_empties_a_pool_ranks_it_as_empty() -> None:
    cleaning = {"outliers": {"flags": ["visual"], "outlier_threshold": ("zscore", 0.99)}}
    config = chain_pipeline(
        workflows=[
            {"name": "basic_clean", "type": "quality", **cleaning},
            {"name": "prio", "type": "prioritization", "prioritization": _KNN},
            {
                "name": "outer",
                "inputs": ["ref", {"name": "pools", "list": True}],
                "steps": [
                    {"name": "cleaning", "workflow": "basic_clean", "input": "pools"},
                    {"name": "ranking", "workflow": "prio", "input": ["ref", "cleaning.clean"]},
                ],
            },
        ],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["ref", "p1", "p2"], "extractor": "flat"}],
        datasets={"ref": ToyImages(count=80), "p1": ToyImages(count=80, seed=1), "p2": ToyImages(count=2, seed=2)},
        extractor=True,
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    selected = result.steps["ranking/selected"].elements or {}
    assert {key: len(element.output) for key, element in selected.items()} == {"p1": 72, "p2": 0}
    assert result.report()
    assert cast(Mapping[str, Any], result.to_dict()["steps"])["ranking/prioritization"]
