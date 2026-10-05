"""An optional step that needs an extractor neither the task nor the step names is skipped, not refused
(data-splitting spec §5.2). `cluster_threshold` puts `outliers` in cluster mode, which reads clusters, so it needs an
extractor."""

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow._cache import DatasetCache
from dataeval_flow.steps import ChainResult
from tests.chain_toys import chain_pipeline, run_chain_task
from tests.evaluator_toys import ToyImages

_REASON = "requires an extractor"


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _config(*, optional: bool, extractor: bool = False) -> Any:
    steps = [
        {"name": "outliers", "evaluator": "clustered", "input": "data", "optional": optional},
        {"name": "rate", "check": "image-outliers", "input": "outliers"},
        {"name": "kept", "transform": "remove", "input": "data", "plans": {"outliers": {"min_flags": 1}}},
    ]
    task: dict[str, Any] = {"name": "t", "workflow": "w", "sources": ["src"]}
    if extractor:
        task["extractor"] = "flat"
    return chain_pipeline(
        workflows=[{"name": "w", "inputs": ["data"], "steps": steps}],
        evaluators=[{"name": "clustered", "type": "outliers", "cluster_threshold": 2.0}],
        tasks=[task],
        datasets={"src": ToyImages(count=12)},
        extractor=extractor,
    )


def _run(config: Any) -> ChainResult:
    result = run_chain_task(config)
    assert isinstance(result, ChainResult)
    return result


def test_a_required_step_with_no_extractor_is_still_refused_at_load() -> None:
    with pytest.raises(ValidationError, match="whose step 'outliers' needs an extractor"):
        _config(optional=False)


def test_an_optional_step_with_no_extractor_is_skipped_with_the_reason() -> None:
    record = _run(_config(optional=True)).steps["outliers"]
    assert (record.status, record.reason) == ("skipped", _REASON)


def test_a_transform_reading_it_is_skipped_with_it() -> None:
    record = _run(_config(optional=True)).steps["kept"]
    assert (record.status, record.reason) == ("skipped", "needs `outliers`, which was skipped")


def test_a_check_reading_it_is_not_assessed() -> None:
    result = _run(_config(optional=True))
    assert [(f.severity, f.brief, f.step) for f in result.findings] == [("info", "not assessed", "rate")]


def test_with_an_extractor_the_optional_step_runs() -> None:
    assert _run(_config(optional=True, extractor=True)).steps["outliers"].status == "ok"
