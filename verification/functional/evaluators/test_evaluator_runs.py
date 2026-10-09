"""TC-20-2 — running every evaluator type as a task and through `run()`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from dataeval_flow import run, run_tasks
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators import EvaluatorResult, get_evaluator
from verification.functional.chains._toys import EVALUATOR_DATA, EVALUATOR_SETTINGS, FLAT, pipeline

pytestmark = pytest.mark.required

_TYPES = sorted(EVALUATOR_DATA)


def _config(name: str, entry_name: str | None = None) -> Any:
    kwargs = {"name": entry_name} if entry_name else {}
    return get_evaluator(name).config_type(**kwargs, **EVALUATOR_SETTINGS.get(name, {}))


def _run(name: str) -> EvaluatorResult[Any]:
    data, needs_extractor = EVALUATOR_DATA[name]
    return run(_config(name), data, extractor=FLAT if needs_extractor else None)


def _run_as_task(name: str) -> EvaluatorResult[Any]:
    data, needs_extractor = EVALUATOR_DATA[name]
    datasets = data if isinstance(data, Mapping) else {"src": data}
    task = TaskConfig(
        name="t",
        evaluator="e",  # type: ignore[call-arg]
        sources=list(datasets),
        extractor="flat" if needs_extractor else None,
    )
    config = pipeline(datasets, evaluators=[_config(name, "e")], tasks=[task], extractor=needs_extractor)
    return run_tasks(config)["t"]  # type: ignore[return-value]


class TestEvaluatorTasks:
    @pytest.mark.parametrize("name", _TYPES)
    def test_every_evaluator_type_runs_as_a_task(self, name: str) -> None:
        result = _run_as_task(name)
        assert isinstance(result, EvaluatorResult)
        assert result.success, result.errors
        assert result.kind == "evaluator"
        assert result.type == name
        assert result.metadata.evaluator == name
        assert result.report().strip()
        payload = result.to_dict()["output"]
        assert payload["shape"] in {"table", "mapping", "array"}

    @pytest.mark.parametrize("name", _TYPES)
    def test_every_evaluator_type_runs_through_run_and_returns_its_config_s_result_class(self, name: str) -> None:
        result = _run(name)
        assert type(result) is _config(name).result_type
        assert result.success, result.errors
        assert result.output is not None
        assert result.metadata.evaluator == name

    def test_a_task_and_run_give_the_same_determinations(self) -> None:
        by_task = _run_as_task("label-health").to_dict()["output"]
        by_run = _run("label-health").to_dict()["output"]
        assert by_task == by_run
        assert by_run["data"]["label_counts_per_class"] == {"a": 20, "b": 20}

    def test_a_task_records_the_evaluator_entry_and_extractor_it_ran_with(self) -> None:
        result = _run_as_task("drift-mmd")
        resolved = result.metadata.resolved_config
        assert resolved["evaluator"]["type"] == "drift-mmd"
        assert resolved["extractor"]["model"] == "flatten"
        assert [source["name"] for source in resolved["sources"]] == ["reference", "test"]
        assert result.metadata.model_id == "flat (flatten)"

    def test_a_task_keeps_the_sources_it_read_in_the_order_it_named_them(self) -> None:
        result = _run_as_task("drift-mmd")
        assert result.sources is not None
        assert list(result.sources) == ["reference", "test"]

    def test_a_failing_evaluator_task_is_a_failed_result_and_the_other_tasks_still_run(self) -> None:
        # factor-leakage reads two sources, so name the same one twice through a second source of the same data.
        config = pipeline(
            {"a": EVALUATOR_DATA["balance"][0], "b": EVALUATOR_DATA["balance"][0]},
            evaluators=[
                {"name": "bad", "type": "factor-leakage", "factors": ["missing"]},
                {"name": "ok", "type": "label-health"},
            ],
            tasks=[
                {"name": "bad_task", "evaluator": "bad", "sources": ["a", "b"]},
                {"name": "ok_task", "evaluator": "ok", "sources": ["a"]},
            ],
        )
        results = run_tasks(config)
        assert list(results) == ["bad_task", "ok_task"]
        bad, ok = results["bad_task"], results["ok_task"]
        assert isinstance(bad, EvaluatorResult)
        assert not bad.success
        assert any("missing" in error for error in bad.errors)
        assert "output" not in bad.to_dict() or bad.to_dict()["output"] is None
        assert ok.success
        with pytest.raises(Exception, match="missing"):
            bad.output  # noqa: B018
