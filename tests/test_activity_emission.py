"""A run records each task's and step's state as it changes (builder spec §6.2)."""

from typing import Any

import pytest

from dataeval_flow import _orchestrator, run_tasks
from dataeval_flow._activity import fold, read_events, recording
from dataeval_flow._cache import DatasetCache
from dataeval_flow._service._worker import _activity_path
from dataeval_flow.evaluators.quality import DuplicatesConfig
from tests.chain_toys import chain_pipeline, register_toys
from tests.evaluator_toys import ToyImages


@pytest.fixture(autouse=True)
def toys(plugins):
    register_toys(plugins)
    DatasetCache.clear_instances()
    yield plugins
    DatasetCache.clear_instances()


def _recorded(tmp_path, steps: list[dict[str, Any]], task: dict[str, Any], inputs: list[Any] | None = None, **kwargs):
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": inputs or ["a"], "steps": steps}],
        tasks=[task],
        evaluators=[DuplicatesConfig(name="dupes")],
        **kwargs,
    )
    path = tmp_path / "activity.jsonl"
    with recording(path):
        results = run_tasks(config)
    return results, read_events(path)


def test_each_step_goes_pending_running_then_to_its_outcome(tmp_path):
    steps = [
        {"name": "boom", "transform": "toy-explode", "input": "a"},
        {"name": "after", "transform": "toy-keep", "input": "boom"},
        {"name": "kept", "transform": "toy-keep", "input": "a"},
    ]
    results, events = _recorded(tmp_path, steps, {"name": "t", "workflow": "w", "sources": "src"})
    history: dict[str, list[str]] = {}
    for event in events:
        if event["kind"] == "step_state":
            history.setdefault(event["step"]["node_id"], []).append(event["step"]["status"])
    assert history == {
        "t:boom": ["pending", "running", "failed"],
        "t:after": ["pending", "running", "skipped"],
        "t:kept": ["pending", "running", "ok"],
    }
    state = fold(events)
    rows = {row["node_id"]: row for row in state["steps"]}
    assert rows["t:boom"]["reason"] == "RuntimeError: boom on a"
    assert rows["t:after"]["reason"] == "needs `boom`, which failed: RuntimeError: boom on a"
    assert [event["task"]["status"] for event in events if event["kind"] == "task_state"] == [
        "pending",
        "running",
        "ok" if results["t"].success else "failed",
    ]


def test_a_step_run_per_element_records_each_elements_outcome(tmp_path):
    _, events = _recorded(
        tmp_path,
        [{"name": "boom", "transform": "toy-explode", "input": "all", "only": "all[more]"}],
        {"name": "t", "workflow": "w", "sources": ["src", "more"]},
        inputs=[{"name": "all", "list": True}],
        datasets={"src": ToyImages(), "more": ToyImages(seed=1)},
    )
    (row,) = fold(events)["steps"]
    assert {key: element["status"] for key, element in row["elements"].items()} == {"src": "ok", "more": "failed"}


def test_each_matrix_run_records_its_own_instances(tmp_path):
    _, events = _recorded(
        tmp_path,
        [{"name": "few", "transform": "toy-first", "input": "a", "n": 2}],
        {"name": "t", "workflow": "w", "sources": "src", "matrix": {"steps.few.n": [2, 3]}},
    )
    rows = {row["node_id"]: row for row in fold(events)["steps"]}
    assert set(rows) == {"t:few#1", "t:few#2"}
    assert all(row["definition_node_id"] == "t:few" and row["status"] == "ok" for row in rows.values())


def test_a_task_that_raises_is_recorded_as_failed(tmp_path, monkeypatch):
    def unresolvable(*_args: Any, **_kwargs: Any):
        raise ValueError("no such dataset")

    monkeypatch.setattr(_orchestrator, "_run_single_task", unresolvable)
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["a"], "steps": [{"name": "k", "transform": "toy-keep", "input": "a"}]}],
        tasks=[{"name": "t", "workflow": "w", "sources": "src"}],
    )
    path = tmp_path / "activity.jsonl"
    with recording(path), pytest.raises(ValueError, match="no such dataset"):
        run_tasks(config)
    task = fold(read_events(path))["tasks"][0]
    assert (task["status"], task["reason"]) == ("failed", "ValueError: no such dataset")


def test_the_worker_takes_its_activity_file_off_the_command_line(tmp_path):
    argv = ["worker", "--activity", str(tmp_path / "a.jsonl"), "--config", "x"]
    assert _activity_path(argv) == tmp_path / "a.jsonl"
    assert argv == ["worker", "--config", "x"]
    assert _activity_path(["worker"]) is None
