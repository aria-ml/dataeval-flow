"""The worker uses real Flow results and retains partial outcomes when later work fails."""

import json
import signal
import sys
from types import SimpleNamespace

import pytest

from dataeval_flow_service import _worker
from dataeval_flow_service._store import RunStore


def snapshot(staged, tmp_path):
    root, pipeline = staged
    store = RunStore(tmp_path / "runs")
    run = store.create({"tasks": ["quality", "metadata"]}, pipeline)
    return root, store.directory(run["id"])


def test_real_quality_and_metadata_workflows(staged, tmp_path):
    root, directory = snapshot(staged, tmp_path)
    assert _worker.execute(directory, root, tmp_path / "cache")
    payload = json.loads((directory / "results.json").read_text())
    assert set(payload["tasks"]) == {"quality", "metadata"}
    assert all(task["success"] for task in payload["tasks"].values())
    # Duplicates and imbalance in the fixture must reach the structured findings.
    text = json.dumps(payload["tasks"]["quality"]["result"])
    assert "Duplicate" in text or "duplicate" in text
    assert "swimmer" in text and "boat" in text
    assert (directory / "quality.html").is_file() and (directory / "metadata.html").is_file()
    assert json.loads((directory / "outcome.json").read_text())["success"] is True
    assert len((directory / "events.jsonl").read_text().splitlines()) == 5


def test_partial_results_and_failed_result(staged, tmp_path, monkeypatch):
    root, directory = snapshot(staged, tmp_path)
    result = SimpleNamespace(
        success=False,
        errors=["insufficient fixture"],
        to_dict=lambda: {"errors": ["failure"]},
        to_html=lambda: "<html>Failed result</html>",
    )
    calls = []

    def task(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("second task exploded")
        return result

    monkeypatch.setattr(_worker, "run_task", task)
    assert not _worker.execute(directory, root, tmp_path / "cache")
    payload = json.loads((directory / "results.json").read_text())
    assert set(payload["tasks"]) == {"quality"} and payload["tasks"]["quality"]["success"] is False
    assert "second task exploded" in json.loads((directory / "outcome.json").read_text())["error"]
    monkeypatch.setattr(_worker, "run_task", lambda *args, **kwargs: result)
    assert not _worker.execute(directory, root, tmp_path / "cache")
    assert json.loads((directory / "outcome.json").read_text())["error"] == "One or more tasks failed"
    (directory / "pipeline.json").write_text("not json")
    assert not _worker.execute(directory, root, tmp_path / "cache")
    assert "ValidationError" in json.loads((directory / "outcome.json").read_text())["error"]


def test_worker_parent_watch_and_entrypoint(tmp_path, monkeypatch):
    parents = iter([10, 11])
    signals = []
    monkeypatch.setattr(_worker.os, "getppid", lambda: next(parents))
    monkeypatch.setattr(_worker.os, "getpgrp", lambda: 99)
    monkeypatch.setattr(_worker.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(_worker.time, "sleep", lambda _: None)
    _worker._watch_parent(10)
    assert signals == [(99, signal.SIGTERM)]
    monkeypatch.setattr(_worker.os, "getppid", lambda: 10)
    monkeypatch.setattr(_worker.threading, "Thread", lambda **kwargs: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(sys, "argv", ["worker", str(tmp_path), str(tmp_path), str(tmp_path)])
    monkeypatch.setattr(_worker, "execute", lambda *args: True)
    with pytest.raises(SystemExit) as exited:
        _worker.main()
    assert exited.value.code == 0
