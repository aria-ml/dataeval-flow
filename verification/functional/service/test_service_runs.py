"""TC-21-2 — Analysis service runs: queueing, one-at-a-time execution, results and logs, cancellation, and history."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dataeval_flow._service._app import create_app
from dataeval_flow._service._manager import RunManager
from dataeval_flow._service._store import RunStore

pytestmark = pytest.mark.required

_SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]
# Holds a run until the file named by its first argument exists.
_GATED = "import pathlib, sys, time\nwhile not pathlib.Path(sys.argv[1]).exists():\n    time.sleep(0.05)\n"
# A run that ignores SIGTERM, so cancelling it has to escalate.
_STUBBORN = (
    "import signal, pathlib, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "pathlib.Path(sys.argv[1]).touch(); time.sleep(60)"
)


def _submit(client: TestClient, pipeline: dict[str, Any], **body: Any) -> dict[str, Any]:
    response = client.post("/v1/runs", json={"pipeline": pipeline, **body})
    assert response.status_code == 202, response.text
    return response.json()


def _appears(path: Path, timeout: float = 10) -> None:
    import time

    end = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < end:
        time.sleep(0.03)
    assert path.exists(), path


class TestRunsOfARealPipeline:
    def test_a_submitted_pipeline_is_queued_run_and_leaves_its_files_with_the_run(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any], wait_for
    ) -> None:
        app = create_app(data_root, tmp_path / "out")
        with TestClient(app) as client:
            record = _submit(client, pipeline)
            assert record["status"] == "queued"
            assert record["request"] == {"tasks": ["digest", "labels"]}
            assert record["api_version"] == 1
            run_id = record["id"]
            final = wait_for(app.state.store, run_id, {"succeeded", "failed"}, timeout=120)
            logs = client.get(f"/v1/runs/{run_id}/logs").text
            assert final["status"] == "succeeded", logs
            assert final["exit_code"] == 0
            assert final["started_at"] is not None
            assert final["finished_at"] >= final["started_at"] >= final["created_at"]
            results = client.get(f"/v1/runs/{run_id}/results").json()
            assert set(results) == {"digest", "labels"}
            assert results["digest"]["output"]["data"]["items"] == 12
            artifacts = client.get(f"/v1/runs/{run_id}/artifacts").json()
            assert {
                "pipeline.json",
                "request.json",
                "console.log",
                "result.log",
                "results/result.json",
                "results/manifests/digest/content-digest.json",
            } <= set(artifacts)
            manifest = client.get(f"/v1/runs/{run_id}/artifacts/results/manifests/digest/content-digest.json").json()
            assert manifest["items"] == 12
            assert "digest" in logs
        assert (tmp_path / "out" / "runs" / run_id / "pipeline.json").is_file()

    def test_only_the_tasks_named_in_the_request_run(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any], wait_for
    ) -> None:
        app = create_app(data_root, tmp_path / "out")
        with TestClient(app) as client:
            run_id = _submit(client, pipeline, tasks=["labels"])["id"]
            assert wait_for(app.state.store, run_id, {"succeeded", "failed"}, timeout=120)["status"] == "succeeded"
            assert set(client.get(f"/v1/runs/{run_id}/results").json()) == {"labels"}

    def test_a_run_that_fails_is_recorded_as_failed_with_its_exit_code_and_a_reason(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any], wait_for
    ) -> None:
        missing = {**pipeline, "datasets": [{"name": "ds", "format": "coco", "path": "nowhere"}]}
        app = create_app(data_root, tmp_path / "out")
        with TestClient(app) as client:
            run_id = _submit(client, missing)["id"]
            final = wait_for(app.state.store, run_id, {"succeeded", "failed"}, timeout=120)
            assert final["status"] == "failed"
            assert final["exit_code"] not in (None, 0, 3, 4)
            assert f"exited with code {final['exit_code']}" in final["error"]
            assert client.get(f"/v1/runs/{run_id}/logs").text


class TestExecution:
    def test_runs_start_one_at_a_time_oldest_first(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any], wait_for
    ) -> None:
        gate = tmp_path / "gate"
        app = create_app(data_root, tmp_path / "out")
        app.state.manager.command = lambda _directory: [sys.executable, "-c", _GATED, str(gate)]
        with TestClient(app) as client:
            ids = [_submit(client, pipeline)["id"] for _ in range(3)]
            store = app.state.store
            wait_for(store, ids[0], {"running"})
            assert [store.get(run_id)["status"] for run_id in ids[1:]] == ["queued", "queued"]
            gate.touch()
            records = [wait_for(store, run_id, {"succeeded"}) for run_id in ids]
        started = [record["started_at"] for record in records]
        assert started == sorted(started)
        assert records[0]["finished_at"] <= records[1]["started_at"]
        assert records[1]["finished_at"] <= records[2]["started_at"]

    @pytest.mark.parametrize(("code", "status"), [(0, "succeeded"), (3, "succeeded"), (4, "succeeded"), (1, "failed")])
    def test_the_exit_code_decides_the_final_status(self, tmp_path: Path, wait_for, code: int, status: str) -> None:
        store = RunStore(tmp_path / "runs")
        manager = RunManager(
            store, tmp_path, tmp_path / "cache", lambda _: [sys.executable, "-c", f"import sys; sys.exit({code})"]
        )
        run_id = manager.submit({}, {})["id"]
        manager.start()
        try:
            final = wait_for(store, run_id, {"succeeded", "failed"})
        finally:
            manager.stop()
        assert (final["status"], final["exit_code"]) == (status, code)

    def test_a_run_killed_by_a_signal_or_never_started_is_failed_with_a_reason(self, tmp_path: Path, wait_for) -> None:
        commands = {
            "killed": [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"],
            "missing": ["/nonexistent/command"],
        }
        store = RunStore(tmp_path / "runs")
        manager = RunManager(
            store,
            tmp_path,
            tmp_path / "cache",
            lambda d: commands[json.loads((d / "request.json").read_text())["mode"]],
        )
        ids = {mode: manager.submit({"mode": mode}, {})["id"] for mode in commands}
        manager.start()
        try:
            finals = {mode: wait_for(store, run_id, {"failed"}) for mode, run_id in ids.items()}
        finally:
            manager.stop()
        assert "killed by signal 9" in finals["killed"]["error"]
        assert "could not start" in finals["missing"]["error"]

    def test_a_service_setting_in_the_environment_never_reaches_a_run(
        self, tmp_path: Path, wait_for, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DATAEVAL_REQUIRE", "ready")
        seen = tmp_path / "seen.txt"
        code = (
            "import os, pathlib; "
            f"pathlib.Path({str(seen)!r}).write_text(repr(sorted(k for k in os.environ if k.startswith('DATAEVAL_'))))"
        )
        store = RunStore(tmp_path / "runs")
        manager = RunManager(store, tmp_path, tmp_path / "cache", lambda _: [sys.executable, "-c", code])
        run_id = manager.submit({}, {})["id"]
        manager.start()
        try:
            wait_for(store, run_id, {"succeeded"})
        finally:
            manager.stop()
        assert seen.read_text() == "[]"

    def test_only_one_service_may_run_the_queue_of_an_output_directory(self, data_root: Path, tmp_path: Path) -> None:
        with (
            TestClient(create_app(data_root, tmp_path / "out")),
            pytest.raises(RuntimeError, match="Another service is running the queue"),
        ):
            RunManager(RunStore(tmp_path / "out" / "runs"), data_root, tmp_path / "cache").start()


class TestResultsAndLogs:
    def test_a_running_run_shows_the_tasks_it_has_finished(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any], wait_for
    ) -> None:
        app = create_app(data_root, tmp_path / "out")
        app.state.manager.command = lambda _directory: _SLEEP
        with TestClient(app) as client:
            run_id = _submit(client, pipeline)["id"]
            wait_for(app.state.store, run_id, {"running"})
            prefix = f"/v1/runs/{run_id}"
            assert client.get(prefix + "/results").status_code == 409
            directory = app.state.store.directory(run_id)
            (directory / "results").mkdir()
            (directory / "results" / "result.json").write_text(json.dumps({"digest": {"success": True}}))
            assert client.get(prefix + "/results").json() == {"digest": {"success": True}}
            assert app.state.store.get(run_id)["status"] == "running"
            client.post(prefix + "/cancel")

    def test_the_batch_command_rewrites_result_json_to_hold_each_task_as_it_finishes(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dataeval_flow import _orchestrator
        from dataeval_flow._runner import run
        from verification.functional.integrity._data import write_pipeline

        out = tmp_path / "batch"
        config = write_pipeline(tmp_path / "pipeline.yaml", pipeline)
        real = _orchestrator.run_tasks
        held: list[list[str]] = []

        def spy(*args: Any, on_result: Any, **kwargs: Any) -> Any:
            def finished(name: str, result: Any) -> None:
                on_result(name, result)
                held.append(sorted(json.loads((out / "results" / "result.json").read_text())))

            return real(*args, on_result=finished, **kwargs)

        monkeypatch.setattr(_orchestrator, "run_tasks", spy)
        assert run(config, out, data_dir=data_root, report_images=False) == 0
        assert held == [["digest"], ["digest", "labels"]]

    def test_per_task_result_files_are_merged_into_one_object(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any], wait_for
    ) -> None:
        per_task = {**pipeline, "result": {"per_task": True, "name": "out"}}
        app = create_app(data_root, tmp_path / "out")
        app.state.manager.command = lambda _directory: _SLEEP
        with TestClient(app) as client:
            run_id = _submit(client, per_task)["id"]
            wait_for(app.state.store, run_id, {"running"})
            results = app.state.store.directory(run_id) / "results"
            results.mkdir()
            (results / "out-digest.json").write_text(json.dumps({"digest": 1}))
            (results / "out-labels.json").write_text(json.dumps({"labels": 2}))
            assert client.get(f"/v1/runs/{run_id}/results").json() == {"digest": 1, "labels": 2}
            client.post(f"/v1/runs/{run_id}/cancel")

    def test_logs_return_the_last_64_kib_of_the_console_output(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any], wait_for
    ) -> None:
        app = create_app(data_root, tmp_path / "out")
        app.state.manager.command = lambda _directory: _SLEEP
        with TestClient(app) as client:
            run_id = _submit(client, pipeline)["id"]
            wait_for(app.state.store, run_id, {"running"})
            assert client.get(f"/v1/runs/{run_id}/logs").text == ""
            (app.state.store.directory(run_id) / "console.log").write_text("early\n" + "x" * 70000 + "\nlate\n")
            tail = client.get(f"/v1/runs/{run_id}/logs").text
            client.post(f"/v1/runs/{run_id}/cancel")
        assert len(tail.encode()) == 65536
        assert tail.endswith("late\n")
        assert "early" not in tail

    def test_an_artifact_outside_the_run_s_directory_is_not_found(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any], wait_for
    ) -> None:
        app = create_app(data_root, tmp_path / "out")
        app.state.manager.command = lambda _directory: _SLEEP
        with TestClient(app) as client:
            run_id = _submit(client, pipeline)["id"]
            wait_for(app.state.store, run_id, {"running"})
            prefix = f"/v1/runs/{run_id}/artifacts"
            assert client.get(prefix + "/%2E%2E/runs.sqlite3").status_code == 404
            assert client.get(prefix + "/missing.json").status_code == 404
            assert client.get(prefix + "/pipeline.json").status_code == 200
            client.post(f"/v1/runs/{run_id}/cancel")

    def test_an_unknown_run_is_not_found_on_every_run_route(self, data_root: Path, tmp_path: Path) -> None:
        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            for route in ("", "/results", "/logs", "/artifacts", "/items/data", "/items/data/0"):
                assert client.get(f"/v1/runs/missing{route}").status_code == 404, route
            assert client.post("/v1/runs/missing/cancel").status_code == 404


class TestCancellation:
    def test_a_queued_run_is_cancelled_without_ever_starting(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any], wait_for
    ) -> None:
        app = create_app(data_root, tmp_path / "out")
        app.state.manager.command = lambda _directory: _SLEEP
        with TestClient(app) as client:
            first, second = (_submit(client, pipeline)["id"] for _ in range(2))
            wait_for(app.state.store, first, {"running"})
            response = client.post(f"/v1/runs/{second}/cancel")
            assert response.status_code == 202
            assert response.json()["status"] == "cancelled"
            assert client.get(f"/v1/runs/{second}").json()["started_at"] is None
            client.post(f"/v1/runs/{first}/cancel")

    def test_a_running_run_is_stopped_and_the_next_queued_run_starts(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any], wait_for
    ) -> None:
        app = create_app(data_root, tmp_path / "out")
        app.state.manager.command = lambda _directory: _SLEEP
        with TestClient(app) as client:
            first, second = (_submit(client, pipeline)["id"] for _ in range(2))
            store = app.state.store
            wait_for(store, first, {"running"})
            assert client.post(f"/v1/runs/{first}/cancel").status_code == 202
            assert wait_for(store, first, {"cancelled"})["finished_at"] is not None
            wait_for(store, second, {"running"})
            client.post(f"/v1/runs/{second}/cancel")
            wait_for(store, second, {"cancelled"})

    def test_a_run_that_ignores_the_stop_signal_is_killed(self, tmp_path: Path, wait_for) -> None:
        store = RunStore(tmp_path / "runs")
        manager = RunManager(store, tmp_path, tmp_path, lambda d: [sys.executable, "-c", _STUBBORN, str(d / "ready")])
        run_id = manager.submit({}, {})["id"]
        manager.start()
        try:
            wait_for(store, run_id, {"running"})
            _appears(store.directory(run_id) / "ready")
            assert manager.cancel(run_id)["status"] == "cancelling"
            assert wait_for(store, run_id, {"cancelled"}, timeout=30)["status"] == "cancelled"
        finally:
            manager.stop()

    def test_cancelling_a_finished_run_is_refused_with_409(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any], wait_for
    ) -> None:
        app = create_app(data_root, tmp_path / "out")
        app.state.manager.command = lambda _directory: _SLEEP
        with TestClient(app) as client:
            run_id = _submit(client, pipeline)["id"]
            wait_for(app.state.store, run_id, {"running"})
            client.post(f"/v1/runs/{run_id}/cancel")
            wait_for(app.state.store, run_id, {"cancelled"})
            response = client.post(f"/v1/runs/{run_id}/cancel")
            assert response.status_code == 409
            assert "already finished" in response.json()["detail"]
            assert client.get(f"/v1/runs/{run_id}").json()["status"] == "cancelled"


class TestHistory:
    def test_history_lists_every_run_newest_first_and_a_restarted_service_reads_it_back(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any], wait_for
    ) -> None:
        app = create_app(data_root, tmp_path / "out")
        app.state.manager.command = lambda _directory: [sys.executable, "-c", "pass"]
        with TestClient(app) as client:
            ids = [_submit(client, pipeline)["id"] for _ in range(3)]
            for run_id in ids:
                wait_for(app.state.store, run_id, {"succeeded"})
            history = client.get("/v1/runs").json()
            assert [run["id"] for run in history] == ids[::-1]
        with TestClient(create_app(data_root, tmp_path / "out")) as restarted:
            again = restarted.get("/v1/runs").json()
            assert [run["id"] for run in again] == ids[::-1]
            assert {run["status"] for run in again} == {"succeeded"}
            assert restarted.get(f"/v1/runs/{ids[0]}").json()["pipeline"] == history[-1]["pipeline"]

    def test_stopping_the_service_interrupts_the_running_run_and_keeps_the_queued_ones(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any], wait_for
    ) -> None:
        app = create_app(data_root, tmp_path / "out")
        app.state.manager.command = lambda _directory: _SLEEP
        with TestClient(app) as client:
            running, queued = (_submit(client, pipeline)["id"] for _ in range(2))
            wait_for(app.state.store, running, {"running"})
        interrupted = app.state.store.get(running)
        assert interrupted["status"] == "interrupted"
        assert interrupted["error"] == "The service stopped mid-run"
        assert app.state.store.get(queued)["status"] == "queued"

        restarted = create_app(data_root, tmp_path / "out")
        restarted.state.manager.command = lambda _directory: [sys.executable, "-c", "pass"]
        with TestClient(restarted) as client:
            wait_for(restarted.state.store, queued, {"succeeded"})
            assert client.get(f"/v1/runs/{running}").json()["status"] == "interrupted"
            assert client.post(f"/v1/runs/{running}/cancel").status_code == 409

    def test_a_run_lost_to_a_crash_is_marked_interrupted_when_the_service_starts_again(
        self, data_root: Path, tmp_path: Path
    ) -> None:
        store = RunStore(tmp_path / "out" / "runs")
        lost = store.create({"tasks": []}, {})["id"]
        store.update(lost, status="running")
        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            record = client.get(f"/v1/runs/{lost}").json()
        assert record["status"] == "interrupted"
        assert record["error"] == "The service stopped mid-run"

    def test_a_finished_record_never_changes(self, tmp_path: Path) -> None:
        store = RunStore(tmp_path / "runs")
        run_id = store.create({}, {})["id"]
        store.update(run_id, status="failed", exit_code=1)
        assert store.update(run_id, status="succeeded", exit_code=0)["status"] == "failed"
        assert store.get(run_id)["exit_code"] == 1
