"""The HTTP contract: probes, validation, run history and results, against a stand-in run process."""

import json
import sys

import pytest

pytest.importorskip("fastapi", reason="needs the 'service' extra")

from fastapi.testclient import TestClient

from dataeval_flow._service import _app
from dataeval_flow._service._app import create_app
from dataeval_flow.config import PipelineConfig

pytestmark = pytest.mark.optional

_SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]


def test_probes_report_what_stops_the_service(data_root, tmp_path, monkeypatch):
    app = create_app(data_root, tmp_path / "out")
    manager, store = app.state.manager, app.state.store
    with TestClient(app) as client:
        for probe in ("/livez", "/readyz", "/healthz"):
            assert client.get(probe).json() == {"status": "ok"}
        monkeypatch.setattr(store, "usable", lambda: False)
        assert client.get("/livez").status_code == 200
        unready = client.get("/readyz")
        assert unready.status_code == 503
        assert unready.json()["reasons"] == ["store-unavailable"]
        assert client.get("/healthz").status_code == 503
        monkeypatch.undo()
        # A queue consumer that died needs a restart: liveness fails too.
        manager.stopping.set()
        manager.wake.set()
        manager.thread.join()
        manager.stopping.clear()
        assert client.get("/livez").json() == {"status": "unavailable", "reasons": ["queue-stopped"]}
        assert client.get("/readyz").json()["reasons"] == ["queue-stopped"]


def test_capabilities_and_interface_documentation(data_root, tmp_path):
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        capabilities = client.get("/v1/capabilities").json()
        assert capabilities["api_version"] == 1
        assert capabilities["max_active_runs"] == 1
        assert capabilities["steps"]["format"] == 1
        schema = client.get("/openapi.json").json()
        assert {"/livez", "/readyz", "/healthz", "/v1/runs", "/v1/runs/{run_id}/cancel"} <= set(schema["paths"])
        operations = [op["operationId"] for path in schema["paths"].values() for op in path.values()]
        assert len(operations) == len(set(operations))


def test_validation_refuses_what_would_fail_before_a_task_ran(data_root, tmp_path, pipeline):
    with TestClient(create_app(data_root, tmp_path / "out")) as client:

        def status(body):
            return client.post("/v1/validate", json=body).status_code

        assert status({"pipeline": pipeline, "tasks": ["missing"]}) == 422
        assert status({"pipeline": pipeline, "unexpected": True}) == 422
        assert status({"pipeline": {**pipeline, "workflows": [{"name": "quality", "type": "data-cleaning"}]}}) == 422
        assert status({"pipeline": {**pipeline, "result": {"formats": ["html"]}}}) == 422
        assert status({"pipeline": {**pipeline, "result": {"require": "ready"}}}) == 422
        disabled = [{**task, "enabled": False} for task in pipeline["tasks"]]
        response = client.post("/v1/validate", json={"pipeline": {**pipeline, "tasks": disabled}})
        assert response.status_code == 422
        assert "disabled" in response.text
        checked = client.post("/v1/validate", json={"pipeline": pipeline}).json()
        assert checked["tasks"] == ["quality", "metadata"]
        # The snapshot holds every setting, defaults included, and reads back as the same pipeline.
        assert PipelineConfig.model_validate(checked["pipeline"]) == PipelineConfig.model_validate(pipeline)


def test_runs_history_results_and_artifacts(data_root, tmp_path, pipeline, wait_for):
    app = create_app(data_root, tmp_path / "out")
    app.state.manager.command = lambda _: _SLEEP
    with TestClient(app) as client:
        assert client.get("/v1/runs/missing").status_code == 404
        submitted = client.post("/v1/runs", json={"pipeline": pipeline, "tasks": ["metadata"]})
        assert submitted.status_code == 202
        run_id = submitted.json()["id"]
        assert submitted.json()["request"] == {"tasks": ["metadata"]}
        wait_for(app.state.store, run_id, {"running"})
        prefix = f"/v1/runs/{run_id}"
        assert client.get(prefix + "/results").status_code == 409
        assert client.get(prefix + "/logs").text == ""
        directory = app.state.store.directory(run_id)
        (directory / "results").mkdir()
        (directory / "results" / "result.json").write_text(json.dumps({"metadata": {"success": True}}))
        (directory / "console.log").write_bytes(b"x" * 70000 + b"\xff")
        assert client.get(prefix + "/results").json() == {"metadata": {"success": True}}
        assert len(client.get(prefix + "/logs").content) <= 65539
        assert {"pipeline.json", "request.json", "results/result.json"} <= set(client.get(prefix + "/artifacts").json())
        assert client.get(prefix + "/artifacts/results/result.json").json() == {"metadata": {"success": True}}
        assert client.get(prefix + "/artifacts/%2E%2E/runs.sqlite3").status_code == 404
        assert client.get(prefix + "/artifacts/missing.json").status_code == 404
        assert client.post(prefix + "/cancel").status_code == 202
        wait_for(app.state.store, run_id, {"cancelled"})
        assert client.post(prefix + "/cancel").status_code == 409
    # A new service, and a new client, read the same history.
    with TestClient(create_app(data_root, tmp_path / "out")) as restarted:
        assert restarted.get("/v1/runs").json()[0]["id"] == run_id
        assert restarted.get(prefix).json()["status"] == "cancelled"


def test_per_task_result_files_are_merged(tmp_path):
    run = {"pipeline": {"result": {"per_task": True, "name": "out"}}, "request": {"tasks": ["a", "b", "c"]}}
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "out-a.json").write_text(json.dumps({"a": 1}))
    (tmp_path / "results" / "out-b.json").write_text(json.dumps({"b": 2}))
    assert _app._results(tmp_path, run) == {"a": 1, "b": 2}


def test_serve_logs_through_flow_and_runs_uvicorn(tmp_path, monkeypatch):
    from dataeval_flow import _logging

    captured = {}
    monkeypatch.setattr(_logging, "setup_logging", lambda **kwargs: captured.update(logging=kwargs))
    monkeypatch.setattr(_app.logging, "captureWarnings", lambda capture: captured.update(warnings=capture))
    monkeypatch.setattr(_app.uvicorn, "run", lambda app, **kwargs: captured.update(kwargs, app=app))
    _app.serve(tmp_path / "data", tmp_path / "out", None, "0.0.0.0", 8123, "plain")  # noqa: S104
    assert captured["logging"] == {"verbosity": 2, "log_format": "plain"}
    assert captured["warnings"] is True
    assert captured["host"] == "0.0.0.0"  # noqa: S104
    assert captured["port"] == 8123
    assert captured["log_config"] is None
    assert captured["app"].state.manager.cache_root == (tmp_path / "out" / "cache").resolve()
