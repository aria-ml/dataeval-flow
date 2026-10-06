"""HTTP contracts expose durable results and contain input/artifact paths."""

import json
import sys

from fastapi.testclient import TestClient
from test_lifecycle import wait_for

from dataeval_flow_service import _cli
from dataeval_flow_service._app import create_app


def test_api_and_history_snapshot(staged, tmp_path):
    root, pipeline = staged
    app = create_app(root, tmp_path / "runs")
    app.state.manager.command = lambda _: [sys.executable, "-c", "import time; time.sleep(60)"]
    request = {"pipeline": pipeline, "tasks": ["metadata"]}
    with TestClient(app) as client:
        assert client.get("/v1/health").json()["api_version"] == 1
        assert client.get("/v1/runs/missing").status_code == 404
        assert client.post("/v1/validate", json={**request, "tasks": ["missing"]}).status_code == 422
        assert client.post("/v1/validate", json={**request, "unexpected": True}).status_code == 422
        invalid = {**pipeline, "workflows": [{"name": "quality", "type": "data-cleaning"}]}
        assert client.post("/v1/validate", json={"pipeline": invalid}).status_code == 422
        checked = client.post("/v1/validate", json=request).json()
        assert checked["valid"] and checked["tasks"] == ["metadata"]
        submitted = client.post("/v1/runs", json=request)
        assert submitted.status_code == 202
        run_id = submitted.json()["id"]
        assert submitted.json()["request"] == {"tasks": ["metadata"]}
        wait_for(app.state.store, run_id, {"running"})
        prefix = f"/v1/runs/{run_id}"
        assert client.get(prefix + "/results").status_code == 409
        assert client.get(prefix + "/events").json() == []
        assert client.get(prefix + "/logs").status_code == 200
        directory = app.state.store.directory(run_id)
        (directory / "results.json").write_text(json.dumps({"format": 1, "tasks": {}}))
        (directory / "events.jsonl").write_text('{"type":"task_started"}\n{"partial":')
        (directory / "worker.log").write_bytes(b"x" * 70000 + b"\xff")
        assert client.get(prefix + "/results").json()["format"] == 1
        assert client.get(prefix + "/events").json() == [{"type": "task_started"}]
        assert len(client.get(prefix + "/logs").content) <= 65539
        assert "pipeline.json" in client.get(prefix + "/artifacts").json()
        assert client.get(prefix + "/artifacts/pipeline.json").json() == checked["pipeline"]
        assert client.get(prefix + "/artifacts/worker.log").status_code == 404
        assert client.post(prefix + "/cancel").status_code == 202
        wait_for(app.state.store, run_id, {"cancelled"})
        assert client.post(prefix + "/cancel").status_code == 409
    # A new service instance and a new client both read the same saved history.
    with TestClient(create_app(root, tmp_path / "runs")) as restarted:
        assert restarted.get("/v1/runs").json()[0]["id"] == run_id
        assert restarted.get(prefix).json()["status"] == "cancelled"


def test_disabled_tasks_rejected(staged, tmp_path):
    root, pipeline = staged
    disabled = {**pipeline, "tasks": [{**task, "enabled": False} for task in pipeline["tasks"]]}
    with TestClient(create_app(root, tmp_path / "runs")) as client:
        response = client.post("/v1/validate", json={"pipeline": disabled})
        assert response.status_code == 422 and "disabled" in response.text


def test_empty_logs_and_cli(tmp_path, monkeypatch):
    app = create_app(tmp_path / "data", tmp_path / "runs")
    run = app.state.store.create({}, {})
    with TestClient(app) as client:
        assert client.get(f"/v1/runs/{run['id']}/logs").text == ""
    captured = {}
    monkeypatch.setattr(_cli.uvicorn, "run", lambda app, **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "service",
            "--data",
            str(tmp_path),
            "--output",
            str(tmp_path / "other"),
            "--cache",
            str(tmp_path / "cache"),
            "--port",
            "8123",
        ],
    )
    _cli.main()
    assert captured == {"host": "127.0.0.1", "port": 8123}
