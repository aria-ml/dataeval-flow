"""The builder contracts' shared behavior: JSON 404s, the extension error, limits and request bounds (builder spec §4,
§7)."""

import sys

import pytest

pytest.importorskip("fastapi", reason="needs the 'service' extra")

from fastapi.testclient import TestClient

from dataeval_flow._service import _contract
from dataeval_flow._service._app import create_app

pytestmark = pytest.mark.optional


def test_an_unknown_run_is_a_json_404(data_root, tmp_path):
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        response = client.get("/v1/runs/missing")
        assert response.status_code == 404
        assert response.json() == {"detail": "Unknown run"}


def test_a_finished_run_without_a_result_says_so(data_root, tmp_path, pipeline, wait_for):
    app = create_app(data_root, tmp_path / "out")
    app.state.manager.command = lambda _: [sys.executable, "-c", "raise SystemExit(1)"]
    with TestClient(app) as client:
        run = client.post("/v1/runs", json={"pipeline": pipeline}).json()
        wait_for(app.state.store, run["id"], {"failed"})
        response = client.get(f"/v1/runs/{run['id']}/results")
        assert response.status_code == 409
        assert response.json() == {"detail": "The run finished without a result"}


def test_capabilities_advertise_every_limit(data_root, tmp_path):
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        assert client.get("/v1/capabilities").json()["limits"] == {
            "page_size": 100,
            "max_document_bytes": 1048576,
            "max_resolved_nodes": 1000,
            "activity_page_size": 100,
        }


def test_a_body_over_the_limit_is_refused_before_it_is_read(data_root, tmp_path, monkeypatch):
    monkeypatch.setattr(_contract, "MAX_DOCUMENT_BYTES", 64)
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        response = client.post("/v1/validate", json={"pipeline": {}, "padding": "x" * 100})
        assert response.status_code == 413
        assert response.json() == {
            "schema": 1,
            "code": "document_too_large",
            "message": "The request body is over 64 bytes.",
            "retryable": False,
            "details": {"limit": 64},
        }
        assert client.get("/v1/capabilities").status_code == 200  # a GET carries no body to bound
