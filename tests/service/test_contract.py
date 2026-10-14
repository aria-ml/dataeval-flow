"""The builder contracts' shared behavior: JSON 404s, the extension error, limits and request bounds (builder spec §4,
§7)."""

import re
import sys

import pytest

pytest.importorskip("fastapi", reason="needs the 'service' extra")

from fastapi.testclient import TestClient

from dataeval_flow._service import _contract
from dataeval_flow._service._app import create_app
from dataeval_flow._service._contract import fingerprint_of, schema_fingerprint
from dataeval_flow.config import PipelineConfig
from dataeval_flow.steps import list_steps

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


_SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]
_STALE = "sha256:" + "0" * 64


def test_the_fingerprint_hashes_canonical_json_in_array_order() -> None:
    assert fingerprint_of({"b": 1, "a": [1, 2]}) == fingerprint_of({"a": [1, 2], "b": 1})
    assert fingerprint_of({"a": [1, 2]}) != fingerprint_of({"a": [2, 1]})
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", fingerprint_of({}))


def test_the_schema_fingerprint_covers_the_schema_and_every_installed_step(plugins) -> None:
    from tests.chain_toys import register_toys

    register_toys(plugins)
    schema_fingerprint.cache_clear()
    try:
        catalog = list_steps().model_dump(mode="json")
        assert any(entry["type"] == "toy-keep" for entry in catalog["steps"])
        assert schema_fingerprint() == fingerprint_of(
            {"schema": PipelineConfig.model_json_schema(), "catalog": catalog}
        )
    finally:
        schema_fingerprint.cache_clear()


def test_a_stale_pin_is_refused_and_queues_nothing(data_root, tmp_path, pipeline):
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        for route in ("/v1/validate", "/v1/runs"):
            response = client.post(route, json={"pipeline": pipeline, "expected_schema_fingerprint": _STALE})
            assert response.status_code == 409
            body = response.json()
            assert (body["code"], body["retryable"]) == ("schema_changed", False)
            assert body["details"] == {"expected": _STALE, "actual": schema_fingerprint()}
        assert client.get("/v1/runs").json() == []


def test_a_matching_pin_is_accepted_and_every_run_records_the_fingerprint(data_root, tmp_path, pipeline):
    app = create_app(data_root, tmp_path / "out")
    app.state.manager.command = lambda _: _SLEEP
    with TestClient(app) as client:
        capabilities = client.get("/v1/capabilities").json()
        fingerprint = capabilities["schema_fingerprint"]
        assert capabilities["features"]["builder_metadata"] == 1
        pinned = {"pipeline": pipeline, "expected_schema_fingerprint": fingerprint}
        assert client.post("/v1/validate", json=pinned).json()["schema_fingerprint"] == fingerprint
        run = client.post("/v1/runs", json={"pipeline": pipeline}).json()
        assert run["schema_fingerprint"] == fingerprint
        client.post(f"/v1/runs/{run['id']}/cancel")


def test_a_malformed_fingerprint_is_a_422_not_a_mismatch(data_root, tmp_path, pipeline):
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        response = client.post("/v1/validate", json={"pipeline": pipeline, "expected_schema_fingerprint": "abc"})
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "expected_schema_fingerprint"]


def test_a_pipeline_that_is_not_an_object_is_a_located_422(data_root, tmp_path):
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        for pipeline in ([], "params.yaml", None):
            response = client.post("/v1/validate", json={"pipeline": pipeline})
            assert response.status_code == 422
            assert response.json()["detail"][0]["loc"][:2] == ["body", "pipeline"]


def test_a_malformed_pipeline_is_located_as_fastapi_located_it(data_root, tmp_path, pipeline):
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        broken = {**pipeline, "tasks": [{"name": "quality", "workflow": "quality", "sources": "data", "x": 1}]}
        response = client.post("/v1/validate", json={"pipeline": broken})
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "pipeline", "tasks", 0, "x"]
