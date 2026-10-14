"""`POST /v1/resolve`, and the located graph 422s `/validate` and `/runs` now give (builder spec §5)."""

import copy

import pytest

pytest.importorskip("fastapi", reason="needs the 'service' extra")

from fastapi.testclient import TestClient

from dataeval_flow._service import _contract
from dataeval_flow._service._app import create_app
from tests.service.builder_fixtures import fixture_pipeline

pytestmark = pytest.mark.optional


def _unreadable() -> dict:
    """The fixture pipeline over a dataset that isn't there: resolving it must not try to read it."""
    pipeline = fixture_pipeline()
    pipeline["datasets"][0]["path"] = "not-there"
    return pipeline


def _broken() -> dict:
    pipeline = _unreadable()
    pipeline["workflows"][0]["steps"][1]["input"] = "missing"
    return pipeline


@pytest.fixture
def client(data_root, tmp_path):
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        yield client


def test_a_pipeline_that_connects_resolves_without_reading_data(client):
    assert client.get("/v1/capabilities").json()["features"]["graph_resolution"] == 1
    response = client.post("/v1/resolve", json={"pipeline": _unreadable()})
    assert response.status_code == 200
    body = response.json()
    assert (body["schema"], body["valid"], body["diagnostics"], body["workflows"]) == (1, True, [], [])
    assert body["schema_fingerprint"] == _contract.schema_fingerprint()
    assert body["tasks"][0]["execution_order"] == ["review:dupes", "review:judge"]
    assert client.get("/v1/runs").json() == []


def test_a_pipeline_that_does_not_connect_resolves_as_far_as_it_goes(client):
    body = client.post("/v1/resolve", json={"pipeline": _broken()}).json()
    assert body["valid"] is False
    (problem,) = body["diagnostics"]
    assert (problem["code"], problem["document_pointer"], problem["node_id"], problem["port"]) == (
        "unknown_address",
        "/workflows/0/steps/1/input",
        "review:judge",
        "input",
    )
    assert (body["tasks"][0]["complete"], body["tasks"][0]["execution_order"]) == (False, ["review:dupes"])


def test_validate_and_submit_locate_what_does_not_connect(client):
    for route in ("/v1/validate", "/v1/runs"):
        response = client.post(route, json={"pipeline": _broken()})
        assert response.status_code == 422
        (entry,) = response.json()["detail"]
        assert entry["loc"] == ["body", "pipeline", "workflows", 0, "steps", 1, "input"]
        assert "`missing`" in entry["msg"]
    assert client.get("/v1/runs").json() == []


def test_a_malformed_pipeline_is_a_native_422_on_resolve_too(client):
    pipeline = copy.deepcopy(_unreadable())
    pipeline["tasks"][0]["x"] = 1
    response = client.post("/v1/resolve", json={"pipeline": pipeline})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "pipeline", "tasks", 0, "x"]


def test_a_stale_pin_is_refused_by_resolve(client):
    body = {"pipeline": _unreadable(), "expected_schema_fingerprint": "sha256:" + "0" * 64}
    response = client.post("/v1/resolve", json=body)
    assert (response.status_code, response.json()["code"]) == (409, "schema_changed")


def test_too_many_nodes_is_a_413(client, monkeypatch):
    monkeypatch.setattr(_contract, "MAX_RESOLVED_NODES", 1)
    response = client.post("/v1/resolve", json={"pipeline": _unreadable()})
    assert response.status_code == 413
    body = response.json()
    assert (body["code"], body["details"]) == ("too_many_nodes", {"limit": 1, "count": 2})
