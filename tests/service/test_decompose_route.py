"""`POST /v1/presets/decompose`: the copy preparation over HTTP, with the service's pin, size and 422 conventions."""

import pytest

pytest.importorskip("fastapi", reason="needs the 'service' extra")

from fastapi.testclient import TestClient

from dataeval_flow._service import _contract
from dataeval_flow._service._app import create_app
from tests.service.builder_fixtures import fixture_pipeline

pytestmark = pytest.mark.optional


def _with_quality() -> dict:
    pipeline = fixture_pipeline()
    pipeline["workflows"].append(
        {"name": "q", "type": "quality", "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}}
    )
    pipeline["tasks"].append({"name": "clean", "workflow": "q", "sources": "data"})
    return pipeline


def _body(**request) -> dict:
    return {
        "pipeline": _with_quality(),
        "operation": "copy_definition",
        "target_pointer": "/workflows/1",
        "new_name": "qc",
        **request,
    }


@pytest.fixture
def client(data_root, tmp_path):
    with TestClient(create_app(data_root, tmp_path / "out")) as client:
        yield client


def test_the_service_advertises_decomposition(client) -> None:
    assert client.get("/v1/capabilities").json()["features"]["preset_decomposition"] == 1


def test_a_copy_is_prepared_without_queueing_anything(client) -> None:
    response = client.post("/v1/presets/decompose", json=_body(reassign_tasks=["clean"]))
    assert response.status_code == 200
    body = response.json()
    assert (body["schema"], body["status"], body["valid"]) == (1, "ready", True)
    assert body["supported_operations"] == ["copy_definition", "inline_step"]
    assert body["pipeline"]["tasks"][1]["workflow"] == "qc"
    assert client.get("/v1/runs").json() == []


def test_a_step_is_inlined_without_queueing_anything(client) -> None:
    pipeline = _with_quality()
    pipeline["workflows"].append(
        {"name": "outer", "inputs": ["data"], "steps": [{"name": "c", "workflow": "q", "input": "data"}]}
    )
    body = {"pipeline": pipeline, "operation": "inline_step", "target_pointer": "/workflows/2/steps/0"}
    response = client.post("/v1/presets/decompose", json=body)
    assert response.status_code == 200
    answer = response.json()
    assert (answer["status"], answer["supported_operations"]) == ("ready", ["copy_definition", "inline_step"])
    assert answer["names"]["steps"]["outliers"] == "c-outliers"
    assert client.get("/v1/runs").json() == []


def test_a_stale_pin_is_refused_before_preparing(client) -> None:
    response = client.post("/v1/presets/decompose", json=_body(expected_schema_fingerprint="sha256:" + "0" * 64))
    assert response.status_code == 409
    assert response.json()["code"] == "schema_changed"


def test_a_request_naming_no_preset_is_a_located_422(client) -> None:
    response = client.post("/v1/presets/decompose", json=_body(target_pointer="/workflows/0"))
    assert response.status_code == 422
    (problem,) = response.json()["detail"]
    assert problem["loc"] == ["body", "target_pointer"]


def test_a_malformed_draft_is_a_422_located_in_the_pipeline(client) -> None:
    body = _body()
    body["pipeline"]["workflows"][1]["type"] = "no-such-type"
    response = client.post("/v1/presets/decompose", json=body)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][:2] == ["body", "pipeline"]


def test_the_contract_is_in_openapi(client) -> None:
    schema = client.get("/openapi.json").json()
    assert "/v1/presets/decompose" in schema["paths"]
    assert {"DecomposeRequest", "DecomposeReady", "DecomposeUnsupported"} <= set(schema["components"]["schemas"])


def test_the_answer_is_pinned_to_this_service(client) -> None:
    body = client.post("/v1/presets/decompose", json=_body()).json()
    assert body["schema_fingerprint"] == _contract.schema_fingerprint()


def test_openapi_tells_the_answers_apart_by_status_and_requires_what_is_always_sent(client) -> None:
    schema = client.get("/openapi.json").json()
    answer = schema["paths"]["/v1/presets/decompose"]["post"]["responses"]["200"]["content"]["application/json"]
    assert answer["schema"]["discriminator"]["propertyName"] == "status"
    components = schema["components"]["schemas"]
    for name in ("DecomposeReady", "DecomposeUnsupported"):
        assert {"status", "schema", "schema_fingerprint", "supported_operations"} <= set(components[name]["required"])
    assert {"missing_capability", "details"} <= set(components["DecomposeReason"]["required"])
    assert "effective_value" in components["DecomposeStructuralDecision"]["required"]
    assert set(components["DecomposeReason"]["properties"]["code"]["enum"]) == {
        "unsupported_matrix_rewrite",
        "undeclared_structure",
        "unrepresentable_preflight",
    }
    assert set(components["DecomposeRepresentationalChange"]["properties"]["code"]["enum"]) == {
        "result_type",
        "outputs_not_carried",
        "step_names",
        "record_criteria",
    }
    assert "addresses" in components["DecomposeNames"]["required"]
