"""Schema discovery describes the actual runtime and accepts the submitted pipeline."""

from dataeval_flow.config import PipelineConfig
from fastapi.testclient import TestClient

from dataeval_flow_service._app import create_app


def test_capabilities_and_openapi(staged, tmp_path):
    root, pipeline = staged
    with TestClient(create_app(root, tmp_path / "runs")) as client:
        response = client.get("/v1/capabilities")
        assert response.status_code == 200
        capabilities = response.json()
        assert capabilities["api_version"] == 1 and capabilities["max_active_runs"] == 1
        assert capabilities["device"] == "cpu" and capabilities["steps"]["format"] == 1
        assert "pipeline" in capabilities["run_request"]["properties"]
        validated = client.post("/v1/validate", json={"pipeline": pipeline}).json()
        assert validated["tasks"] == ["quality", "metadata"]
        assert PipelineConfig.model_validate(validated["pipeline"]).tasks[0].name == "quality"
        assert "/v1/runs/{run_id}/cancel" in client.get("/openapi.json").json()["paths"]
