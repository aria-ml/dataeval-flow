"""TC-21-1 — Analysis service interface: probes, interface description, capabilities, schema and request validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dataeval_flow._service._app import create_app
from dataeval_flow.config import PipelineConfig

pytestmark = pytest.mark.required

_PROBES = ("/healthz", "/livez", "/readyz")


def _stop_queue(app: Any) -> None:
    """Stop the queue consumer as a crash would, leaving the service up."""
    manager = app.state.manager
    manager.stopping.set()
    manager.wake.set()
    manager.thread.join()
    manager.stopping.clear()


class TestProbes:
    def test_every_probe_answers_200_ok_while_the_service_runs_work(self, data_root: Path, tmp_path: Path) -> None:
        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            for probe in _PROBES:
                response = client.get(probe)
                assert (response.status_code, response.json()) == (200, {"status": "ok"}), probe

    def test_a_stopped_queue_fails_every_probe_and_names_the_reason(self, data_root: Path, tmp_path: Path) -> None:
        app = create_app(data_root, tmp_path / "out")
        with TestClient(app) as client:
            _stop_queue(app)
            for probe in _PROBES:
                response = client.get(probe)
                assert response.status_code == 503, probe
                assert response.json() == {"status": "unavailable", "reasons": ["queue-stopped"]}, probe

    def test_an_unusable_run_store_fails_readiness_and_health_but_not_liveness(
        self, data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = create_app(data_root, tmp_path / "out")
        with TestClient(app) as client:
            monkeypatch.setattr(app.state.store, "usable", lambda: False)
            assert client.get("/livez").status_code == 200
            for probe in ("/readyz", "/healthz"):
                response = client.get(probe)
                assert response.status_code == 503, probe
                assert response.json()["reasons"] == ["store-unavailable"], probe


class TestInterfaceDescription:
    def test_openapi_json_describes_every_route(self, data_root: Path, tmp_path: Path) -> None:
        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            response = client.get("/openapi.json")
        assert response.status_code == 200
        paths = response.json()["paths"]
        expected = {
            *_PROBES,
            "/v1/capabilities",
            "/v1/schema",
            "/v1/validate",
            "/v1/runs",
            "/v1/runs/{run_id}",
            "/v1/runs/{run_id}/cancel",
            "/v1/runs/{run_id}/results",
            "/v1/runs/{run_id}/logs",
            "/v1/runs/{run_id}/artifacts",
            "/v1/runs/{run_id}/artifacts/{name}",
            "/v1/runs/{run_id}/items/{source}",
            "/v1/runs/{run_id}/items/{source}/{index}",
            "/v1/runs/{run_id}/items/{source}/{index}/image",
            "/v1/runs/{run_id}/selections",
            "/v1/runs/{run_id}/selections/{selection_id}",
            "/v1/runs/{run_id}/selections/{selection_id}/view",
        }
        assert expected <= set(paths), expected - set(paths)

    def test_every_operation_has_a_unique_id(self, data_root: Path, tmp_path: Path) -> None:
        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            paths = client.get("/openapi.json").json()["paths"]
        operations = [operation["operationId"] for path in paths.values() for operation in path.values()]
        assert len(operations) == len(set(operations))

    def test_capabilities_give_versions_features_limits_and_every_step(self, data_root: Path, tmp_path: Path) -> None:
        import dataeval

        from dataeval_flow import __version__

        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            capabilities = client.get("/v1/capabilities").json()
        assert capabilities["api_version"] == 1
        assert capabilities["flow_version"] == __version__
        assert capabilities["dataeval_version"] == dataeval.__version__
        assert capabilities["max_active_runs"] == 1
        assert capabilities["features"] == {"items": 1, "profiles": 1, "selections": 1, "schema": 1}
        assert capabilities["limits"] == {"page_size": 100}
        assert capabilities["steps"]["format"] == 1

    def test_schema_is_the_json_schema_of_a_pipeline(self, data_root: Path, tmp_path: Path) -> None:
        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            response = client.get("/v1/schema")
        assert response.status_code == 200
        assert response.json() == PipelineConfig.model_json_schema()


class TestRequestValidation:
    @staticmethod
    def _refusal(client: TestClient, body: dict[str, Any], route: str = "/v1/validate") -> list[dict[str, Any]]:
        response = client.post(route, json=body)
        assert response.status_code == 422, response.text
        return response.json()["detail"]

    def test_a_valid_request_is_resolved_without_queueing_anything(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any]
    ) -> None:
        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            checked = client.post("/v1/validate", json={"pipeline": pipeline}).json()
            assert client.get("/v1/runs").json() == []
        assert checked["valid"] is True
        assert checked["tasks"] == ["digest", "labels"]
        # The snapshot holds every setting, defaults included, and reads back as the same pipeline.
        assert PipelineConfig.model_validate(checked["pipeline"]) == PipelineConfig.model_validate(pipeline)

    def test_a_named_task_is_resolved_in_the_order_given(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any]
    ) -> None:
        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            checked = client.post("/v1/validate", json={"pipeline": pipeline, "tasks": ["labels", "digest"]}).json()
        assert checked["tasks"] == ["labels", "digest"]

    def test_a_task_the_pipeline_does_not_define_is_refused_at_body_tasks(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any]
    ) -> None:
        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            (error,) = self._refusal(client, {"pipeline": pipeline, "tasks": ["missing"]})
        assert error["loc"] == ["body", "tasks"]
        assert {"type", "loc", "msg"} <= set(error)

    def test_a_pipeline_whose_every_task_is_disabled_is_refused(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any]
    ) -> None:
        disabled = [{**task, "enabled": False} for task in pipeline["tasks"]]
        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            (error,) = self._refusal(client, {"pipeline": {**pipeline, "tasks": disabled}})
        assert error["loc"] == ["body", "pipeline", "tasks"]
        assert "disabled" in error["msg"]

    def test_results_without_json_are_refused_at_the_formats_field(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any]
    ) -> None:
        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            (error,) = self._refusal(client, {"pipeline": {**pipeline, "result": {"formats": ["html"]}}})
        assert error["loc"] == ["body", "pipeline", "result", "formats"]

    def test_a_required_verdict_no_task_gives_is_refused_at_the_require_field(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any]
    ) -> None:
        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            (error,) = self._refusal(client, {"pipeline": {**pipeline, "result": {"require": "ready"}}})
        assert error["loc"] == ["body", "pipeline", "result", "require"]

    def test_a_field_the_request_does_not_define_is_refused_by_name(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any]
    ) -> None:
        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            (error,) = self._refusal(client, {"pipeline": pipeline, "unexpected": True})
        assert error["loc"] == ["body", "unexpected"]

    def test_a_workflow_type_that_does_not_exist_is_refused(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any]
    ) -> None:
        bad = {**pipeline, "workflows": [{"name": "w", "type": "no-such-preset"}]}
        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            errors = self._refusal(client, {"pipeline": bad})
        assert errors
        assert all(error["loc"][:2] == ["body", "pipeline"] for error in errors)

    def test_submitting_a_refused_request_queues_nothing(
        self, data_root: Path, tmp_path: Path, pipeline: dict[str, Any]
    ) -> None:
        with TestClient(create_app(data_root, tmp_path / "out")) as client:
            self._refusal(client, {"pipeline": pipeline, "tasks": ["missing"]}, route="/v1/runs")
            assert client.get("/v1/runs").json() == []
