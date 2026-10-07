"""TC-23-1 — Execution service: health, liveness and readiness probes, interface documentation and help."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="needs the 'service' extra")

from fastapi.testclient import TestClient

from dataeval_flow._service._app import create_app

pytestmark = pytest.mark.optional

_PROBES = ("/healthz", "/livez", "/readyz")


@pytest.mark.test_case("23-1")
class TestExecutionService:
    def test_serve_help_documents_its_configuration(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "dataeval_flow", "serve", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        for documented in (
            "DATAEVAL_DATA",
            "DATAEVAL_OUTPUT",
            "DATAEVAL_CACHE",
            "DATAEVAL_SERVICE_HOST",
            "DATAEVAL_SERVICE_PORT",
            "/healthz",
            "/openapi.json",
        ):
            assert documented in result.stdout

    def test_probes_answer_200_while_operational(self, tmp_path: Path) -> None:
        with TestClient(create_app(tmp_path, tmp_path / "out")) as client:
            for probe in _PROBES:
                assert client.get(probe).status_code == 200, probe

    def test_probes_answer_503_once_the_queue_stops(self, tmp_path: Path) -> None:
        app = create_app(tmp_path, tmp_path / "out")
        with TestClient(app) as client:
            manager = app.state.manager
            manager.stopping.set()
            manager.wake.set()
            manager.thread.join()
            manager.stopping.clear()
            for probe in _PROBES:
                assert client.get(probe).status_code == 503, probe

    def test_openapi_describes_the_api(self, tmp_path: Path) -> None:
        with TestClient(create_app(tmp_path, tmp_path / "out")) as client:
            paths = client.get("/openapi.json").json()["paths"]
        assert {*_PROBES, "/v1/runs", "/v1/runs/{run_id}"} <= set(paths)
