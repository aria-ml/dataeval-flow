"""A real pipeline, submitted over HTTP, runs through the batch command and leaves its results with the run."""

import pytest

pytest.importorskip("fastapi", reason="needs the 'service' extra")

from fastapi.testclient import TestClient

from dataeval_flow._service._app import create_app

pytestmark = pytest.mark.optional


def test_quality_and_triage_run_end_to_end(data_root, tmp_path, pipeline, wait_for):
    app = create_app(data_root, tmp_path / "out")
    with TestClient(app) as client:
        run_id = client.post("/v1/runs", json={"pipeline": pipeline}).json()["id"]
        record = wait_for(app.state.store, run_id, {"succeeded", "failed"}, timeout=300)
        logs = client.get(f"/v1/runs/{run_id}/logs").text
        assert record["status"] == "succeeded", logs
        assert record["exit_code"] == 0
        results = client.get(f"/v1/runs/{run_id}/results").json()
        assert set(results) == {"quality", "metadata"}
        # The fixture's duplicates and rare class reach the structured findings.
        assert "duplicate" in str(results["quality"]).lower()
        artifacts = client.get(f"/v1/runs/{run_id}/artifacts").json()
        assert {"results/result.html", "results/result.txt", "result.log", "console.log"} <= set(artifacts)
        assert "Task: quality" in logs
