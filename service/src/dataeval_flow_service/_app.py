"""Versioned API that queues Flow pipelines and serves their durable run records."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import dataeval
from dataeval_flow import __version__
from dataeval_flow._orchestrator import select_tasks
from dataeval_flow.config import PipelineConfig
from dataeval_flow.steps import list_steps
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict

from dataeval_flow_service._manager import RunManager
from dataeval_flow_service._store import RunStore


class RunRequest(BaseModel):
    """A pipeline and the tasks to run from it: the batch CLI's config and `--task` options."""

    model_config = ConfigDict(extra="forbid")
    pipeline: PipelineConfig
    tasks: list[str] | None = None


def create_app(data_root: Path, run_root: Path, cache_root: Path | None = None) -> FastAPI:
    """Create a service whose input root is read-only and whose receipts survive restarts."""
    store = RunStore(run_root)
    manager = RunManager(store, data_root, cache_root or run_root / "cache")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        manager.start()
        try:
            yield
        finally:
            manager.stop()

    app = FastAPI(title="DataEval Flow execution service", version="0.1.0", lifespan=lifespan)
    app.state.manager = manager
    app.state.store = store

    @app.exception_handler(KeyError)
    async def missing(request: Any, error: KeyError) -> PlainTextResponse:
        return PlainTextResponse("Unknown run", status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request: Any, error: ValueError) -> PlainTextResponse:
        return PlainTextResponse(str(error), status_code=422)

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "api_version": 1}

    @app.get("/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "api_version": 1,
            "flow_version": __version__,
            "dataeval_version": dataeval.__version__,
            "device": "cpu",
            "max_active_runs": 1,
            "run_request": RunRequest.model_json_schema(),
            "steps": list_steps().model_dump(mode="json"),
            "pipeline_schema": PipelineConfig.model_json_schema(),
        }

    def validate(request: RunRequest) -> dict[str, Any]:
        tasks = [task.name for task in select_tasks(request.pipeline, request.tasks)]
        return {"valid": True, "tasks": tasks, "pipeline": request.pipeline.model_dump(mode="json", exclude_none=True)}

    @app.post("/v1/validate")
    def validate_run(request: RunRequest) -> dict[str, Any]:
        return validate(request)

    @app.post("/v1/runs", status_code=202)
    def submit(request: RunRequest) -> dict[str, Any]:
        checked = validate(request)
        return manager.submit({"tasks": checked["tasks"]}, checked["pipeline"])

    @app.get("/v1/runs")
    def runs() -> list[dict[str, Any]]:
        return store.list()

    @app.get("/v1/runs/{run_id}")
    def run(run_id: str) -> dict[str, Any]:
        return store.get(run_id)

    @app.post("/v1/runs/{run_id}/cancel", status_code=202)
    def cancel(run_id: str) -> dict[str, Any]:
        try:
            return manager.cancel(run_id)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/v1/runs/{run_id}/results")
    def results(run_id: str) -> dict[str, Any]:
        path = store.directory(run_id) / "results.json"
        if not path.exists():
            raise HTTPException(409, "No results are available yet")
        return json.loads(path.read_text())

    @app.get("/v1/runs/{run_id}/events")
    def events(run_id: str) -> list[dict[str, Any]]:
        path = store.directory(run_id) / "events.jsonl"
        rows = []
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # The worker may be in the middle of appending its final line.
        return rows

    @app.get("/v1/runs/{run_id}/logs", response_class=PlainTextResponse)
    def logs(run_id: str) -> str:
        path = store.directory(run_id) / "worker.log"
        if not path.exists():
            return ""
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 65536))
            return stream.read().decode(errors="replace")

    @app.get("/v1/runs/{run_id}/artifacts")
    def artifacts(run_id: str) -> list[str]:
        return sorted(path.name for path in store.directory(run_id).iterdir() if path.suffix in {".html", ".json"})

    @app.get("/v1/runs/{run_id}/artifacts/{name}")
    def artifact(run_id: str, name: str) -> FileResponse:
        directory = store.directory(run_id)
        allowed = {path.name for path in directory.iterdir() if path.suffix in {".html", ".json"}}
        if name not in allowed:
            raise HTTPException(404, "Unknown artifact")
        return FileResponse(directory / name)

    return app
