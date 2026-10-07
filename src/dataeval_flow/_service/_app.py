"""The HTTP API of ``dataeval-flow serve``: queue pipelines, follow their runs, and read what each run wrote."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import dataeval
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from dataeval_flow import __version__
from dataeval_flow._orchestrator import select_tasks
from dataeval_flow._runner import _requirement
from dataeval_flow._service._evidence import PAGE_SIZE, Evidence, UnknownItemError
from dataeval_flow._service._manager import RunManager
from dataeval_flow._service._selections import SelectionError, SelectionRequest, Selections
from dataeval_flow._service._store import RunStore, UnknownRunError
from dataeval_flow.config import PipelineConfig, ResultConfig
from dataeval_flow.steps import list_steps

API_VERSION = 1
_LOG_TAIL = 65536


class RunRequest(BaseModel):
    """A pipeline, and which of its tasks to run: what ``dataeval-flow`` reads from ``--config`` and ``--task``."""

    model_config = ConfigDict(extra="forbid")
    pipeline: PipelineConfig
    tasks: list[str] | None = Field(
        default=None,
        description=(
            "Tasks to run, in order; every enabled task when omitted. Naming a task runs it whether or not the "
            "pipeline marks it enabled."
        ),
    )


def _refused(loc: list[str], message: str) -> HTTPException:
    """A 422 shaped as FastAPI shapes a model's: where in the request it failed, and why."""
    return HTTPException(422, [{"type": "value_error", "loc": loc, "msg": message}])


def _checked(request: RunRequest) -> dict[str, Any]:
    """The tasks a request runs and the snapshot it runs from, or a 422 saying why it would fail before any ran."""
    pipeline = request.pipeline
    try:
        tasks = [task.name for task in select_tasks(pipeline, request.tasks)]
    except ValueError as error:
        where = ["body", "tasks"] if request.tasks is not None else ["body", "pipeline", "tasks"]
        raise _refused(where, str(error)) from None
    try:
        _requirement(pipeline, tasks, None)
    except ValueError as error:
        raise _refused(["body", "pipeline", "result", "require"], str(error)) from None
    if "json" not in pipeline.result.formats:
        message = "The service reads each run's results as JSON: keep `json` in `result: formats`."
        raise _refused(["body", "pipeline", "result", "formats"], message)
    return {"valid": True, "tasks": tasks, "pipeline": pipeline.model_dump(mode="json", by_alias=True)}


def _results(directory: Path, run: dict[str, Any]) -> dict[str, Any]:
    """Every finished task's result, by task name, from the result files the run has written so far."""
    settings = ResultConfig.model_validate(run["pipeline"].get("result") or {})
    tasks = run["request"]["tasks"]
    stems = [f"{settings.name}-{task}" for task in tasks] if settings.per_task else [settings.name]
    results: dict[str, Any] = {}
    for stem in stems:
        path = directory / "results" / f"{stem}.json"
        if path.is_file():
            results.update(json.loads(path.read_text()))
    return results


def _files(directory: Path) -> list[str]:
    """Every file a run's directory holds, as paths relative to it."""
    return sorted(path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file())


def create_app(  # noqa: C901 - one nested route per endpoint
    data_root: Path, output_root: Path, cache_root: Path | None = None
) -> FastAPI:
    """The service: runs under ``output_root/runs``, read-only inputs under ``data_root``, a cache shared by runs."""
    store = RunStore(output_root / "runs")
    manager = RunManager(store, data_root, cache_root or output_root / "cache")
    evidence = Evidence(store, data_root)
    selections = Selections(store)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        manager.start()
        try:
            yield
        finally:
            manager.stop()

    app = FastAPI(
        title="DataEval Flow service",
        version=__version__,
        summary="Queue DataEval Flow pipelines, follow their runs, and read their results.",
        lifespan=lifespan,
    )
    app.state.manager = manager
    app.state.store = store

    @app.exception_handler(UnknownRunError)
    async def unknown(_request: Request, _error: UnknownRunError) -> PlainTextResponse:
        return PlainTextResponse("Unknown run", status_code=404)

    @app.exception_handler(UnknownItemError)
    async def unknown_item(_request: Request, error: UnknownItemError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=404)

    @app.exception_handler(SelectionError)
    async def unanswerable(_request: Request, error: SelectionError) -> JSONResponse:
        if error.status == 422:
            refusal = {"type": "value_error", "loc": error.loc, "msg": str(error)}
            return JSONResponse({"detail": [refusal]}, status_code=422)
        return JSONResponse({"detail": str(error)}, status_code=error.status)

    def probe(reasons: list[str]) -> JSONResponse:
        if reasons:
            return JSONResponse({"status": "unavailable", "reasons": reasons}, status_code=503)
        return JSONResponse({"status": "ok"})

    def unready() -> list[str]:
        reasons = [] if manager.alive else ["queue-stopped"]
        return reasons if store.usable() else [*reasons, "store-unavailable"]

    @app.get("/livez", tags=["probes"], summary="200 while the service can run work; 503 once it needs a restart")
    def livez() -> JSONResponse:
        return probe([] if manager.alive or manager.stopping.is_set() else ["queue-stopped"])

    @app.get("/readyz", tags=["probes"], summary="200 while the service takes and runs work; 503 with reasons if not")
    def readyz() -> JSONResponse:
        return probe(unready())

    @app.get("/healthz", tags=["probes"], summary="200 when the service is healthy and operational; 503 if not")
    def healthz() -> JSONResponse:
        return probe(unready())

    @app.get("/v1/capabilities", tags=["service"])
    def capabilities() -> dict[str, Any]:
        """Versions, how many runs run at once, the features and limits this service has, and every step a pipeline
        can chain."""
        return {
            "api_version": API_VERSION,
            "flow_version": __version__,
            "dataeval_version": dataeval.__version__,
            "max_active_runs": 1,
            "features": {"items": 1, "profiles": 1, "selections": 1, "schema": 1},
            "limits": {"page_size": PAGE_SIZE},
            "steps": list_steps().model_dump(mode="json"),
        }

    @app.get("/v1/schema", tags=["service"])
    def schema() -> dict[str, Any]:
        """The JSON Schema of a pipeline, every registered step included: what `pipeline` in a run request takes."""
        return PipelineConfig.model_json_schema()

    @app.post("/v1/validate", tags=["runs"])
    def validate(request: RunRequest) -> dict[str, Any]:
        """Check a pipeline and resolve the tasks it would run, without queueing it."""
        return _checked(request)

    @app.post("/v1/runs", status_code=202, tags=["runs"])
    def submit(request: RunRequest) -> dict[str, Any]:
        """Check a pipeline and queue a snapshot of it; the run continues whoever is listening."""
        checked = _checked(request)
        return manager.submit({"tasks": checked["tasks"]}, checked["pipeline"])

    @app.get("/v1/runs", tags=["runs"])
    def runs() -> list[dict[str, Any]]:
        """Every run, newest first."""
        return store.history()

    @app.get("/v1/runs/{run_id}", tags=["runs"])
    def run(run_id: str) -> dict[str, Any]:
        """A run's status, exit code and the snapshot it ran from."""
        return store.get(run_id)

    @app.post("/v1/runs/{run_id}/cancel", status_code=202, tags=["runs"])
    def cancel(run_id: str) -> dict[str, Any]:
        """Cancel a queued run, or stop a running one; 409 once it has finished."""
        try:
            return manager.cancel(run_id)
        except ValueError as error:
            raise HTTPException(409, str(error)) from None

    @app.get("/v1/runs/{run_id}/results", tags=["runs"])
    def results(run_id: str) -> dict[str, Any]:
        """Each finished task's result, keyed by task, as ``result.json`` holds it; 409 before any task finishes."""
        found = _results(store.directory(run_id), store.get(run_id))
        if not found:
            raise HTTPException(409, "No task of this run has written a result yet")
        return found

    @app.get("/v1/runs/{run_id}/logs", tags=["runs"], response_class=PlainTextResponse)
    def logs(run_id: str) -> str:
        """The last 64 KiB of what the run printed."""
        path = store.directory(run_id) / "console.log"
        if not path.exists():
            return ""
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - _LOG_TAIL))
            return stream.read().decode(errors="replace")

    @app.get("/v1/runs/{run_id}/artifacts", tags=["runs"])
    def artifacts(run_id: str) -> list[str]:
        """Every file the run's directory holds: its snapshot, logs, result files, manifests and exports."""
        return _files(store.directory(run_id))

    @app.get("/v1/runs/{run_id}/artifacts/{name:path}", tags=["runs"])
    def artifact(run_id: str, name: str) -> FileResponse:
        """One of the run's files, by a path ``artifacts`` lists."""
        directory = store.directory(run_id)
        if name not in _files(directory):
            raise HTTPException(404, "Unknown artifact")
        return FileResponse(directory / name)

    @app.get("/v1/runs/{run_id}/items/{source}", tags=["evidence"])
    def items(
        run_id: str,
        source: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(24, ge=1, le=PAGE_SIZE),
    ) -> dict[str, Any]:
        """A page of a source's items in the order the run read them, each checked against the run's manifest."""
        return evidence.items(run_id, source, offset, limit)

    @app.get("/v1/runs/{run_id}/items/{source}/{index}", tags=["evidence"])
    def item(run_id: str, source: str, index: int) -> dict[str, Any]:
        """One item the run read, with its boxes or label and metadata while it matches the run's manifest; else its
        status (`input_changed`, `input_unavailable` or `evidence_unavailable`) and why."""
        return evidence.item(run_id, source, index)

    @app.get(
        "/v1/runs/{run_id}/items/{source}/{index}/image",
        tags=["evidence"],
        response_class=Response,
        responses={200: {"content": {"image/png": {}}}, 409: {"description": "The item is not verified"}},
    )
    def image(
        run_id: str,
        source: str,
        index: int,
        target: int | None = Query(None, ge=0, description="A box to crop, by its index in the item's annotation"),
        max_side: int | None = Query(None, ge=16, le=8192, description="Shrink to fit this many pixels across"),
    ) -> Response:
        """The item's image as a PNG, or one box cropped from it; 409 unless the item matches the run's manifest."""
        try:
            picture = evidence.image(run_id, source, index, target, max_side)
        except LookupError as error:  # no such source, item or box
            raise HTTPException(404, str(error)) from None
        except (TypeError, ValueError) as error:  # not an image, or a box with no area inside it
            raise _refused(["query", "target"] if target is not None else ["path", "index"], str(error)) from None
        if picture is None:
            return JSONResponse(evidence.item(run_id, source, index), status_code=409)
        return Response(picture, media_type="image/png")

    @app.post("/v1/runs/{run_id}/selections", tags=["selections"])
    def select(run_id: str, request: SelectionRequest) -> dict[str, Any]:
        """Resolve a selection over what the run wrote: its id, definition, scope, and how many rows, images and boxes
        it holds. The same request always gives the same id."""
        return selections.create(run_id, request)

    @app.get("/v1/runs/{run_id}/selections/{selection_id}", tags=["selections"])
    def selection(
        run_id: str,
        selection_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(24, ge=1, le=PAGE_SIZE),
    ) -> dict[str, Any]:
        """A selection's summary and a page of its members, each naming its source, item and box."""
        return selections.page(run_id, selection_id, offset, limit)

    @app.get("/v1/runs/{run_id}/selections/{selection_id}/view", tags=["selections"])
    def selection_view(run_id: str, selection_id: str, parents: bool = False) -> dict[str, Any]:
        """The selection's images as a `views:` entry and a `sources:` entry to merge into the run's pipeline; with
        `parents`, the images holding a selection of boxes."""
        return selections.view(run_id, selection_id, parents=parents)

    return app


def serve(data_root: Path, output_root: Path, cache_root: Path | None, host: str, port: int, log_format: str) -> None:
    """Serve until interrupted, logging requests and every run's output to the console."""
    from dataeval_flow._logging import setup_logging

    setup_logging(verbosity=2, log_format=log_format)
    logging.captureWarnings(True)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    uvicorn.run(create_app(data_root, output_root, cache_root), host=host, port=port, log_config=None)
