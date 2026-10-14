"""The HTTP API of ``dataeval-flow serve``: queue pipelines, follow their runs, and read what each run wrote."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import dataeval
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dataeval_flow import __version__
from dataeval_flow._activity import ACTIVITY_FILE, page, read_events, snapshot
from dataeval_flow._orchestrator import select_tasks
from dataeval_flow._runner import _requirement
from dataeval_flow._service import _contract
from dataeval_flow._service._evidence import PAGE_SIZE, Evidence, UnknownItemError
from dataeval_flow._service._manager import RunManager
from dataeval_flow._service._resolve import Resolution, resolve
from dataeval_flow._service._selections import SelectionError, SelectionRequest, Selections
from dataeval_flow._service._store import TERMINAL, RunStore, UnknownRunError
from dataeval_flow.config import PipelineConfig, ResultConfig, TaskConfig
from dataeval_flow.steps import list_steps

API_VERSION = 1
_LOG_TAIL = 65536


_FINGERPRINT = r"^sha256:[0-9a-f]{64}$"


class RunRequest(BaseModel):
    """A pipeline, and which of its tasks to run: what ``dataeval-flow`` reads from ``--config`` and ``--task``."""

    model_config = ConfigDict(extra="forbid")
    pipeline: dict[str, Any] = Field(
        description="The pipeline configuration, as JSON: what `--config` holds. `GET /v1/schema` describes it."
    )
    tasks: list[str] | None = Field(
        default=None,
        description=(
            "Tasks to run, in order; every enabled task when omitted. Naming a task runs it whether or not the "
            "pipeline marks it enabled."
        ),
    )
    expected_schema_fingerprint: str | None = Field(
        default=None,
        pattern=_FINGERPRINT,
        description=(
            "The `schema_fingerprint` the client read the schema and catalog under. When given and the service's "
            "differs, the request is refused with `409 schema_changed`, before the pipeline is read."
        ),
    )


def _refused(loc: list[str], message: str) -> HTTPException:
    """A 422 shaped as FastAPI shapes a model's: where in the request it failed, and why."""
    return HTTPException(422, [{"type": "value_error", "loc": loc, "msg": message}])


def _pinned(request: RunRequest) -> None:
    """Refuse a request pinned to another schema than this service's, before reading its pipeline under this one."""
    actual = _contract.schema_fingerprint()
    expected = request.expected_schema_fingerprint
    if expected is not None and expected != actual:
        raise _contract.ContractError(
            409,
            "schema_changed",
            "The service's pipeline schema changed since the client read it: read /v1/capabilities again.",
            expected=expected,
            actual=actual,
        )


def _parsed(request: RunRequest, *, graphs: bool = True) -> PipelineConfig:
    """`request`'s pipeline, checked as a config file is; a 422 locates what it refuses as FastAPI does.

    Without `graphs`, whether its workflows connect is left for the caller to check and locate (builder spec §5.1).
    """
    try:
        return PipelineConfig.model_validate(request.pipeline, context=None if graphs else {"graphs": False})
    except ValidationError as error:
        problems = [
            {**problem, "loc": ("body", "pipeline", *problem["loc"])} for problem in error.errors(include_url=False)
        ]
        raise RequestValidationError(problems) from None


def _selected(request: RunRequest, pipeline: PipelineConfig) -> list[TaskConfig]:
    """The tasks a request runs, or a 422 saying why it would fail before any ran."""
    try:
        tasks = select_tasks(pipeline, request.tasks)
    except ValueError as error:
        where = ["body", "tasks"] if request.tasks is not None else ["body", "pipeline", "tasks"]
        raise _refused(where, str(error)) from None
    try:
        _requirement(pipeline, [task.name for task in tasks], None)
    except ValueError as error:
        raise _refused(["body", "pipeline", "result", "require"], str(error)) from None
    if "json" not in pipeline.result.formats:
        message = "The service reads each run's results as JSON: keep `json` in `result: formats`."
        raise _refused(["body", "pipeline", "result", "formats"], message)
    return tasks


def _resolved(request: RunRequest) -> tuple[PipelineConfig, list[TaskConfig], Resolution]:
    """A request's pipeline, its tasks, and its graphs resolved: what `/validate`, `/resolve` and `/runs` share
    (builder spec §5.1). Its graphs are checked here rather than as it loads, so each problem is located."""
    _pinned(request)
    pipeline = _parsed(request, graphs=False)
    tasks = _selected(request, pipeline)
    return pipeline, tasks, resolve(pipeline, tasks)


def _located(diagnostic: dict[str, Any]) -> dict[str, Any]:
    """A diagnostic as one entry of a 422, located at its pointer into the pipeline."""
    parts = (diagnostic["document_pointer"] or "").split("/")[1:]
    loc = ["body", "pipeline", *(int(part) if part.isdigit() else part for part in parts)]
    return {"type": "value_error", "loc": loc, "msg": diagnostic["message"]}


def _checked(request: RunRequest) -> dict[str, Any]:
    """The tasks a request runs and the snapshot it runs from, or a 422 saying why it would fail before any ran."""
    pipeline, tasks, resolution = _resolved(request)
    if resolution.diagnostics:
        raise HTTPException(422, [_located(diagnostic) for diagnostic in resolution.diagnostics])
    return {
        "valid": True,
        "tasks": [task.name for task in tasks],
        "pipeline": pipeline.model_dump(mode="json", by_alias=True),
        "schema_fingerprint": _contract.schema_fingerprint(),
    }


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
    async def unknown(_request: Request, _error: UnknownRunError) -> JSONResponse:
        return JSONResponse({"detail": "Unknown run"}, status_code=404)

    @app.exception_handler(_contract.ContractError)
    async def refused(_request: Request, error: _contract.ContractError) -> JSONResponse:
        return JSONResponse(error.body, status_code=error.status)

    @app.middleware("http")
    async def bounded(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # ponytail: trusts Content-Length, so a chunked body isn't bounded; count the stream if a client sends one.
        length = request.headers.get("content-length", "")
        limit = _contract.MAX_DOCUMENT_BYTES
        if length.isdigit() and int(length) > limit:
            refusal = _contract.ContractError(
                413, "document_too_large", f"The request body is over {limit} bytes.", limit=limit
            )
            return JSONResponse(refusal.body, status_code=413)
        return await call_next(request)

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
            "schema_fingerprint": _contract.schema_fingerprint(),
            "max_active_runs": 1,
            "features": {
                "items": 1,
                "profiles": 1,
                "selections": 1,
                "schema": 1,
                "builder_metadata": 1,
                "graph_resolution": 1,
                "step_activity": 1,
            },
            "limits": {
                "page_size": PAGE_SIZE,
                "max_document_bytes": _contract.MAX_DOCUMENT_BYTES,
                "max_resolved_nodes": _contract.MAX_RESOLVED_NODES,
                "activity_page_size": _contract.ACTIVITY_PAGE_SIZE,
            },
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

    @app.post("/v1/resolve", tags=["builder"])
    def resolve_graphs(request: RunRequest) -> dict[str, Any]:
        """Resolve each task's graph without reading data or queueing: its steps, ports and bindings, and where it does
        not connect (builder spec §5)."""
        pipeline, _, resolution = _resolved(request)
        limit = _contract.MAX_RESOLVED_NODES
        if resolution.node_count > limit:
            raise _contract.ContractError(
                413,
                "too_many_nodes",
                f"The pipeline resolves to {resolution.node_count} nodes; the service resolves at most {limit}.",
                limit=limit,
                count=resolution.node_count,
            )
        return {
            "schema": _contract.CONTRACT_SCHEMA,
            "schema_fingerprint": _contract.schema_fingerprint(),
            "valid": not resolution.diagnostics,
            "pipeline": pipeline.model_dump(mode="json", by_alias=True),
            "tasks": resolution.tasks,
            "workflows": resolution.workflows,
            "diagnostics": resolution.diagnostics,
        }

    @app.post("/v1/runs", status_code=202, tags=["runs"])
    def submit(request: RunRequest) -> dict[str, Any]:
        """Check a pipeline and queue a snapshot of it; the run continues whoever is listening."""
        checked = _checked(request)
        return manager.submit({"tasks": checked["tasks"]}, checked["pipeline"], checked["schema_fingerprint"])

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
        record = store.get(run_id)
        found = _results(store.directory(run_id), record)
        if not found:
            finished = record["status"] in TERMINAL
            raise HTTPException(
                409, "The run finished without a result" if finished else "No task of this run has written a result yet"
            )
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

    def recorded(run_id: str) -> list[dict[str, Any]]:
        path = store.directory(run_id) / ACTIVITY_FILE
        if not path.exists():
            raise _contract.ContractError(
                409,
                "activity_unavailable",
                "The run was queued before the service recorded activity; "
                "its results, logs and artifacts stay readable.",
            )
        return read_events(path)

    @app.get("/v1/runs/{run_id}/activity", tags=["builder"])
    def activity(run_id: str) -> dict[str, Any]:
        """Each task's and step's state, consistent at the `sequence` of the last event read (builder spec §6.5)."""
        record = store.get(run_id)
        return snapshot(recorded(run_id), run_id, record.get("schema_fingerprint"))

    @app.get("/v1/runs/{run_id}/events", tags=["builder"])
    def events(
        run_id: str,
        after: int = Query(0, ge=0, description="The last sequence the client applied: a snapshot's `sequence`"),
        limit: int = Query(_contract.ACTIVITY_PAGE_SIZE, ge=1, le=_contract.ACTIVITY_PAGE_SIZE),
    ) -> dict[str, Any]:
        """The events after `after`, in order, each carrying the full row it changed (builder spec §6.5)."""
        found = recorded(run_id)
        if after > len(found):
            raise _refused(["query", "after"], f"The run's last event is {len(found)}; `after` is {after}.")
        return page(found, run_id, after, limit)

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
