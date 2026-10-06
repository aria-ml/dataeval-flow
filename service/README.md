# Flow companion execution service

This independently installed package runs Flow pipelines asynchronously behind a versioned HTTP API. Flow's existing library, CLI, schemas, workflows, and report generation are used without changes.

```text
Client (e.g. Studio) → /v1 API → Flow companion service
                                     ↓ one subprocess per run
                                 Flow → DataEval

Read-only: data root (datasets, models)
Writable:  run receipts, SQLite history, structured results, reports, cache
```

## Execution contract

`POST /v1/runs` validates a pipeline and its task selection and returns `202` with a durable run ID. The service snapshots the pipeline and the resolved task list before queueing. A single queue consumer starts an isolated process group for each run. Runs continue without an open browser. Cancelling queued work prevents execution; cancelling active work terminates its group, escalating after three seconds. A file lock prevents two service instances from executing the same queue.

States are `queued`, `running`, `cancelling`, and terminal `succeeded`, `failed`, `cancelled`, or `interrupted`. Service shutdown interrupts active work and preserves queued work. Startup marks lost active executions interrupted and starts queued work. An orphan watchdog terminates a worker process group if its service parent disappears. Success requires both a zero process exit and a successful Flow outcome receipt; warnings remain findings, rather than execution failures.

Each completed task writes structured results atomically before its HTML report. Earlier results remain available if a later task fails. Logs and events are retained alongside request/pipeline JSON, result JSON, and HTML artifacts. Flow's result metadata records runtime/library versions, resolved configuration, data lineage, and diagnostics. SQLite connections are short lived, and terminal records cannot be overwritten.

## API surface

| Endpoint | Behavior |
|---|---|
| `GET /v1/health` | Liveness and API version |
| `GET /v1/capabilities` | Runtime versions, component schemas, pipeline schema, run request schema |
| `POST /v1/validate` | Validate a pipeline and resolve which of its tasks would run |
| `POST /v1/runs` | Validate again and durably queue a snapshot |
| `GET /v1/runs` | Persistent history |
| `GET /v1/runs/{id}` | Status and immutable input snapshot |
| `POST /v1/runs/{id}/cancel` | Request cancellation; terminal runs return `409` |
| `GET /v1/runs/{id}/results` | Structured partial/final task results; `409` before first result |
| `GET /v1/runs/{id}/events` | Task-boundary events and errors |
| `GET /v1/runs/{id}/logs` | Last 64 KiB of worker logs |
| `GET /v1/runs/{id}/artifacts` | Available JSON/HTML artifact names; append `/{name}` to retrieve |

A request carries the pipeline configuration the batch CLI reads and, optionally, the tasks to run, in order. Omitting `tasks` runs every enabled task, as the batch CLI does:

```json
{
  "pipeline": {
    "datasets": [{"name": "boats", "format": "coco", "path": "data/boats"}],
    "sources": [{"name": "data", "dataset": "boats"}],
    "workflows": [{"name": "metadata", "type": "triage"}],
    "tasks": [{"name": "metadata", "workflow": "metadata", "sources": "data"}]
  },
  "tasks": ["metadata"]
}
```

Dataset and model paths resolve against the configured data root.

The result envelope is `{"format": 1, "tasks": {"quality": {"success": true, "result": ...}}}`. `result` is Flow's structured result, including findings, blocks, steps, errors, coverage gaps, metadata, and assets. Clients consume this JSON directly; HTML is an optional presentation artifact.

## Scope

One CPU worker runs at a time. Authentication, remote storage resolution, package publication, process resume, and container/offline bundle validation are outside this first slice. Dataset staging and catalogs belong to clients.
