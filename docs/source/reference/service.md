# Service Reference

Reference for `dataeval-flow serve`, the long-running HTTP service, and version 1 of its API. To set the service up
and use it, see {doc}`../how_to/run_flow_as_a_service`. For why it runs each pipeline as a batch command, see
{doc}`../concepts/ExecutionModes`.

The service publishes every request and response schema at `/openapi.json`. If this page and a running service's
`/openapi.json` disagree, trust the service.

## Starting the service

`dataeval-flow serve` needs the `service` extra (`pip install "dataeval-flow[service]"`). Every container image ships
it. It serves until `SIGINT` or `SIGTERM`.

| Option         | Environment variable    | Default                                             | Purpose                                         |
| -------------- | ----------------------- | --------------------------------------------------- | ----------------------------------------------- |
| `-d, --data`   | `DATAEVAL_DATA`         | `/dataeval` in the container; the current directory | Data root that pipelines' paths resolve against |
| `-o, --output` | `DATAEVAL_OUTPUT`       | `/output` in the container; **required** otherwise  | Run history and every run's files               |
| `-k, --cache`  | `DATAEVAL_CACHE`        | `/cache` when mounted, else `<output>/cache`        | Computation cache the runs share                |
| `--host`       | `DATAEVAL_SERVICE_HOST` | `0.0.0.0` in the container; `127.0.0.1` otherwise   | Address to listen on                            |
| `--port`       | `DATAEVAL_SERVICE_PORT` | `8001`                                              | Port to listen on                               |

A command-line option overrides its environment variable, which overrides the default. Options follow `serve`.
`--log-format` (`DATAEVAL_LOG_FORMAT`) is the top-level option and precedes it:
`dataeval-flow --log-format plain serve`. The service exits `1` before it serves when `--output` is missing, the data
root does not exist, or the `service` extra is not installed. It exits `3` when another service already holds the
run queue in the same output directory.

The service reads no secrets and has **no authentication**. Anyone who can reach its port can queue, cancel and read
runs. Bind it to a trusted network only.

## Endpoints

| Method and path                                  | Success                        | Errors          | Does                                                               |
| ------------------------------------------------ | ------------------------------ | --------------- | ------------------------------------------------------------------ |
| `GET /healthz`                                   | `200`                          | `503`           | Whether the service takes and runs work                            |
| `GET /readyz`                                    | `200`                          | `503`           | The same, as a readiness probe                                     |
| `GET /livez`                                     | `200`                          | `503`           | Whether the process can run work, as a liveness probe              |
| `GET /openapi.json`                              | `200`                          |                 | The API's OpenAPI description                                      |
| `GET /docs`                                      | `200`                          |                 | That description rendered; its page assets load from a CDN         |
| `GET /v1/capabilities`                           | `200`                          |                 | Versions, features, limits, and every step a pipeline uses         |
| `GET /v1/schema`                                 | `200`                          |                 | The JSON Schema of a pipeline, every registered step in it         |
| `POST /v1/validate`                              | `200` with the checked request | `422`           | Check a run request without queueing it                            |
| `POST /v1/runs`                                  | `202` with the run record      | `422`           | Check a run request and queue it                                   |
| `GET /v1/runs`                                   | `200` with every run record    |                 | Run history, newest first                                          |
| `GET /v1/runs/{id}`                              | `200` with the run record      | `404`           | One run                                                            |
| `POST /v1/runs/{id}/cancel`                      | `202` with the run record      | `404, 409`      | Cancel a queued run, or stop a running one                         |
| `GET /v1/runs/{id}/results`                      | `200` with the results         | `404, 409`      | Each finished task's result                                        |
| `GET /v1/runs/{id}/logs`                         | `200`, plain text              | `404`           | The last 64 KiB of what the run printed                            |
| `GET /v1/runs/{id}/artifacts`                    | `200` with a list of paths     | `404`           | Every file in the run's directory                                  |
| `GET /v1/runs/{id}/artifacts/{path}`             | `200` with the file            | `404`           | One file, by a path the list gives                                 |
| `GET /v1/runs/{id}/items/{source}`               | `200` with a page of items     | `404, 422`      | A source's items as the run read them; see [Items](#items)         |
| `GET /v1/runs/{id}/items/{source}/{index}`       | `200` with the item            | `404`           | One item, checked against the run's manifest                       |
| `GET /v1/runs/{id}/items/{source}/{index}/image` | `200`, PNG                     | `404, 409`      | The item's image, or one box cropped from it                       |
| `POST /v1/runs/{id}/selections`                  | `200` with the summary         | `404, 409, 422` | Select rows of a profile or a table; see [Selections](#selections) |
| `GET /v1/runs/{id}/selections/{sel}`             | `200` with a page of members   | `404, 409`      | A selection's summary and members                                  |
| `GET /v1/runs/{id}/selections/{sel}/view`        | `200` with a source            | `404, 422`      | A selection's images as a source a later run reads                 |

Request and response bodies are JSON, except the logs and the files. Times are Unix epoch seconds.

### Errors

| Status | When                                                                                                                                                       |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `404`  | No run has the ID; the run holds no file at the path; its pipeline names no such source; the source held no such item; no selection has the ID             |
| `409`  | Cancelling a run that has finished; reading results, or selecting from them, before the task has written them; the image of an item that is not `verified` |
| `422`  | The request is malformed, the run would fail before any task ran, or the run cannot answer the selection; the body says why                                |
| `503`  | From a probe only: the service cannot run work; the body names the reasons                                                                                 |

A `422` body is `{"detail": [{"type", "loc", "msg"}, ...]}`: `loc` is where in the request the error is, such as
`["body", "pipeline", "result", "formats"]`, as Pydantic gives it for a field the configuration refuses, or
`["body", "predicate", "index"]` for a bin a selection names that the histogram doesn't have. Every `404` and `409`
carries `{"detail": "<why>"}`.

## Run request

`POST /v1/validate` and `POST /v1/runs` take the same body:

| Field      | Type                   | Default            | Meaning                                                              |
| ---------- | ---------------------- | ------------------ | -------------------------------------------------------------------- |
| `pipeline` | Pipeline configuration | Required           | The configuration the batch command reads, as JSON                   |
| `tasks`    | List of task names     | Every enabled task | The tasks to run, in order; a named task runs even if it is disabled |

Both refuse with `422` a request that would fail before any task ran:

- a field the pipeline configuration does not define, or a value it refuses
- a task the pipeline does not define, or a pipeline whose every task is disabled
- a `result: require` when none of the tasks to run gives a verdict (only presets such as `audit` do)
- a `result: formats` without `json`: the service reads each run's results as JSON

`POST /v1/validate` answers with `valid`, the resolved `tasks`, and the `pipeline` the run would read. Validation reads
no data: a pipeline that validates can still fail on a dataset that is missing or unreadable, which the run reports.

## Capabilities

`GET /v1/capabilities` returns `api_version`, `flow_version`, `dataeval_version`, `max_active_runs`, `features`,
`limits` and `steps`. `features` names each optional part of the API with its version, so a client checks for a
feature rather than a Flow version: `items`, `profiles`, `selections` and `schema`, each `1`. `limits.page_size` is the
largest page the item and selection routes serve. `steps` is the step catalog `dataeval-flow steps` prints: every
evaluator, transform, combine, check and workflow, with its ports and its settings' JSON Schema. `GET /v1/schema` is
the JSON Schema of the whole pipeline, which `pipeline` in a run request follows.

## Run record

| Field         | Meaning                                                                        |
| ------------- | ------------------------------------------------------------------------------ |
| `api_version` | `1`                                                                            |
| `id`          | The run's ID, a UUID                                                           |
| `status`      | Where the run is; see [Run states](#run-states)                                |
| `created_at`  | When it was queued                                                             |
| `started_at`  | When its process started, else `null`                                          |
| `finished_at` | When it reached its final state, else `null`                                   |
| `exit_code`   | Its process's exit code, once it has one; negative when a signal killed it     |
| `request`     | `{"tasks": [...]}`, the resolved task list                                     |
| `pipeline`    | The snapshot it runs from: the submitted pipeline with every default filled in |
| `error`       | Why it failed or was interrupted, else `null`                                  |

The snapshot alone defines a run. No `DATAEVAL_*` environment variable set for the service reaches a run's process.

## Run states

```text
queued ──► running ──► succeeded | failed
  │           │
  │           ├──► cancelling ──► cancelled
  │           └──► interrupted            (the service stopped)
  └──► cancelled
```

| Status        | Meaning                                                                                                      |
| ------------- | ------------------------------------------------------------------------------------------------------------ |
| `queued`      | Waiting; runs start one at a time, oldest first                                                              |
| `running`     | Its process is running                                                                                       |
| `cancelling`  | Asked to stop: its process group is sent `SIGTERM`, then `SIGKILL` three seconds later                       |
| `succeeded`   | It ran to its end: `exit_code` is `0`, or `3` or `4` when the pipeline's `result:` gate tripped              |
| `failed`      | Any other exit code, such as `1` for a failed task or export; also a run killed by a signal or never started |
| `cancelled`   | Cancelled while queued, or stopped while running                                                             |
| `interrupted` | The service stopped while it ran; it is not resumed, and a new run must be queued                            |

`succeeded`, `failed`, `cancelled` and `interrupted` are final: a final record never changes. `exit_code` follows the
batch command's: `3` when a finding breached its health threshold under `fail_on: warning`, and `4` when a verdict
fell short of `require`. Stopping the service interrupts the running run and keeps the queued ones, which start when
it serves again.

## Results

`GET /v1/runs/{id}/results` returns what the run's `result.json` holds: an object keyed by task name, each value a
task's result as the batch command writes it. Each task's result is there as soon as the task finishes, so a running
run shows the tasks it has finished. With `result: per_task`, the service merges the run's per-task files into one
object. A failed workflow task appears with the steps it completed; a failed evaluator task does not appear.

## Run directory

Under `--output`, the service keeps:

```text
<output>/runs/
├── runs.sqlite3          run history
├── service.lock          held by the service running the queue
└── <id>/
    ├── request.json      the resolved task list
    ├── pipeline.json     the snapshot
    ├── console.log       everything the run printed
    ├── result.log        the run's DEBUG log, with timestamps
    ├── results/          result.json, result.txt, result.html, and any manifests and encoding.json
    │   ├── manifests/    each content-digest task's manifest: <task>/content-digest.json
    │   └── profiles/     each profile task's rows: <task>/image.parquet, <task>/target.parquet
    ├── selections/       each selection's definition: <selection id>.json
    └── datasets/         exports the pipeline declares
```

Every file in a run's directory is listed by `GET /v1/runs/{id}/artifacts`. Keep the directory while its run is in
the history.

## Items

The item routes serve the images, boxes, labels and metadata a run read, for as long as the data still matches what
the run read. A run records what it read when it runs a `content-digest` task over the source: the manifest holds each
item's SHA-256 over its image and labels, and over its metadata, with its position under the source's view and beneath
it. The service keeps no copy of the data. To serve an item, it loads the source from the run's own snapshot, finds the
item the run read by its position beneath the view, hashes it again, and compares. A run that reads a source to show
it, before any assessment, is an ordinary run with a `content-digest` task:

```json
{
  "pipeline": {
    "datasets": [{"name": "harbor", "format": "coco", "path": "harbor"}],
    "sources": [{"name": "harbor", "dataset": "harbor"}],
    "evaluators": [
      {"name": "digest", "type": "content-digest"},
      {"name": "labels", "type": "label-health"},
      {"name": "profile", "type": "profile"}
    ],
    "tasks": [
      {"name": "digest", "evaluator": "digest", "sources": "harbor"},
      {"name": "labels", "evaluator": "labels", "sources": "harbor"},
      {"name": "profile", "evaluator": "profile", "sources": "harbor"}
    ]
  }
}
```

`GET /v1/runs/{id}/items/{source}?offset=0&limit=24` returns `schema`, `source`, `status`, `total`, `offset`, `limit`
and `items`, in the order the run read them. `status` is `recorded`, or `evidence_unavailable` with `total` `0` and a
`reason`: the run recorded no manifest of the source, or the source's view draws at random with no `seed` while the
pipeline sets none, so each task of the run drew its own items and a position in one task's result names another item
in another's. Give the operation or the pipeline a `seed:`. `limit` is at most `limits.page_size`. Each item holds:

| Field                         | Meaning                                                                                                                          |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `schema`                      | `1`                                                                                                                              |
| `source`, `index`             | The item, as a finding's evidence names it: its source, and its position after the source's view                                 |
| `root_index`                  | Its position in the dataset beneath the view                                                                                     |
| `id`                          | Its metadata `id`, where it has one                                                                                              |
| `status`                      | `verified`, `input_changed`, `input_unavailable` or `evidence_unavailable`                                                       |
| `reason`                      | Why it is not `verified`                                                                                                         |
| `width`, `height`, `channels` | The image's size, in pixels                                                                                                      |
| `box_format`                  | `xyxy`: each box is `x0, y0, x1, y1` in pixels, as MAITE holds it                                                                |
| `targets`                     | On detection data, each box: `target` (its index in the item, as `target_index` in a finding counts), `box`, `label` and `class` |
| `label`                       | On classification data, `{"label", "class"}`, or `null` for an item with no label                                                |
| `metadata`                    | The item's metadata as the dataset gives it, as JSON                                                                             |
| `image_url`                   | The path of the item's image                                                                                                     |

Only a `verified` item carries its image, boxes and metadata.

| Status                 | Meaning                                                                                                                                |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `verified`             | The item's image, labels and metadata hash as the run recorded them                                                                    |
| `input_changed`        | The item's image, labels or metadata differ, or the source no longer holds the item                                                    |
| `input_unavailable`    | The source does not load, or the item does not read, such as an image file that was removed                                            |
| `evidence_unavailable` | The run recorded no manifest of the source, such as a run made before manifests recorded it, or each task drew the source's items anew |

`GET /v1/runs/{id}/items/{source}/{index}/image` returns the item's image as a PNG at its own size. `target` crops one
box with a margin of a tenth of its size. `max_side` shrinks the image to fit that many pixels across. An item that is
not `verified` answers `409` with the item's status, a box the item doesn't hold `404`, and an item that is no image
`422`. A float image is scaled to 0–255 by the dataset's `value_range`, else by its own range.

## Selections

A selection is the exact set of rows a predicate holds over what a run wrote: a `profile` task's rows of one field, or
the rows of a table a task wrote, such as `outliers`'. It never measures anything again and is never a sample: every
row the run kept is considered.

`POST /v1/runs/{id}/selections` takes:

| Field       | Meaning                                                                                                 |
| ----------- | ------------------------------------------------------------------------------------------------------- |
| `task`      | The task whose result is selected from                                                                  |
| `step`      | A step of the task's chain, for a table a step wrote from a workflow input; unset for an evaluator task |
| `field`     | The profile field, by name; unset for `rows`                                                            |
| `scope`     | `image` or `target`; needed only where the field was profiled in both                                   |
| `origin`    | `computed` or `supplied`; needed only where a statistic and a metadata field share the name             |
| `predicate` | One of the predicates below                                                                             |

| Predicate                                                       | Selects                                                                                                                 |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `{"kind": "bin", "index": 2}`                                   | The rows the field's histogram counts in that bin: `low <= x < high`, and `low <= x <= high` for the last bin           |
| `{"kind": "range", "min", "max", "include_min", "include_max"}` | The finite values within the bounds; an unset bound is open. `include_min` defaults to `true`, `include_max` to `false` |
| `{"kind": "category", "values": [...]}`                         | The rows holding any of the values; `1`, `"1"` and `true` are three values                                              |
| `{"kind": "other"}`                                             | The rows holding a value the profile did not name                                                                       |
| `{"kind": "missing"}`                                           | The rows with no value                                                                                                  |
| `{"kind": "non_finite"}`                                        | The rows holding NaN or an infinity                                                                                     |
| `{"kind": "rows", "metric": "brightness"}`                      | A table's rows, each a flag on an item or a box; only one metric's when `metric` is set                                 |

A step's rows count items within the Dataset it read, so a `rows` selection through a step that reads what an earlier
step derived, such as a `view` or a split, is refused: only a step that reads a workflow input counts items within the
source. A selection over a source each task drew anew is refused too, as its items are under [Items](#items).

The response is the summary: `schema`, `id`, `definition`, `source`, `scope`, and `total` rows, the distinct `images`
they fall on, and the distinct `targets` (boxes). One image can carry several flags and several boxes, so the three
counts differ. The same request gives the same `id`, after a restart too.

`GET /v1/runs/{id}/selections/{sel}?offset=0&limit=24` returns the summary with `offset`, `limit` and `members`. Each
member names its `source`, `index` and `target` (`null` for an image), which the item routes take, with its `value`
for a profile field, or the table's row (`metric_name`, `metric_value`, `bound`, `direction` and so on). Members come
in a fixed order, by item then box for a profile and in the table's order for rows, so the pages together hold each
member once.

`GET /v1/runs/{id}/selections/{sel}/view` returns a `views:` entry and a `sources:` entry. Merged into the run's
pipeline, the new source reads the selection's images through the source's own view and then an `Indices` operation.
A selection of boxes is refused unless `parents=true` asks for the images that hold them: keeping only those boxes, or
removing them, is not supported. A source whose view shuffles without a seed is refused, since no later run would
read it alike.

## Health probes

| Endpoint   | `200` when                      | `503` when                                          |
| ---------- | ------------------------------- | --------------------------------------------------- |
| `/healthz` | The service takes and runs work | Its queue has stopped, or its run store is unusable |
| `/readyz`  | Same as `/healthz`              | Same as `/healthz`                                  |
| `/livez`   | The process can run work        | Its queue has stopped and it needs a restart        |

A `200` body is `{"status": "ok"}`. A `503` body is `{"status": "unavailable", "reasons": [...]}`. Its reasons are
`queue-stopped` and `store-unavailable`. The store is unusable when its database cannot be queried within a second,
or the run directory cannot be written. A probe never reads a run's data, so it answers quickly even while a run is
in progress.

## Logs

The service logs to standard output. Each line carries a UTC timestamp and a severity, as the batch command's
`structured` format does, or the bare message under `--log-format plain`. It logs every request, each run's
lifecycle (queued, started, and its final status), and every line a run prints, prefixed `run <first 8 characters
of the ID>:`. A line a run prints as a warning or an error keeps that severity; everything else a run writes to
standard error is logged as a warning.

`GET /v1/runs/{id}/logs` returns the last 64 KiB of the run's `console.log`. The run's `result.log` holds its full
`DEBUG` log, and is listed with its artifacts.

## Limits

- Only one run executes at a time. The item and selection routes read what runs wrote, so they answer while a run
  executes.
- The item routes load a source in the service's own process and keep the four most recently read loaded. A source
  that takes long to load makes its first page slow.
- A page holds at most 100 items or members.
- `GET /v1/runs` returns the whole history in one response.
- The service runs on Linux and macOS: it relies on POSIX process groups and file locks.
- `/docs` needs internet access in the viewer's browser for its page assets; `/openapi.json` needs none.
