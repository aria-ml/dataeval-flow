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

| Method and path                                  | Success                        | Errors          | Does                                                                                      |
| ------------------------------------------------ | ------------------------------ | --------------- | ----------------------------------------------------------------------------------------- |
| `GET /healthz`                                   | `200`                          | `503`           | Whether the service takes and runs work                                                   |
| `GET /readyz`                                    | `200`                          | `503`           | The same, as a readiness probe                                                            |
| `GET /livez`                                     | `200`                          | `503`           | Whether the process can run work, as a liveness probe                                     |
| `GET /openapi.json`                              | `200`                          |                 | The API's OpenAPI description                                                             |
| `GET /docs`                                      | `200`                          |                 | That description rendered; its page assets load from a CDN                                |
| `GET /v1/capabilities`                           | `200`                          |                 | Versions, features, limits, and every step a pipeline uses                                |
| `GET /v1/schema`                                 | `200`                          |                 | The JSON Schema of a pipeline, every registered step in it                                |
| `POST /v1/validate`                              | `200` with the checked request | `409, 413, 422` | Check a run request without queueing it                                                   |
| `POST /v1/resolve`                               | `200` with the resolved graphs | `409, 413, 422` | Resolve each task's graph without reading data; see [Graph resolution](#graph-resolution) |
| `POST /v1/runs`                                  | `202` with the run record      | `409, 413, 422` | Check a run request and queue it                                                          |
| `GET /v1/runs`                                   | `200` with every run record    |                 | Run history, newest first                                                                 |
| `GET /v1/runs/{id}`                              | `200` with the run record      | `404`           | One run                                                                                   |
| `POST /v1/runs/{id}/cancel`                      | `202` with the run record      | `404, 409`      | Cancel a queued run, or stop a running one                                                |
| `GET /v1/runs/{id}/results`                      | `200` with the results         | `404, 409`      | Each finished task's result                                                               |
| `GET /v1/runs/{id}/logs`                         | `200`, plain text              | `404`           | The last 64 KiB of what the run printed                                                   |
| `GET /v1/runs/{id}/activity`                     | `200` with the snapshot        | `404, 409`      | Each task's and step's state; see [Activity](#activity)                                   |
| `GET /v1/runs/{id}/events`                       | `200` with a page of events    | `404, 409, 422` | The changes after a sequence; see [Activity](#activity)                                   |
| `GET /v1/runs/{id}/artifacts`                    | `200` with a list of paths     | `404`           | Every file in the run's directory                                                         |
| `GET /v1/runs/{id}/artifacts/{path}`             | `200` with the file            | `404`           | One file, by a path the list gives                                                        |
| `GET /v1/runs/{id}/items/{source}`               | `200` with a page of items     | `404, 422`      | A source's items as the run read them; see [Items](#items)                                |
| `GET /v1/runs/{id}/items/{source}/{index}`       | `200` with the item            | `404`           | One item, checked against the run's manifest                                              |
| `GET /v1/runs/{id}/items/{source}/{index}/image` | `200`, PNG                     | `404, 409`      | The item's image, or one box cropped from it                                              |
| `POST /v1/runs/{id}/selections`                  | `200` with the summary         | `404, 409, 422` | Select rows of a profile or a table; see [Selections](#selections)                        |
| `GET /v1/runs/{id}/selections/{sel}`             | `200` with a page of members   | `404, 409`      | A selection's summary and members                                                         |
| `GET /v1/runs/{id}/selections/{sel}/view`        | `200` with a source            | `404, 422`      | A selection's images as a source a later run reads                                        |

Request and response bodies are JSON, except the logs and the files. Times are Unix epoch seconds.

### Errors

| Status | When                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `404`  | No run has the ID; the run holds no file at the path; its pipeline names no such source; the source held no such item; no selection has the ID                                                                                                                                                                                                                                                                                                                                                        |
| `409`  | A request pinning a `schema_fingerprint` other than the service's (`schema_changed`, an extension error, on `/v1/validate`, `/v1/resolve` and `/v1/runs`); a run's activity from before the service recorded it (`activity_unavailable`, an extension error); cancelling a run that has finished; reading results, or selecting from them, before the task has written them, or of a run that finished without any (`The run finished without a result`); the image of an item that is not `verified` |
| `413`  | A `POST` body over `limits.max_document_bytes`, or a pipeline `/v1/resolve` finds to hold more than `limits.max_resolved_nodes` nodes (`too_many_nodes`, with `limit` and `count`); the body is an extension error                                                                                                                                                                                                                                                                                    |
| `422`  | The request is malformed, the run would fail before any task ran, or the run cannot answer the selection; an `after` beyond the run's last event, or a `limit` outside 1 to 100; the body says why                                                                                                                                                                                                                                                                                                    |
| `503`  | From a probe only: the service cannot run work; the body names the reasons                                                                                                                                                                                                                                                                                                                                                                                                                            |

A `422` body is `{"detail": [{"type", "loc", "msg"}, ...]}`: `loc` is where in the request the error is, such as
`["body", "pipeline", "result", "formats"]`, as Pydantic gives it for a field the configuration refuses, or
`["body", "predicate", "index"]` for a bin a selection names that the histogram doesn't have. Every `404` and `409`
carries `{"detail": "<why>"}`, except the extension errors.

The builder contracts' errors (`schema_changed`, `activity_unavailable`, `document_too_large` and `too_many_nodes`)
carry the extension envelope instead: `{"schema": 1, "code": "<code>", "message": "<why>", "retryable": false,
"details": {…}}`. A client branches on `code` and shows `message`.

## Run request

`POST /v1/validate`, `POST /v1/resolve` and `POST /v1/runs` take the same body:

| Field                         | Type                   | Default            | Meaning                                                                                                                              |
| ----------------------------- | ---------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| `pipeline`                    | Pipeline configuration | Required           | The configuration the batch command reads, as JSON                                                                                   |
| `tasks`                       | List of task names     | Every enabled task | The tasks to run, in order; a named task runs even if it is disabled                                                                 |
| `expected_schema_fingerprint` | `schema_fingerprint`   | None               | A `schema_fingerprint` from `/v1/capabilities`. When given and the service's differs, `409 schema_changed`, read before the pipeline |

`/v1/schema` describes `pipeline`; the request's OpenAPI types it as an object.

All three routes refuse with `422` a request that would fail before any task ran:

- a field the pipeline configuration does not define, or a value it refuses
- a task the pipeline does not define, or a pipeline whose every task is disabled
- a `result: require` when none of the tasks to run gives a verdict (only presets such as `audit` do)
- a `result: formats` without `json`: the service reads each run's results as JSON

A pipeline whose workflows don't connect is refused by `/v1/validate` and `/v1/runs` with one `422` entry per problem,
while `/v1/resolve` answers `200` and reports the same problems as `diagnostics`. Each entry's `loc` is the problem's
step, and its port when it has one, such as `["body", "pipeline", "workflows", 0, "steps", 1, "input"]`.

`POST /v1/validate` answers with `valid`, the resolved `tasks`, and the `pipeline` the run would read. Validation reads
no data: a pipeline that validates can still fail on a dataset that is missing or unreadable, which the run reports.

## Graph resolution

`POST /v1/resolve` takes the [run request](#run-request) and gives the graph of each task it would run: the steps, their
ports, what each port reads, and where the graph does not connect. It reads no data, loads no source and queues
nothing, so it answers for a pipeline whose datasets are not there. It checks the pin and parses the pipeline as
`/v1/validate` does, so a malformed pipeline is the same `422`; a graph that does not connect is a `200` with
`valid: false`. A pipeline resolving to more than `limits.max_resolved_nodes` nodes is a `413` `too_many_nodes`.

The response has:

| Field                | Meaning                                                                                                                                      |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema`             | The contract version, `1`                                                                                                                    |
| `schema_fingerprint` | The service's, as in `/v1/capabilities`                                                                                                      |
| `valid`              | Whether `diagnostics` is empty                                                                                                               |
| `pipeline`           | The normalized pipeline, as `/v1/validate` gives it                                                                                          |
| `tasks[]`            | One entry per task to run: `name`, `kind`, `workflow`, `source_bindings`, `nodes`, `execution_order`, `output_aliases`, `complete`, `matrix` |
| `workflows[]`        | The graphs of workflow entries no selected task runs, so a canvas can show them                                                              |
| `diagnostics[]`      | Each problem: `code`, `severity`, `message`, `document_pointer`, `task`, `node_id`, `port`                                                   |

`source_bindings` lists each workflow input with its `slot`, whether it is a `list`, the `sources` bound to it and the
`document_pointer` of the task's `sources`. `execution_order` lists the `node_id`s in the order the steps run.
`output_aliases` maps an address such as `cleaning.clean` to the `node_id` and `port` that holds it. `complete` is
`false` when the graph stops short of a problem. In `tasks[]`, `workflow` names the entry the task runs, and `kind` says
whether that is a `workflows` or an `evaluators` entry.

### Nodes

`node_id` is `"<task>:<step>"`, with the steps a preset adds named `<step>/<inner>`. Treat it as opaque. `path` is the
task followed by the step's name split at `/`. A node in `workflows[]` uses the workflow's name where a task's would
go. For a workflow task, step names are the keys of its result's `steps`, so a resolved node matches the final result
without translation. An evaluator task's single node is `<task>:<task>`.

`origin` says where a node came from:

- `step_pointer` is a JSON Pointer to the step, such as `/workflows/0/steps/1`. An evaluator task's node has `null`.
- `definition_pointer` points at the `evaluators` or `workflows` entry a pooled step uses, and is `null` otherwise.
- `generated` is `true` for a step a preset adds. It carries the preset step's `step_pointer` and the preset entry as
  `definition_pointer`: the service never points at a step that is not in the document.

Each node's `inputs` give the port, its `type`, `count`, accepted `classes` and `kinds`, `shape`, and the `addresses` it
reads. Its `outputs` give the `address` another step reads it by, the `classes` it holds and its `keys`.

An output's `availability` is `available`, or `unavailable` with a `reason` when the step's settings leave it empty, so
a step reading it is an `unavailable_output` problem. `keys` is the list of keys a list output holds, or `null` when the
data decides them.

### Diagnostics

A diagnostic's `document_pointer` points at the port when it has one, else at the step, else at the workflow or task.
`task` is `null` for a workflow no selected task runs. Its `node_id` names the step the problem is on, which may not be
among `nodes`: the graph stops before it, or it is a preset step whose nodes are `<step>/<inner>`. Locate it by
`document_pointer`. A `task_problem` can name a task that isn't in the selected `tasks[]`, because every task's bindings
are checked, as `/v1/validate` checks them. Each `code` is one of:

| Code                 | Raised when                                                                                                                |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `unknown_address`    | An address names no slot, step or output                                                                                   |
| `forward_address`    | It names a step that runs later                                                                                            |
| `ambiguous_output`   | A bare name is used on a multi-output step, or `.output` on a single-output step                                           |
| `unavailable_output` | The settings leave the output empty                                                                                        |
| `wrong_type`         | The data type doesn't match the port                                                                                       |
| `wrong_class`        | The Output class doesn't fit the port's classes                                                                            |
| `list_mismatch`      | A list is read where a non-list port won't broadcast, lists are nested, `[key]` is used on a non-list, or a key is unknown |
| `count_mismatch`     | The port's `count` is violated                                                                                             |
| `wrong_dataset`      | A `same_node`, `shared_datasets` or `computed_on` rule is violated                                                         |
| `too_many_verdicts`  | A workflow has more than one verdict-giving step                                                                           |
| `bad_extractor`      | A step's `extractor` names no entry or is invalid                                                                          |
| `bad_by`             | `by:` or `pairs:` is misused                                                                                               |
| `unknown_entry`      | An evaluator or workflow step names an entry its pool lacks, or a custom workflow                                          |
| `bad_address`        | A port field holds text that isn't an address, or a pooled step has no `input:`                                            |
| `task_problem`       | A task's sources don't bind to its workflow's inputs, an extractor is missing, or exports clash                            |
| `graph_error`        | Anything else                                                                                                              |

A workflow reports at most one graph problem: resolution stops at the first, and the steps before it are kept in the
graph. Steps run in the order written, so reading a step listed later is `forward_address`. A task with a `matrix`
resolves once, as written: `matrix.runs` lists the runs as `{number, label, values}`, but a cell that changes which
outputs a step leaves empty, or which keys a list holds, is not reflected in the graph.

## Capabilities

`GET /v1/capabilities` returns `api_version`, `flow_version`, `dataeval_version`, `schema_fingerprint`,
`max_active_runs`, `features`, `limits` and `steps`. `features` names each optional part of the API with its version, so
a client checks for a feature rather than a Flow version: `items`, `profiles`, `selections`, `schema`,
`builder_metadata`, `graph_resolution` and `step_activity`, each `1`. `limits.page_size` is the largest page the item
and selection routes serve. `limits.max_document_bytes`, `limits.max_resolved_nodes` and `limits.activity_page_size` are
the bounds in [Limits](#limits). `steps` is the step catalog `dataeval-flow steps` prints: every evaluator, transform,
combine, check and workflow, with its ports and its settings' JSON Schema.
`GET /v1/schema` is the JSON Schema of the whole pipeline, which `pipeline` in a run request follows.

`schema_fingerprint` is a SHA-256 over the pipeline schema and the step catalog, so it changes with the Flow and DataEval
versions and with the plugins installed. A client reads it with the schema and catalog, and sends it back as
`expected_schema_fingerprint`. The service computes it once: restart the service after installing a plugin.

The catalog's entries also give: `shape` on each input (the form of the address it takes), `same_dataset` (which inputs
must come from one dataset), `address_grammar` (how an address is written), `step_settings` (where each kind's settings
live in a step) and `classes` (each Output class's ancestors).

`/v1/schema` marks each field that names another pool's entry with `x-flow-ref`, in two forms: `{"pool": "<pool>"}`, and
`{"pool": "ontologies", "or_path": true}` for an ontology field, which takes an entry's name or else a file path. A
task's `workflow` and `evaluator` keys each carry their own pool, `workflows` and `evaluators`; the schema never shows
which kind a task runs.

## Run record

| Field                | Meaning                                                                        |
| -------------------- | ------------------------------------------------------------------------------ |
| `api_version`        | `1`                                                                            |
| `id`                 | The run's ID, a UUID                                                           |
| `status`             | Where the run is; see [Run states](#run-states)                                |
| `created_at`         | When it was queued                                                             |
| `started_at`         | When its process started, else `null`                                          |
| `finished_at`        | When it reached its final state, else `null`                                   |
| `exit_code`          | Its process's exit code, once it has one; negative when a signal killed it     |
| `request`            | `{"tasks": [...]}`, the resolved task list                                     |
| `pipeline`           | The snapshot it runs from: the submitted pipeline with every default filled in |
| `schema_fingerprint` | The fingerprint the run was checked under                                      |
| `error`              | Why it failed or was interrupted, else `null`                                  |

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
    ├── activity.jsonl    each task's and step's state, one event per line
    ├── results/          result.json, result.txt, result.html, and any manifests and encoding.json
    │   ├── manifests/    each content-digest task's manifest: <task>/content-digest.json
    │   └── profiles/     each profile task's rows: <task>/image.parquet, <task>/target.parquet
    ├── selections/       each selection's definition: <selection id>.json
    └── datasets/         exports the pipeline declares
```

Every file in a run's directory is listed by `GET /v1/runs/{id}/artifacts`. Keep the directory while its run is in
the history.

## Activity

`GET /v1/runs/{id}/activity` returns a snapshot: `schema` (`1`), `run_id`, `schema_fingerprint` (the one the run was
checked under), `sequence` (the last event it folds), `first_available_sequence` (`1`), `run_status`, `tasks` and
`steps`. A task row is `{task, status, started_at, finished_at, reason}`. A step row is `{node_id,
definition_node_id, task, path, instances, status, started_at, finished_at, reason, not_assessed, elapsed_s,
elements}`; a step's `node_id` is `<task>:<step>`, with `#<n>` after it for a matrix run's instance.

`GET /v1/runs/{id}/events?after=S&limit=N` returns a page: `schema`, `run_id`, `after`, `next_after`,
`snapshot_sequence` (the run's last event now), `first_available_sequence`, `has_more` and `events`. `after`
defaults to `0` and `limit` to 100, its most. An event is `{sequence, timestamp, kind, run_id}` and one row, the full
state of what it is about: `run` for a `run_state`, `task` for a `task_state`, `step` for a `step_state`.

To follow a run, read a snapshot at `sequence` S, then page from `after=S` until `has_more` is `false`, and again
later. Each event carries the full row, so applying one sets that state: replaying events is safe, and a client that
missed some can fetch a snapshot again.

| Step status               | Meaning                                                                                                               |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `pending`                 | Not started. It stays pending after a cancel or a stop, with a `reason` saying so                                     |
| `running`                 | Started and not finished; at most one step runs at a time                                                             |
| `ok`, `failed`, `skipped` | As the chain records them; only the chain sets `skipped` (a missing input from a failed step, or an optional failure) |
| `cancelled`               | Was running when the run was cancelled                                                                                |
| `interrupted`             | Was running when the worker died or the service stopped                                                               |

A task has the same statuses. `not_assessed` is separate from status: a check that ran `ok` but judged nothing carries
the reason Flow gives. `run_status` is the run's status, except that `cancelling` is not an activity state: the
snapshot shows `running` until the run closes, up to three seconds after a cancel.

Steps run one at a time, in config order. A task with a matrix runs its steps once per run: each is an instance,
`<task>:<step>#<n>`, with `definition_node_id` naming the step it repeats. A step that takes a list reports its
elements, each with a status, a reason and `not_assessed`, on its finished row.

The service closes a run's activity before its record turns final, so activity never trails the record: a run that is
cancelled, fails or is interrupted leaves its running steps `cancelled` or `interrupted`, and the rest `pending` with a
reason. Events are kept for the run's life, so `first_available_sequence` is always `1` and `cursor_expired` is never
sent. A run queued before the service recorded activity has no events: its activity routes answer `409
activity_unavailable`, and its results, logs and artifacts stay readable.

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
- Every `POST` body is at most `max_document_bytes` (1 MiB); a larger one is refused with `413` before it is read. The
  bound checks `Content-Length`, so a body sent without one is not bounded.
- `POST /v1/resolve` resolves at most `max_resolved_nodes` (1000) nodes.
- The activity routes page at most `activity_page_size` (100) events.
- `GET /v1/runs` returns the whole history in one response.
- The service runs on Linux and macOS: it relies on POSIX process groups and file locks.
- `/docs` needs internet access in the viewer's browser for its page assets; `/openapi.json` needs none.
