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

| Method and path                      | Success                        | Errors     | Does                                                       |
| ------------------------------------ | ------------------------------ | ---------- | ---------------------------------------------------------- |
| `GET /healthz`                       | `200`                          | `503`      | Whether the service takes and runs work                    |
| `GET /readyz`                        | `200`                          | `503`      | The same, as a readiness probe                             |
| `GET /livez`                         | `200`                          | `503`      | Whether the process can run work, as a liveness probe      |
| `GET /openapi.json`                  | `200`                          |            | The API's OpenAPI description                              |
| `GET /docs`                          | `200`                          |            | That description rendered; its page assets load from a CDN |
| `GET /v1/capabilities`               | `200`                          |            | Versions, runs at once, and every step a pipeline can use  |
| `POST /v1/validate`                  | `200` with the checked request | `422`      | Check a run request without queueing it                    |
| `POST /v1/runs`                      | `202` with the run record      | `422`      | Check a run request and queue it                           |
| `GET /v1/runs`                       | `200` with every run record    |            | Run history, newest first                                  |
| `GET /v1/runs/{id}`                  | `200` with the run record      | `404`      | One run                                                    |
| `POST /v1/runs/{id}/cancel`          | `202` with the run record      | `404, 409` | Cancel a queued run, or stop a running one                 |
| `GET /v1/runs/{id}/results`          | `200` with the results         | `404, 409` | Each finished task's result                                |
| `GET /v1/runs/{id}/logs`             | `200`, plain text              | `404`      | The last 64 KiB of what the run printed                    |
| `GET /v1/runs/{id}/artifacts`        | `200` with a list of paths     | `404`      | Every file in the run's directory                          |
| `GET /v1/runs/{id}/artifacts/{path}` | `200` with the file            | `404`      | One file, by a path the list gives                         |

Request and response bodies are JSON, except the logs and the files. Times are Unix epoch seconds.

### Errors

| Status | When                                                                                   |
| ------ | -------------------------------------------------------------------------------------- |
| `404`  | No run has the ID, or the run holds no file at the path                                |
| `409`  | Cancelling a run that has finished; reading results before any task has written them   |
| `422`  | The request is malformed, or the run would fail before any task ran; the body says why |
| `503`  | From a probe only: the service cannot run work; the body names the reasons             |

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

`POST /v1/validate` answers with `valid`, the resolved `tasks`, and the `pipeline` the run would read.

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
    └── datasets/         exports the pipeline declares
```

Every file in a run's directory is listed by `GET /v1/runs/{id}/artifacts`. Keep the directory while its run is in
the history.

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

- Only one run executes at a time.
- `GET /v1/runs` returns the whole history in one response.
- The service runs on Linux and macOS: it relies on POSIX process groups and file locks.
- `/docs` needs internet access in the viewer's browser for its page assets; `/openapi.json` needs none.
