# Run Flow as a service

`dataeval-flow serve` keeps DataEval Flow running behind an HTTP API. A client submits a pipeline, the service queues
it, and runs it as the batch `dataeval-flow` command would, in a process of its own. The client can disconnect and
come back for the results. Use it when an application, such as a UI, submits work; use the batch command for CI
and scripted runs.

## Install and start it

The service needs the `service` extra:

```bash
pip install "dataeval-flow[service]"
dataeval-flow serve --data /path/to/data --output /path/to/service-output
```

It listens on `127.0.0.1:8001` and stops on Ctrl+C or `SIGTERM`. `--host` and `--port`, or
`DATAEVAL_SERVICE_HOST` and `DATAEVAL_SERVICE_PORT`, change that. `dataeval-flow serve --help` lists every option and
environment variable.

The roots follow the batch command's:

| Option         | Environment variable | Holds                                                               |
| -------------- | -------------------- | ------------------------------------------------------------------- |
| `-d, --data`   | `DATAEVAL_DATA`      | Datasets and models; pipelines' relative paths resolve against it   |
| `-o, --output` | `DATAEVAL_OUTPUT`    | Each run's snapshot, logs and result files, and the run history     |
| `-k, --cache`  | `DATAEVAL_CACHE`     | The computation cache runs share; defaults to `<output>/cache`      |

The service has no authentication and reads no secrets. Bind it to a trusted network only. A pipeline runs with the
service's own file permissions, as it would from the batch command.

## In a container

Every image ships the service. Publish its port, and mount the same volumes as for a batch run:

```bash
docker run --init -p 8001:8001 \
    --mount type=bind,source=/home/user/myproject,target=/dataeval,readonly \
    --mount type=bind,source=/home/user/service-output,target=/output \
    harbor.jatic.net/aria/dataeval-flow:latest-cpu python -m dataeval_flow serve
```

Images set `DATAEVAL_SERVICE_HOST=0.0.0.0`, so the service is reachable through the published port. `--init`
reaps the processes runs leave behind.

## Submit a pipeline

`POST /v1/runs` takes the pipeline, as the JSON form of the YAML the batch command reads, and the tasks to run. It
answers `202` with the run's record. Leave `tasks` out to run every enabled task:

```bash
curl -s -X POST localhost:8001/v1/runs -H 'Content-Type: application/json' -d '{
  "pipeline": {
    "datasets": [{"name": "boats", "format": "coco", "path": "data/boats"}],
    "sources": [{"name": "data", "dataset": "boats"}],
    "workflows": [{"name": "metadata", "type": "triage"}],
    "tasks": [{"name": "metadata", "workflow": "metadata", "sources": "data"}]
  }
}'
```

`POST /v1/validate` checks the same body without queueing it. Both refuse with `422` a pipeline that would fail
before any task ran, such as an unknown task, or a `require:` no task can judge. The service reads results as JSON,
so keep `json` in `result: formats`.

The record holds a snapshot of the pipeline with every default filled in. The run reads that snapshot alone. No
`DATAEVAL_*` variable set for the service changes a run.

## Follow a run

| Endpoint                                 | Gives                                                              |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `GET /v1/runs`                           | Every run, newest first                                            |
| `GET /v1/runs/{id}`                      | Its status, exit code and snapshot                                 |
| `GET /v1/runs/{id}/results`              | Each finished task's result, as `result.json` holds it             |
| `GET /v1/runs/{id}/logs`                 | The last 64 KiB of what the run printed                            |
| `GET /v1/runs/{id}/artifacts`            | Every file the run wrote; append a listed path to fetch one        |
| `POST /v1/runs/{id}/cancel`              | Cancel a queued run, or stop a running one                         |
| `GET /v1/capabilities`                   | Versions, and every step a pipeline can chain                      |

A run is `queued`, `running` or `cancelling`, then `succeeded`, `failed`, `cancelled` or `interrupted`. One run
runs at a time, and the rest wait in order. `succeeded` means the run ran to its end. Its `exit_code` is the batch
command's: `0`, or `3` or `4` when the pipeline's `result:` block gates on warnings or a verdict. `failed` means a
task failed or the run could not finish. Its `error` says how, and its log says why.

Each task's results are written as soon as it finishes, so `/results` shows a running run's finished tasks. A
cancelled run stops within three seconds. Stopping the service interrupts the running run and keeps the queued
ones, which run when it starts again. The history survives restarts.

The service logs requests, run lifecycle events and every line a run prints, tagged with the run's ID, to standard
output. Each line carries a UTC timestamp and a severity.

## Health checks and interface documentation

| Endpoint       | `200` when                                  | `503` when                                          |
| -------------- | ------------------------------------------- | --------------------------------------------------- |
| `/healthz`     | The service takes and runs work             | Its queue has stopped, or its run store is unusable |
| `/readyz`      | Same as `/healthz`                          | Same as `/healthz`                                  |
| `/livez`       | The process can run work                    | Its queue has stopped and it needs a restart        |

A `503` body names its reasons: `queue-stopped` or `store-unavailable`. `/openapi.json` describes the whole API
offline. `/docs` renders it, but loads its page assets from a CDN.
