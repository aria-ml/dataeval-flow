# Run Flow as a service

Keep DataEval Flow running behind an HTTP API, so a client can submit a pipeline, get a run ID back at once, and
check on the run later. The service runs each pipeline as a separate batch `dataeval-flow` process. Use it when the
caller shouldn't wait for a run, as with a user interface. For CI and scripted runs, use the batch command.
{doc}`../concepts/ExecutionModes` explains the difference.

## Used in these tutorials

- {doc}`Run a full evaluation pipeline end to end <../notebooks/end_to_end>`, whose pipeline you can submit to the
  service as it is

## Prerequisites

- DataEval Flow installed with the `service` extra, or a DataEval Flow container image, which ships it
- Linux or macOS
- A pipeline configuration, and the datasets it names under a data root

## 1. Start the service

```bash
pip install "dataeval-flow[service]"
dataeval-flow serve --data /path/to/data --output /path/to/service-output
```

It listens on `127.0.0.1:8001` and stops on Ctrl+C or `SIGTERM`. `--host` and `--port` change that, as do
`DATAEVAL_SERVICE_HOST` and `DATAEVAL_SERVICE_PORT`. Runs share the cache in `--cache`, else `<output>/cache`.

In a container, mount the same volumes as for a batch run, publish the port, and pass `--init`, which reaps the
processes finished runs leave behind:

```bash
docker run --init -p 8001:8001 \
    --mount type=bind,source=/home/user/myproject,target=/dataeval,readonly \
    --mount type=bind,source=/home/user/service-output,target=/output \
    harbor.jatic.net/aria/dataeval-flow:latest-cpu python -m dataeval_flow serve
```

In the images, the service listens on `0.0.0.0`, so the published port reaches it. It has no authentication, so
publish the port on a trusted network only.

## 2. Submit a pipeline

Send the pipeline, as the JSON form of the YAML the batch command reads, to `POST /v1/runs`. Leave `tasks` out to run
every enabled task, or name the tasks to run, in order:

```bash
curl -s -X POST localhost:8001/v1/runs -H 'Content-Type: application/json' -d '{
  "pipeline": {
    "datasets": [{"name": "boats", "format": "coco", "path": "data/boats"}],
    "sources": [{"name": "data", "dataset": "boats"}],
    "workflows": [{"name": "metadata", "type": "triage"}],
    "tasks": [{"name": "metadata", "workflow": "metadata", "sources": "data"}]
  },
  "tasks": ["metadata"]
}'
```

The service answers `202` with the run's record. Keep its `id`. A pipeline that would fail before any task ran, such
as one naming an unknown task, is refused with `422` and the reason. Send the same body to `POST /v1/validate` to check
it without queueing it.

To submit a YAML pipeline you already have, such as the tutorial's `end_to_end.yaml`, convert it to JSON on the way:

```bash
python -c 'import json, sys, yaml; print(json.dumps({"pipeline": yaml.safe_load(sys.stdin)}))' < end_to_end.yaml \
    | curl -s -X POST localhost:8001/v1/runs -H 'Content-Type: application/json' -d @-
```

Keep `json` in the pipeline's `result: formats`, as the service reads each run's results as JSON.

## 3. Follow the run

```bash
RUN=<the run's id>
curl -s localhost:8001/v1/runs/$RUN            # status, exit_code, error, and the snapshot it runs from
curl -s localhost:8001/v1/runs/$RUN/logs       # the last 64 KiB of what it printed
curl -s localhost:8001/v1/runs/$RUN/results    # each finished task's result, keyed by task
```

A run is `queued`, then `running`, and ends `succeeded`, `failed`, `cancelled` or `interrupted`. `succeeded` means it
ran to its end. Read its `exit_code` as the batch command's: `0`, or `3` or `4` when the pipeline's `result:` block
gates on warnings or a verdict. `/results` answers `409` until the first task finishes, then grows as each one does.

`GET /v1/runs/$RUN/artifacts` lists every file the run wrote, such as `results/result.html`; append a listed path to
fetch one. `GET /v1/runs` lists every run, newest first, and survives restarts.

## 4. Cancel a run

```bash
curl -s -X POST localhost:8001/v1/runs/$RUN/cancel
```

Cancelling a queued run removes it before it starts. A running run stops within three seconds, along with any
processes it started. Cancelling a run that has already finished returns `409`.

## 5. Probe its health

Point an orchestrator's probes at the service's port. `/healthz` and `/readyz` answer `200` while the service takes and
runs work; `/livez` answers `503` only when it needs a restart. On Kubernetes:

```yaml
livenessProbe:
  httpGet: {path: /livez, port: 8001}
readinessProbe:
  httpGet: {path: /readyz, port: 8001}
```

## Next steps

- {doc}`../reference/service` lists every endpoint, the run record, run states, the files a run leaves, the probes and
  the logs.
- {doc}`containerized_workflows` covers the image variants, mounts and GPU flags.
- {doc}`reuse_results_with_cache` explains what the shared cache keeps between runs.
