# Execution Modes: Library, Command and Service

A DataEval Flow pipeline is data: a configuration naming datasets, steps and tasks
(see {doc}`Reproducibility`). Running it means handing a configuration to Flow and
waiting for the result. The **execution mode** determines how the caller waits, where
the work runs, and where the output goes. The pipeline and the results it
produces from the same data are the same in every mode.

| Mode    | Entry point            | Caller                         | Runs in              | Output                      |
| ------- | ---------------------- | ------------------------------ | -------------------- | --------------------------- |
| Library | `run_tasks()`, `run()` | Python code; blocks            | The caller's process | Result objects              |
| Command | `dataeval-flow`        | Shell, CI or scheduler; blocks | Its own process      | `--output` files, exit code |
| Service | `dataeval-flow serve`  | HTTP client; polls             | A process per run    | `<output>/runs/<id>/`, HTTP |

## Library: the pipeline inside your program

`run_tasks()` executes a pipeline in the calling process and returns each task's
result object. Results return as Python objects, with no process startup and no files
written unless the caller asks. The caller decides when to run and what to keep and
where. This mode suits notebooks and applications that hold their data in memory. A
run shares the caller's process: memory it allocates stays allocated there, and a
crash in a native library terminates the caller.

The other modes compute through this one: the command calls `run_tasks()`, and the
service runs the command.

## Command: one run, then exit

The batch command, `dataeval-flow`, reads a configuration from `--config` or finds one
at the data root. It runs the configured tasks, writes each task's result files to
`--output` as the task finishes, and exits with a code that reports the run's outcome.
It is the headless counterpart of the interactive `dataeval-flow app`. A CI job, a
scheduler or a person starts it and waits for it to exit. The run has its own process:
the operating system reclaims its memory on exit, and the caller reads only the exit
code and the result files. The DataEval Flow container image runs this command by
default (see {doc}`../reference/containers`).

## Service: runs queued over HTTP

The service, `dataeval-flow serve`, accepts pipelines over HTTP and returns a run ID
immediately. It suits callers that cannot block on a run, such as a user interface or
an application that submits work for many users. A run outlives the request that
queued it; clients poll for its status and results.

### Each run is the batch command

The service does not compute pipelines itself. It queues each one and, when the run's
turn comes, starts the batch command on it in a separate process.

A run therefore produces the same results, files and exit code as the same pipeline
run from the command line. A run that exhausts memory or crashes in a native library
ends its own process, and the service keeps running. The operating system reclaims
the run's memory on exit, so each run starts clean. Cancellation stops the run's
process group, including any processes the run started, without the run's
cooperation.

The service adds a queue, a durable record of every run, and an HTTP interface for
submitting runs and reading their result files, logs and exit codes.

### A run is defined by its snapshot

A queued run may start minutes later or after a service restart. When it queues a
run, the service records a **snapshot**: the submitted pipeline with every default
filled in, and the resolved task list. The run reads only the snapshot; the service's
environment settings do not reach it. The same snapshot on the same data gives the
same result regardless of the service's configuration. The snapshot extends
{doc}`Reproducibility` to queued runs and is stored with the run's results as part of
its {doc}`Provenance`.

### One run at a time

Memory limits what a run can hold (see the
[hardware guidance](../reference/containers.md#recommended-minimum-hardware)), and
concurrent runs compete for it. The service runs one at a time, oldest first, so the
memory available to a run is independent of the queue. The limit is fixed: no setting
raises it, and `/v1/capabilities` reports it as `max_active_runs: 1`.

### Interrupted runs are not resumed

On shutdown, the service stops the running run and marks it `interrupted`. Queued runs
stay queued, and start, oldest first, when the service starts again. Interrupted runs
are not resumed: Flow's results come from whole tasks, and a partial task leaves no
result to continue from. Re-queuing the snapshot gives the same result, and the
[disk cache](../how_to/reuse_results_with_cache.md) reduces the cost of the repeat.

## Choosing a mode

- Use the library when the pipeline is part of a Python program or the data is
  already in memory.
- Use the command when something waits for the run to end: CI, a scheduler, a script
  or a terminal session. Its exit code can fail a CI job.
- Use the service when the caller does not wait, such as a user interface or a system
  that submits work and polls for results.

The service opens a port and has no authentication. Deploy it on a trusted network,
behind existing access controls.

## Related pages

- {doc}`Reproducibility`: why the configuration alone defines a run
- {doc}`Provenance`: what a result records about how it was produced
- {doc}`../how_to/run_flow_as_a_service`: set the service up and use it
- {doc}`../reference/service`: its endpoints, run states, files and probes
- {doc}`../reference/containers`: the container's mounts, variables and commands
