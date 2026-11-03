# Fail a CI job on warnings

By default a run exits `0` whenever every task ran, whatever its findings say. A warning is a prompt to look, not a
failure. To let a CI job gate on data quality, make the findings that passed their checks' thresholds fatal with
`--fail-on-warning`, or with `fail_on: warning` in the pipeline's `result:` block.
[Read evaluation outputs](read_evaluation_outputs.md) explains the severities and the health line that the gate reads,
and the [Container Reference](../reference/containers.md) lists every exit code.

## Fail the run on warnings

Pass `--fail-on-warning` to the command:

```bash
docker run --rm \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,source="$(pwd)/data",target=/dataeval,readonly \
    --mount type=bind,source="$(pwd)/workspace/output",target=/output \
    harbor.jatic.net/aria/dataeval-flow:latest-cpu \
    python -m dataeval_flow --output /output --fail-on-warning
```

## Read the exit code

| Exit code | Meaning |
| --- | --- |
| `0` | Every task succeeded, with no warnings the gate fails on; or `fail_on: never` with no `require` |
| `1` | A task failed, or an export could not be written |
| `2` | The command line was mistyped: an unknown flag, or a value it refuses |
| `3` | Every task succeeded, but a task raised warnings and the gate fails on them |
| `4` | A task's verdict is worse than `result: require:` or `--require` |

When more than one applies, `1` comes first, then `4`, then `3`. Exit `4` gates on an `audit`'s verdict. See
[Gate training on an audit](gate_training_on_an_audit.md#read-the-verdict).

`fail_on` in the `result:` block sets the gate: `failure` (the default), `warning`, or `never`, which fails the job on
neither a failed task nor a warning, though a set `require` still exits `4`. `--fail-on-warning` and
`--no-fail-on-warning`, or `DATAEVAL_FAIL_ON_WARNING`, override it.

## Gate a CI job

Since a warning exits `3`, a job can tell a data-quality gate from a crash or a mistyped flag:

```yaml
# .gitlab-ci.yml
data-quality:
  script:
    - python -m dataeval_flow --output output
  allow_failure:
    exit_codes: [3]     # a warning marks the job, a failed task fails the pipeline
  artifacts:
    when: always
    paths: [output/results/]
    reports:
      junit: output/results/result.xml
```

The job assumes `fail_on: warning` and `formats: [json, html, junit, markdown]` in the pipeline's `result:` block. With
`per_task: true`, each task writes its own report, so `junit:` takes a glob, `output/results/result-*.xml`. The
"Choose the result files" section of [Run workflows in containers](containerized_workflows.md) lists every key of the
`result:` block.

The JUnit report makes each task a test suite and each finding a test case, failing where the finding is a warning, so
the merge request's test view lists the checks the data did not pass. The Markdown summary is each task's findings as a
table, ready for `$GITHUB_STEP_SUMMARY` or a merge-request comment. Both also name any task that failed.

Results are written either way. The gate decides the exit code, not whether the run's artifacts survive.

## Next steps

- [Gate training on an audit](gate_training_on_an_audit.md) — Gate a run on an audit's verdict with `--require`.
- [Read evaluation outputs](read_evaluation_outputs.md) — Read the severities and the health line behind a warning.
- [Execution Modes: Library, Command and Service](../concepts/ExecutionModes.md) — Learn when to run the batch command,
  which a CI job runs, and when to run the service.
