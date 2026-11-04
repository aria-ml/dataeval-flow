# Run a single evaluator

Run one DataEval evaluator on a source, such as finding duplicates, without configuring a workflow. The result is
DataEval's own output, with no health status. This guide runs [`duplicates`](../reference/evaluators.md#duplicates), and
[Workflows and Evaluators](../concepts/WorkflowsAndEvaluators.md) explains when to use an evaluator instead of a
workflow. You need a config that defines `datasets:` and `sources:`.

## List the evaluators

```bash
dataeval-flow evaluators
```

Each line names an evaluator type, what it consumes, and how many sources its task takes. The command
`dataeval-flow evaluators duplicates` prints that evaluator's parameters. The
[Evaluator Catalog](../reference/evaluators.md) lists the same for every type.

## Define the evaluator

Add an entry under `evaluators:`, next to your datasets and sources:

```yaml
evaluators:
  - name: dupes
    type: duplicates
    flags: [hash_basic, hash_d4]
```

Parameter names are DataEval's own. A misspelled parameter fails the config load rather than being ignored.

## Add a task that runs it

A task names `evaluator:` instead of `workflow:`, never both:

```yaml
tasks:
  - name: find_dupes
    evaluator: dupes
    sources: [train, test]
```

`duplicates` searches every source you name together, so the task above also finds images duplicated between `train` and
`test`. The config load refuses a task that names the wrong number of sources or leaves out an extractor its evaluator
needs.

### Search for near duplicates in embedding space

To add embedding-space near duplicates, set `cluster_sensitivity` and name an extractor. Cluster mode reads one source:

```yaml
extractors:
  - name: bovw_ext
    model: bovw
    vocab_size: 512
    batch_size: 32

evaluators:
  - name: dupes_cluster
    type: duplicates
    cluster_sensitivity: 1.0

tasks:
  - name: find_near_dupes
    evaluator: dupes_cluster
    sources: train
    extractor: bovw_ext
```

BoVW needs no model file, so it works out of the box. An ONNX model is the higher-fidelity choice once you have one: see
[Use an ONNX model for embeddings](../notebooks/onnx_embeddings.py).

### Compare sources by position

An evaluator that compares sources reads them by position. A drift check takes the reference first, then the data to
test:

```yaml
evaluators:
  - name: mmd
    type: drift-mmd

tasks:
  - name: drift_check
    evaluator: mmd
    sources: [train, operational]
    extractor: bovw_ext
```

`drift-wasserstein` takes three sources: the reference, an in-distribution validation set, then the data to test.

## Read the output

Run the pipeline as usual. In `results/result.json` the task's entry has `"kind": "evaluator"`, the result envelope
under `metadata`, and DataEval's output under `output`:

```json
{
  "kind": "evaluator",
  "metadata": {"evaluator": "duplicates", "dataeval": {"version": "1.1.1"}},
  "output": {
    "shape": "table",
    "columns": ["group_id", "level", "dup_type", "item_indices", "dataset_indices", "methods"],
    "rows": [{"group_id": 0, "level": "item", "dup_type": "exact", "item_indices": [0, 5, 0, 5],
              "dataset_indices": [0, 0, 1, 1], "methods": ["xxhash"]}]
  }
}
```

There is no `health` entry. `--fail-on-warning` never fails a run because of an evaluator, though an evaluator that
fails to run still fails it.

## Run it from Python

The same run from Python returns an `EvaluatorResult`. Like a workflow's `ChainResult`, it is a `Result`: `report()`,
`to_dict()` and `export()` work the same way on both, and `output` holds what the run produced. `run_tasks` returns each
task's result keyed by the task's name:

```python
from pathlib import Path

from dataeval_flow import load_config, run_tasks
from dataeval_flow.evaluators.quality import DuplicatesResult

config = load_config(Path("config.yaml"))
result = run_tasks(config, tasks="find_dupes", data_dir=Path("."))["find_dupes"]

print(result.report())  # the full report: what -v prints and result.txt holds
rows = result.to_dict()["output"]["rows"]  # the JSON-ready table
assert isinstance(result, DuplicatesResult)
native = result.output  # DataEval's own DuplicatesOutput
```

`run` runs an evaluator on a dataset already in memory, with no config file, and returns the config's own result class:

```python
from dataeval_flow import run
from dataeval_flow.evaluators.quality import DuplicatesConfig

result = run(DuplicatesConfig(), dataset)  # a DuplicatesResult; `dataset` is any AnnotatedDataset
```

`result.output` is the object DataEval returned, so its own methods work as DataEval documents them. For Duplicates,
`native.aggregate_by_image()` groups the rows by image. `isinstance(result, DuplicatesResult)` types it. Reading
`output` on a failed run raises an error that names what went wrong, so check `result.success` first. The Evaluator
Catalog links each evaluator to its DataEval reference page.

## Next steps

- [Workflows and Evaluators](../concepts/WorkflowsAndEvaluators.md) — Choose between an evaluator and a workflow.
- [Read the result envelope](read_the_result_envelope.md) — Read the envelope that evaluators and workflows share.
- [Find regions the data does not cover](find_uncovered_regions.md) — Run an evaluator that needs an extractor, and read
  its output.
