# Run a Single Evaluator

Run one DataEval evaluator on a source, such as finding duplicates, without
configuring a workflow. The result is DataEval's own output, with no health verdict.

## 1. See what is available

```bash
dataeval-flow evaluators
```

Each line names an evaluator type, what it consumes, and how many sources its task
takes. `dataeval-flow evaluators quality.duplicates` prints that evaluator's
parameters.

## 2. Define the evaluator

Add an entry under `evaluators:`, next to your datasets and sources:

```yaml
evaluators:
  - name: dupes
    type: quality.duplicates
    flags: [hash_basic, hash_d4]
```

Parameter names are DataEval's own. A misspelled parameter fails the config load
rather than being ignored.

## 3. Add a task that runs it

A task names `evaluator:` instead of `workflow:`, never both:

```yaml
tasks:
  - name: find_dupes
    evaluator: dupes
    sources: [train, test]
```

`quality.duplicates` searches every source you name together, so the task above
also finds images duplicated between `train` and `test`.

To add embedding-space near duplicates, set `cluster_sensitivity` and name an
extractor. Cluster mode reads one source:

```yaml
extractors:
  - name: bovw_ext
    model: bovw
    vocab_size: 512
    batch_size: 32

evaluators:
  - name: dupes_cluster
    type: quality.duplicates
    cluster_sensitivity: 1.0

tasks:
  - name: find_near_dupes
    evaluator: dupes_cluster
    sources: train
    extractor: bovw_ext
```

BoVW needs no model file, so it works out of the box; an ONNX model is the
higher-fidelity choice once you have one — see
[Use an ONNX model for embeddings](../notebooks/onnx_embeddings.py).

The config refuses, when it loads, a task that names the wrong number of sources or
leaves out an extractor its evaluator needs.

## 4. Read the output

Run the pipeline as usual. In `results/result.json` the task's entry has
`"kind": "evaluator"`, the result envelope under `metadata`, and DataEval's output
under `output`:

```json
{
  "kind": "evaluator",
  "metadata": {"evaluator": "quality.duplicates", "dataeval": {"version": "1.1.1"}},
  "output": {
    "shape": "table",
    "columns": ["group_id", "level", "dup_type", "item_indices", "dataset_indices", "methods"],
    "rows": [{"group_id": 0, "level": "item", "dup_type": "exact", "item_indices": [0, 5, 0, 5],
              "dataset_indices": [0, 0, 1, 1], "methods": ["xxhash"]}]
  }
}
```

There is no `health` entry. `--fail-on-warning` never fails a run because of an
evaluator, though an evaluator that fails to run still fails it.

## 5. From Python

The same run from Python returns an `EvaluatorResult`. Like a workflow's
`WorkflowResult`, it is a `Result`: `report()`, `to_dict()` and `export()` work the
same way on both, while `output` and `raw` belong to the evaluator result alone:

```python
from pathlib import Path

from dataeval_flow import load_config, run_tasks

config = load_config(Path("config.yaml"))
(result,) = run_tasks(config, tasks="find_dupes", data_dir=Path("."))

print(result.report())  # the text the CLI prints
rows = result.output["rows"]  # the JSON-ready table
native = result.raw  # DataEval's own DuplicatesOutput
```

`result.raw` is the object DataEval returned, so its own methods work as DataEval
documents them — for Duplicates, `native.aggregate_by_image()` groups the rows by
image. The [Evaluator Catalog](../reference/evaluators.md) links each evaluator to its
DataEval reference page.

## See also

- [Workflows and Evaluators](../concepts/WorkflowsAndEvaluators.md) — when to use an
  evaluator instead of a workflow
- [Evaluator Catalog](../reference/evaluators.md) — every evaluator and parameter
- [Read evaluation outputs](read_evaluation_outputs.md) — the result envelope both
  kinds share
