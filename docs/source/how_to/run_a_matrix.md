# Sweep settings with a matrix

A task runs its entry once, with the settings the entry holds. To see how a finding moves with a setting, give the task
a `matrix:`. The task then runs its entry once per combination of the values the matrix lists, and returns one result
that compares the runs in a table. A matrix can vary any setting of the workflow, evaluator or custom workflow a task
runs, the settings of the entries that workflow reads, and the task's own `sources` and `extractor`.

A matrix replaces the `parameter-sweep` workflow type. [Section 7](#7-move-from-parameter-sweep) moves a sweep over.

## 1. Run a task once per value

Write the `data-cleaning` entry as you would for one run, and list the values to try on its task:

```yaml
workflows:
  - name: cleaning
    type: data-cleaning
    outlier_method: adaptive
    outlier_flags: [dimension, pixel, visual]

tasks:
  - name: clean
    workflow: cleaning
    sources: train
    matrix:
      outlier_threshold: [2.5, 3.5, 4.5]
```

Notice:

- The entry must be valid on its own, as every entry must. `data-cleaning` has no default for `outlier_method` or
  `outlier_flags`, so the entry sets both, even in a matrix that varies them. Where the entry sets a value the matrix
  varies, each run replaces it.
- The task keeps its name. `--task clean` runs it, and the results hold one entry for it, under `clean`.
- Each run's configuration is checked when the config loads, as a task's is. A value the setting refuses fails the
  load, and the message names every run that fails. `outlier_threshold: [-1.0, 3.0]` fails with:

  ```text
  Task 'clean' has a matrix that can't run:
  - run 1 (outlier_threshold=-1.0): Input should be greater than or equal to 0
  ```

Run it:

```bash
dataeval-flow -c pipeline.yaml -o out/
```

The runs go in order, each logging `Task 'clean': run 2/3 (outlier_threshold=3.5)` as it starts, which `-vv` shows.
The console prints the matrix's report. Below its banner, on a 1,000-image sample of the
[MilitaryVehicles](../notebooks/tune_data_cleaning.py) dataset with `seed: 42`:

```text
  Health: 9 warnings [!!] across 3 runs — review flagged findings

  #  outlier_threshold  Health  Image Outliers  Classwise Outliers  Duplicates  Class Imbalance
  -  -----------------  ------  --------------  ------------------  ----------  ------------------
  1  2.5                [!!]    [!!] 157        [!!] worst: 2S19    [!!] 4      [..] 24 classes,
                                images (15.7%)  MSTA (25.0%),       exact       1000 items,
                                                23/24 classes over  (0.4%), 2   imbalance 3.0:1
                                                3.0%                near
                                                                    (0.2%)

  2  3.5                [!!]    [!!] 112        [!!] worst:         [!!] 4      [..] 24 classes,
                                images (11.2%)  Tornado (20.4%),    exact       1000 items,
                                                23/24 classes over  (0.4%), 2   imbalance 3.0:1
                                                3.0%                near
                                                                    (0.2%)

  3  4.5                [!!]    [!!] 96 images  [!!] worst:         [!!] 4      [..] 24 classes,
                                (9.6%)          Tornado (16.3%),    exact       1000 items,
                                                23/24 classes over  (0.4%), 2   imbalance 3.0:1
                                                3.0%                near
                                                                    (0.2%)
```

The table has a row per run and a column per finding. Raising the threshold from 2.5 to 4.5 flags 61 fewer images,
and changes which class has the most outliers. The duplicates and the label distribution do not move, since the
threshold does not reach them. Each run has 3 warnings, so the matrix has 9.

With `-v`, the console adds each run's full report below the table, under `RUNS`: the report the task would print run
alone with those values, headed with the run's number and values, `Run 1 · outlier_threshold=2.5`.
`out/results/result.txt` and the HTML report hold the table and every run's report, or only the table under
`result: detail: summary`. In the HTML report, each run's report follows the table as a report of its own, with the
thumbnails of the items it names.

From Python, the task's result is a `MatrixResult`, and each run's result is the one the task would return alone:

```python
from pathlib import Path

from dataeval_flow import MatrixResult, load_config, run_tasks
from dataeval_flow.steps import ChainResult

result = run_tasks(load_config(Path("pipeline.yaml")), tasks="clean", data_dir=Path("."))["clean"]
assert isinstance(result, MatrixResult)
for run in result.runs:
    assert isinstance(run.result, ChainResult)  # data-cleaning is a preset, so each run is a chain
    print(run.number, run.label, run.result.health["status"])
```

## 2. Write the values

Each key takes a list of values, or a range. A bare value fails the load: `outlier_threshold: 3.0` says
`` `outlier_threshold` takes a list of values, such as `[3.0]`, or a range `{from, to, step}`; got 3.0 ``.

**A list** holds one run's value per item, written as the setting would be written, `null` included. To vary a
setting that is itself a list, write a list of lists: `outlier_flags: [[pixel], [pixel, visual]]` is two runs.

**A range** `{from, to, step}` counts up from `from` by `step`, and includes `to` where a step lands on it. It counts
in decimal, so `{from: 2.5, to: 4.5, step: 0.5}` is `2.5, 3.0, 3.5, 4.0, 4.5`, never `3.0000000004`. The values are
integers when all three bounds are integers, and floats otherwise: `{from: 2, to: 10, step: 4}` is `2, 6, 10`, and
`{from: 0, to: 1, step: 0.3}` is `0.0, 0.3, 0.6, 0.9`. `step` must be above 0, and `from` no larger than `to`. A
range with another key, or a bound that is `true`, `false`, `.inf` or `.nan`, fails the load.

**One grid** crosses every key with every other. The runs go as nested loops over the keys in the order written, the
last key changing fastest:

```yaml
tasks:
  - name: clean
    workflow: cleaning
    sources: train
    matrix:
      outlier_threshold: {from: 2.5, to: 4.5, step: 0.5}
      checks.image-outliers.warning: [3.0, 10.0]
```

That is 5 × 2 = 10 runs: run 1 is `outlier_threshold=2.5, checks.image-outliers.warning=3.0`, run 2 is
`outlier_threshold=2.5, checks.image-outliers.warning=10.0`, and so on. Each run's label, in the table, the logs
and the JSON, names its grid's keys and values in the order written.

**Several grids**, a list of them, run in turn, each grid's runs together, numbered from 1 across the matrix. Write
grids when one setting means something different under another. An outlier threshold counts standard deviations under
`zscore` and `modzscore`, and interquartile ranges beyond the quartiles under `iqr`, where 1.5 is the usual fence. In
one grid, every method would run at every threshold, `iqr` at 4 and `zscore` at 1.5 among them. Two grids give each
method its own thresholds:

```yaml
tasks:
  - name: clean
    workflow: cleaning
    sources: train
    matrix:
      - outlier_method: [zscore, modzscore]
        outlier_threshold: {from: 2, to: 4, step: 1}
      - outlier_method: [iqr]
        outlier_threshold: [1.5, 3.0]
```

That is 6 + 2 = 8 runs. A grid need not set every key; where it does not, its runs' cells for that key are blank in
the table, and the entry's own value holds.

The load refuses:

- **Duplicate runs**, two runs that set the same settings to the same values once validated, whatever the order of
  their keys. On a float setting, `outlier_threshold: [3, 3.0]` is a duplicate:
  `runs 1 and 2 (outlier_threshold=3.0) set the same settings; drop one`. So is one combination two grids both reach.
- **Two keys in one grid where one sets part of the other**, such as `checks` and
  `checks.image-outliers.warning`, since which one wins would be a guess. In separate grids they are fine.
- An empty matrix, a grid with no keys, and a key with an empty list.

## 3. Name the setting a key varies

A key is a dotted path, read as the first of these forms that matches:

| Key | Reaches |
| --- | --- |
| `sources` | The task's `sources`. A value is a source name or a list of them. |
| `extractor` | The task's `extractor`. A value is an extractor name, or `null` for none. |
| `evaluators.<name>.<path>` | A setting of the `evaluators:` entry `<name>` |
| `workflows.<name>.<path>` | A setting of the `workflows:` entry `<name>` |
| `extractors.<name>.<path>` | A setting of the `extractors:` entry `<name>` |
| `steps.<step>.<setting>` | A setting of a transform, combine or check step in the custom workflow the task runs |
| anything else | A path into the entry the task runs |

A path walks fields by name: `checks.image-outliers.warning` sets one field of `data-cleaning`'s
`checks`. The sections below give an example of each other form.

### List items by name

Where a path meets a list of entries, such as an `ood-detection` entry's `detectors`, it picks the item by its `name`.
An entry without a `name` is named by its type, so an unnamed detector is reached by its type:

```yaml
workflows:
  - name: ood
    type: ood-detection
    detectors:
      - {name: knn, type: ood-kneighbors, k: 10}
      - {type: ood-domain-classifier, n_folds: 5}

tasks:
  - name: ood_check
    workflow: ood
    sources: [train, operational]
    extractor: bovw_ext
    matrix:
      detectors.knn.k: [5, 10, 20]
      detectors.ood-domain-classifier.n_folds: [3, 5]
```

A path may end on a whole item, written whole, and the item keeps its name:
`detectors.knn: [{type: ood-kneighbors, k: 5}, {type: ood-kneighbors, k: 20, distance_metric: euclidean}]`. Items
are not reached by their position. A name no item has fails the load, listing the names there are:
`` `detectors` in ood-detection has no item named `knn2` (it has `knn`, `ood-domain-classifier`) ``.

### Keys of a mapping

A setting that maps names to values, such as an `outliers` evaluator's per-metric `outlier_threshold`, is walked by
key, and a key the mapping lacks is added for the run:

```yaml
evaluators:
  - name: outl
    type: outliers
    flags: [pixel, visual]
    outlier_threshold:
      brightness: [zscore, 3.0]
      contrast: [zscore, 3.0]

tasks:
  - name: outliers_only
    evaluator: outl
    sources: train
    matrix:
      outlier_threshold.brightness: [[zscore, 2.0], [zscore, 3.0], [iqr, 1.5]]
```

Each value is the whole threshold for `brightness`, a method and its bound, so the values are lists. A path through a
setting left unset fails the load, as there is nothing to walk into. Without `outlier_threshold` on the entry, the load
says `` `outlier_threshold` in outliers is unset, so `outlier_threshold.brightness` has nothing to set; vary
`outlier_threshold` whole ``.

### The task's sources and extractor

`sources` runs the task on each source, or list of sources, in turn. `extractor` runs it with each extractor:

```yaml
extractors:
  - name: resnet
    model: onnx
    model_path: ./models/resnet50-v2-7.onnx
    batch_size: 32

tasks:
  - name: clean_each
    workflow: cleaning
    sources: train
    matrix:
      sources: [train, validation, test]
  - name: ood_by_extractor
    workflow: ood
    sources: [train, operational]
    matrix:
      extractor: [bovw_ext, resnet]
```

A value that names no source or extractor fails the load. `ood_by_extractor` sets no `extractor:` of its own, which
is fine since every run has one: a task as written is not checked, only its runs.

### Another entry: a custom workflow's evaluators and steps

A custom workflow's evaluator steps run `evaluators:` entries, whose settings live in those entries. Vary them with
`evaluators.<name>.<path>`, and a transform's, combine's or check's settings with `steps.<step>.<setting>`:

```yaml
evaluators:
  - {name: knn, type: ood-kneighbors, k: 10}

workflows:
  - name: novelty
    inputs: [reference, test]
    steps:
      - {name: knn, evaluator: knn, input: [reference, test]}
      - {name: knn_check, check: ood, input: knn}

tasks:
  - name: novelty_check
    workflow: novelty
    sources: [train, operational]
    extractor: bovw_ext
    matrix:
      evaluators.knn.k: [5, 10, 20]
      steps.knn_check.warning: [5.0, 10.0]
```

`steps.` reaches a setting whether the step writes it or leaves it at its default, as `knn_check` leaves `warning` at
10.0. Changing an entry changes it for every step in the run that reads it: two steps naming `knn` both see the new
`k`.
`workflows.<name>.<path>` reaches a `workflows:` entry the same way, such as a `data-cleaning` entry a custom workflow
runs as a step. `steps.knn_check.warning` and `workflows.novelty.steps.knn_check.warning` are one key.

### An extractor's settings

`extractors.<name>.<path>` varies an extractor's settings:

```yaml
tasks:
  - name: ood_vocab
    workflow: ood
    sources: [train, operational]
    extractor: bovw_ext
    matrix:
      extractors.bovw_ext.vocab_size: [256, 512, 1024]
```

### What a key may not vary

A matrix varies settings. It may not change which entry is which, or how a chain is wired, since every other key
relies on them. The load refuses:

- an entry's `name` and `type` (an extractor's `name` and `model`, a custom workflow's `inputs`), and a list item's
  `name`: `` `type` varies `type`, which says which entry this is: a matrix varies settings, not identities ``;
- on a custom workflow's step, its kind (`evaluator`, `workflow`, `transform`, `combine`, `check`), `name`, `input`,
  `by`, `optional`, `pairs` and `extractor`;
- an `export` step's `to`, since each run already writes under a directory of its own (section 6);
- `steps.<step>.…` where the step runs an evaluator or workflow entry, with the key to use instead:
  `` `steps.knn.k`: step 'knn' runs the evaluator entry `knn`, whose settings live there: vary `evaluators.knn.k` ``;
- the settings of an extractor given as a Python object, not by its settings;
- a setting the entry does not have: `` data-cleaning has no setting `outlier_thresh` ``;
- the `datasets:`, `sources:`, `views:`, `preprocessors:`, `metadata:`, `stats:` and `ontologies:` pools. An entry's
  reference to one of them by name can vary, such as a `data-cleaning` entry's `stats:`.

### A key must reach every run

A key that names another entry must name one each of its runs reads: the entry the task runs, the entries its custom
workflow's steps run, and the extractors any of them use. On an `ood_compare` task running `ood`, the matrix
`{extractor: [bovw_ext, resnet], extractors.bovw_ext.vocab_size: [256, 512]}` means to compare two extractors and two
BoVW vocabularies. But the runs with `resnet` read no `bovw_ext`, so `vocab_size` would change nothing in them, and
the load refuses:

```text
Task 'ood_compare' has a matrix that can't run:
- run 3 (extractor=resnet, extractors.bovw_ext.vocab_size=256): the run reads no extractor `bovw_ext`, so
  `extractors.bovw_ext.vocab_size` would change nothing; give it a grid of its own with the runs that read it
- run 4 (extractor=resnet, extractors.bovw_ext.vocab_size=512): the run reads no extractor `bovw_ext`, so
  `extractors.bovw_ext.vocab_size` would change nothing; give it a grid of its own with the runs that read it
```

Two grids say what was meant, three runs:

```yaml
tasks:
  - name: ood_compare
    workflow: ood
    sources: [train, operational]
    matrix:
      - extractor: [bovw_ext]
        extractors.bovw_ext.vocab_size: [256, 512]
      - extractor: [resnet]
```

## 4. Read the comparison

The report opens with the matrix's health and the table:

- **Health.** The worst of the runs: `failed` where a run failed, else `warning` where one warned, else `ok`. The
  warnings add up across the runs.
- **Rows.** One per run: its number, its value for each key (blank where its grid sets none), and its health: `[ok]`,
  `[!!]` where it warned, or `failed`.
- **Columns.** One per finding, in the order the findings first appear across the runs. Where two steps make findings
  with one title, each header adds its step's name; where one step makes two, the second is numbered, `Outliers (2)`.
- **Cells.** The finding's severity marker and its brief: `[!!]` a warning, `[..]` info and `[ok]` ok. A run that
  made no such finding shows `—`.

**A failed run** is kept, and the runs after it still run. Its row says `failed`, its error is listed under the table
as `Run 3 failed: …`, and the matrix fails: the health line names the failed runs, and the command exits 1. The table
and every run that finished are still printed and written.

**An evaluator's task has no finding columns.** An evaluator measures, and judges nothing, so its table lists the runs,
their values and whether each ran (`[ok]` or `failed`), its health line says the runs ran, and each run's report shows
its output. To compare runs by a verdict, vary a workflow that judges:

- a preset's settings, such as `detectors.knn.k` on an `ood-detection` task, whose checks judge each detector;
- or a custom workflow that runs the evaluator and a check on it, varying `evaluators.knn.k` as in section 3.

## 5. What the runs share

The runs differ only in what the matrix varies:

- **One draw of each source.** Each source is resolved and its view drawn once, before the first run, and every run
  reads that draw. A view with an unseeded `Shuffle` gives every run the same items, so no run looks better for having
  sampled other ones. With the pipeline's `seed:`, the draw is the one a lone task makes. Where `sources` is a key,
  each source is still drawn once.
- **A fresh seed per run.** The pipeline's `seed:` is applied again before each run, so no run's result depends on the
  runs before it. Without a `seed:`, nothing is reseeded.
- **Statistics, embeddings and clusters, computed once.** They come from the cache, which the runs share in memory,
  and on disk with `--cache`. A threshold matrix computes the statistics once, the clusters once per algorithm and
  cluster count, and the embeddings once per extractor and source, then re-judges them in each run.
- **One fit of a stateful extractor per `sources` value.** BoVW learns its vocabulary from the data it embeds. The runs
  that read the same sources share one fit, made as a lone task would make it. A matrix over the extractor's own
  settings fits once per setting, as separate extractors would.
- **Not a custom workflow's `view` step.** It draws inside each run, after that run's seed.

Apart from its timestamp and duration, each run's result is the result of the same configuration run as a lone task.
Its `metadata.resolved_config` records that configuration, so each run can be read and reproduced on its own. The runs
go one at a time, and every run's result is held until the matrix ends.

## 6. Exports, gating and files

**Exports.** An `export` step inside a run writes under `datasets/<to>/run-<n>/`, where `n` is the run's number in
the table, and under `datasets/<to>/run-<n>/<key>/` where it writes a list, once per element. Numbers name the
directories, since a source list or a float does not always make a safe directory name; the run's values are in the
result. Its `provenance.json` names it `<to>/run-<n>`. Here each run writes its cleaned dataset, to
`out/datasets/clean_export.dataset/run-1/` and `run-2/`:

```yaml
workflows:
  - name: tidy
    inputs: [data]
    steps:
      - {name: cleaning, workflow: cleaning, input: data}
      - {name: dataset, transform: export, input: cleaning.clean, format: coco}

tasks:
  - name: clean_export
    workflow: tidy
    sources: train
    matrix:
      workflows.cleaning.outlier_threshold: [3.5, 4.5]
```

The runs cannot clash with each other, but another task writing to the same `to` still fails the load. A matrix may
not vary `to` itself: `steps.dataset.to` fails the load, since each run already writes under its own `run-<n>/`.

**Gating.** By default a failed run fails the command (exit 1), and warnings do not. `result: fail_on: warning`, or
`--fail-on-warning`, exits 3 when any run warns. For an exploratory matrix, where most runs are meant to warn,
`fail_on: never` keeps the exit code 0:

```yaml
result:
  fail_on: never
```

**JSON.** In `result.json`, the task's entry holds the matrix. Each run's `result` is its type's JSON, its `assets`
included: each run keeps the thumbnails of the items its own report names, and the matrix has none of its own. A chain
names an item by its Dataset's address, such as `data` for a preset's input, whichever source the task binds there.
Where the runs read other sources, or other items through a varied view, one address holds other images in each run.
`errors` is there only where a run failed. From the run in section 1, trimmed:

```json
{
  "clean": {
    "kind": "matrix",
    "type": "data-cleaning",
    "keys": ["outlier_threshold"],
    "metadata": {"source_descriptions": ["train (ds[sample])"], "model_id": null, "metadata_binning": null,
                 "resolved_config": {"task": {"name": "clean", "workflow": "cleaning", "sources": "train",
                                              "matrix": {"outlier_threshold": [2.5, 3.5, 4.5]}},
                                     "sources": ["train"], "seed": 42}},
    "health": {"status": "warning", "warnings": 9, "failed_runs": []},
    "runs": [
      {"number": 1, "label": "outlier_threshold=2.5", "values": {"outlier_threshold": 2.5},
       "result": {"kind": "workflow", "metadata": {}, "health": {}, "steps": {}, "findings": [], "assets": []}}
    ]
  }
}
```

The table is not stored: it reads off the runs. The matrix's `metadata` names `model_id` only where every run used
one extractor, and leaves the encoding fields `null`, since each run records its own.

**Other files.** A JUnit report has a test suite per run, named `clean · run 3`, its findings as cases. A Markdown
report gives the task's health, the table and the failed runs' errors. With `per_task: true`, a matrix task's files
hold its matrix result, as any task's hold its result.

**The encoding descriptor.** `encoding.json`, and `dataeval-flow encoding result.json`, describe a matrix task by its
runs' descriptor where they agree. Where its runs were encoded differently, the command refuses the task, naming the
runs that differ, and the run writes no `encoding.json`, as when two tasks disagree.

## 7. Move from parameter-sweep

`type: parameter-sweep` now fails the load as any unknown type does. Its fields become a `data-cleaning` entry and a
matrix on its task. A sweep such as:

```text
workflows:
  - name: sweep
    type: parameter-sweep
    outlier_threshold: [2.5, 3.5, 4.5]
    duplicate_cluster_sensitivity: [0.5, 2.0]
    duplicate_cluster_algorithm: [hdbscan]

tasks:
  - name: tune
    workflow: sweep
    sources: train
    extractor: bovw_ext
```

becomes:

```yaml
workflows:
  - name: cleaning_tune
    type: data-cleaning
    outlier_method: adaptive
    outlier_flags: [dimension, pixel, visual]
    duplicate_cluster_algorithm: hdbscan

tasks:
  - name: tune
    workflow: cleaning_tune
    sources: train
    extractor: bovw_ext
    matrix:
      outlier_threshold: [2.5, 3.5, 4.5]
      duplicate_cluster_sensitivity: [0.5, 2.0]
```

Notice:

- **Set `outlier_method` and `outlier_flags` on the entry.** The sweep defaulted them to `[adaptive]` and all three
  flag groups, and `data-cleaning` has no default for either.
- A field the sweep held one value for is a setting of the entry. A field with several values is a key of the matrix,
  `null` included where the sweep tried DataEval's default.
- The sweep's statistics ignored `value_range`. `data-cleaning` reads the dataset's `value_range`, so on float
  imagery that declares one, its statistics can differ from the sweep's.
- **On detection data, `data-cleaning` also judges boxes**, and its outlier counts include them, where the sweep
  counted whole images. On classification data, the count of images flagged and of near-duplicate groups match what
  the sweep reported for each combination, where `duplicate_merge_near` is left at its default: the sweep merged
  near-duplicate groups whatever it said.
- The sweep printed one pivot table per count. The matrix prints one table of every run's findings, and each run's
  full result is in `result.runs`, with every step's output.

The [Tune data cleaning with a matrix](../notebooks/tune_data_cleaning.py) tutorial runs this on MilitaryVehicles.

## 8. Limits

- The TUI keeps a task's matrix when it edits the task, and shows a matrix's result, but it does not edit matrices.
  Write them in YAML or Python.
- Renaming or deleting a source or extractor in the TUI does not rename it inside a matrix. The next load refuses the
  old name, such as `` `sources` value tran names no source `tran` ``.
- A matrix runs the whole task per combination. It does not fan out one step inside a custom workflow.
- There is no limit on the number of runs, and they do not run in parallel.

## See also

- [Tune data cleaning with a matrix](../notebooks/tune_data_cleaning.py): choose `data-cleaning` thresholds on
  MilitaryVehicles from a matrix's table
- [Configure outlier detection](configure_outlier_detection.md): the outlier settings most worth varying
- [Read evaluation outputs](read_evaluation_outputs.md): the report, its severities and the JSON
- [Reuse results with the disk cache](reuse_results_with_cache.md): keep statistics and embeddings across commands
- [Chain steps into a workflow of your own](write_a_custom_workflow.md): the custom workflows a matrix can vary
