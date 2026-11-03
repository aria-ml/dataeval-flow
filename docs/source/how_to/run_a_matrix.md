# Sweep settings with a matrix

A task runs its entry once, with the settings the entry holds. To see how a finding moves with a setting, give the task
a `matrix:`. The task then runs its entry once per combination of the values the matrix lists, and returns one result
that compares the runs in a table. A matrix can vary any setting of the workflow, evaluator or
[custom workflow](write_a_custom_workflow.md) a task runs, the settings of the entries that workflow reads, and the
task's own `sources` and `extractor`. The examples here sweep a [`quality`](../reference/presets.md#quality) entry. You
need a pipeline with a `train` source and the extractor `bovw_ext`.

A matrix replaces the `parameter-sweep` workflow type.
[Move from a parameter sweep to a matrix](move_from_parameter_sweep.md) moves a sweep over.

## Run a task once per value

Write the `quality` entry as you would for one run, and list the values to try on its task:

```yaml
workflows:
  - name: cleaning
    type: quality
    outliers:
      flags: [dimension, pixel, visual]
      outlier_threshold: adaptive

tasks:
  - name: clean
    workflow: cleaning
    sources: train
    matrix:
      outliers.outlier_threshold: [[adaptive, 2.5], [adaptive, 3.5], [adaptive, 4.5]]
```

Keep three points in mind:

- The entry must be valid on its own, as every entry must. `quality` has no default for `outliers.flags` or
  `outliers.outlier_threshold`, so the entry sets both, even in a matrix that varies them. Where the entry sets a value
  the matrix varies, each run replaces it. [Configure outlier detection](configure_outlier_detection.md) covers the
  outlier settings most worth varying.
- The task keeps its name. `--task clean` runs it, and the results hold one entry for it, under `clean`.
- Each run's configuration is checked when the config loads, as a task's is. A value the setting refuses fails the load,
  and the message names every run that fails. `outliers.flags: [[pixel], [bogus]]` fails with:

  ```text
  Task 'clean' has a matrix that can't run:
  - run 2 (outliers.flags=[bogus]): Input should be 'dimension', 'pixel' or 'visual'
  ```

## Run the matrix

```bash
dataeval-flow -c pipeline.yaml -o out/
```

The runs go in order, each logging `Task 'clean': run 2/3 (outliers.outlier_threshold=[adaptive, 3.5])` as it starts,
which `-vv` shows. The console prints the matrix's report. Below its banner, on a 1,000-image sample of the
[MilitaryVehicles](../notebooks/tune_data_cleaning.py) dataset with `seed: 42`:

```text
  Health: 9 warnings [!!] across 3 runs — review flagged findings

  #  outliers.outlier_threshold  Health  Image Outliers  Class Outliers  Image Duplicates  Class Imbalance
  -  --------------------------  ------  --------------  --------------  ----------------  ------------------
  1  [adaptive, 2.5]         [!!]    [!!] 157        [!!] worst: 2S19    [!!] 4            [..] 24 classes,
                                         images (15.7%)  MSTA (25.0%),       exact             1000 items,
                                                         23/24 classes over  (0.4%), 2         imbalance 3.0:1
                                                         3.0%                near
                                                                             (0.2%)

  2  [adaptive, 3.5]         [!!]    [!!] 112        [!!] worst:         [!!] 4            [..] 24 classes,
                                         images (11.2%)  Tornado (20.4%),    exact             1000 items,
                                                         23/24 classes over  (0.4%), 2         imbalance 3.0:1
                                                         3.0%                near
                                                                             (0.2%)

  3  [adaptive, 4.5]         [!!]    [!!] 96 images  [!!] worst:         [!!] 4            [..] 24 classes,
                                         (9.6%)          Tornado (16.3%),    exact             1000 items,
                                                         23/24 classes over  (0.4%), 2         imbalance 3.0:1
                                                         3.0%                near
                                                                             (0.2%)
```

The table has a row per run and a column per finding. Raising the threshold from 2.5 to 4.5 flags 61 fewer images, and
changes which class has the most outliers. The duplicates and the label distribution do not move, since the threshold
does not reach them. Each run has 3 warnings, so the matrix has 9.

With `-v`, the console adds each run's full report below the table, under `RUNS`: the report the task would print run
alone with those values, headed with the run's number and values, `Run 1 · outliers.outlier_threshold=[adaptive, 2.5]`.
`out/results/result.txt` and the HTML report hold the table and every run's report, or only the table under
`result: detail: summary`. In the HTML report, each run's report follows the table as a report of its own, with the
thumbnails of the items it names. [Read evaluation outputs](read_evaluation_outputs.md) describes the report and its
severities, and [Read the result envelope](read_the_result_envelope.md) describes the JSON.

From Python, the task's result is a `MatrixResult`, and each run's result is the one the task would return alone:

```python
from pathlib import Path

from dataeval_flow import MatrixResult, load_config, run_tasks
from dataeval_flow.steps import ChainResult

result = run_tasks(load_config(Path("pipeline.yaml")), tasks="clean", data_dir=Path("."))["clean"]
assert isinstance(result, MatrixResult)
for run in result.runs:
    assert isinstance(run.result, ChainResult)  # quality is a preset, so each run is a chain
    print(run.number, run.label, run.result.health["status"])
```

## Write the values

Each key takes a list of values, or a range. A bare value fails the load: `outliers.n_clusters: 3` says
`` `outliers.n_clusters` takes a list of values, such as `[3]`, or a range `{from, to, step}`; got 3 ``.

### Write a list

A list holds one run's value per item, written as the setting would be written, `null` included. To vary a setting that
is itself a list, write a list of lists: `outliers.flags: [[pixel], [pixel, visual]]` is two runs.

### Write a range

A range `{from, to, step}` counts up from `from` by `step`, and includes `to` where a step lands on it. It counts in
decimal, so `{from: 2.5, to: 4.5, step: 0.5}` is `2.5, 3.0, 3.5, 4.0, 4.5`, never `3.0000000004`. The values are
integers when all three bounds are integers, and floats otherwise: `{from: 2, to: 10, step: 4}` is `2, 6, 10`, and
`{from: 0, to: 1, step: 0.3}` is `0.0, 0.3, 0.6, 0.9`. `step` must be above 0, and `from` no larger than `to`. A range
with another key, or a bound that is `true`, `false`, `.inf` or `.nan`, fails the load.

### Cross keys in one grid

One grid crosses every key with every other. The runs go as nested loops over the keys in the order written, the last
key changing fastest:

```yaml
tasks:
  - name: clean
    workflow: cleaning
    sources: train
    extractor: bovw_ext
    matrix:
      outliers.cluster_threshold: {from: 2.5, to: 4.5, step: 0.5}
      checks.image-outliers.warning: [3.0, 10.0]
```

That is 5 × 2 = 10 runs: run 1 is `outliers.cluster_threshold=2.5, checks.image-outliers.warning=3.0`, run 2 is
`outliers.cluster_threshold=2.5, checks.image-outliers.warning=10.0`, and so on. Each run's label, in the table, the
logs and the JSON, names its grid's keys and values in the order written.

### Run several grids

Several grids, a list of them, run in turn, each grid's runs together, numbered from 1 across the matrix. Write grids
when one setting means something under one value of another and nothing under another. `kmeans` needs a cluster count,
and `hdbscan` finds its own. In one grid, every algorithm would run at every count, and `hdbscan` would run three times
over with the same result. Two grids give each algorithm its own counts:

```yaml
tasks:
  - name: clean
    workflow: cleaning
    sources: train
    extractor: bovw_ext
    matrix:
      - outliers.cluster_algorithm: [kmeans]
        outliers.n_clusters: {from: 2, to: 4, step: 1}
      - outliers.cluster_algorithm: [hdbscan]
```

That is 3 + 1 = 4 runs. A grid need not set every key; where it does not, its runs' cells for that key are blank in the
table, and the entry's own value holds.

### Know what the load refuses

The load refuses:

- Duplicate runs, two runs that set the same settings to the same values once validated, whatever the order of their
  keys. On a float setting, `outliers.cluster_threshold: [3, 3.0]` is a duplicate:
  `runs 1 and 2 (outliers.cluster_threshold=3.0) set the same settings; drop one`. So is one combination two grids both
  reach.
- Two keys in one grid where one sets part of the other, such as `checks` and `checks.image-outliers.warning`, since
  which one wins would be a guess. In separate grids they are fine.
- An empty matrix, a grid with no keys, and a key with an empty list.

A key is a dotted path. [Name the setting a matrix key varies](name_a_matrix_key.md) covers the settings of other
entries, of the task's sources and extractor, and of the items of a list.

## Read the comparison

The report opens with the matrix's health and the table:

| Part | What it shows |
| --- | --- |
| Health | The worst of the runs: `failed` where a run failed, else `warning` where one warned, else `ok`. The warnings add up across the runs. |
| Rows | One per run: its number, its value for each key (blank where its grid sets none), and its health: `[ok]`, `[!!]` where it warned, or `failed`. |
| Columns | One per finding, in the order the findings first appear across the runs. Where two steps make findings with one title, each header adds its step's name; where one step makes two, the second is numbered, `Outliers (2)`. |
| Cells | The finding's severity marker and its brief: `[!!]` a warning, `[..]` info and `[ok]` ok. A run that made no such finding shows an em dash. |

### Read a failed run

A failed run is kept, and the runs after it still run. Its row says `failed`, its error is listed under the table as
`Run 3 failed: …`, and the matrix fails: the health line names the failed runs, and the command exits 1. The table and
every run that finished are still printed and written.

### Read an evaluator's task

An evaluator's task has no finding columns. An evaluator measures, and judges nothing, so its table lists the runs,
their values and whether each ran (`[ok]` or `failed`), its health line says the runs ran, and each run's report shows
its output. To compare runs by their findings, vary a workflow that judges:

- a preset's settings, such as `detectors.knn.k` on a `shift` task, whose checks judge each detector;
- or a custom workflow that runs the evaluator and a check on it, varying `evaluators.knn.k` as in the "Reach the
  entries a custom workflow runs" section of [Name the setting a matrix key varies](name_a_matrix_key.md).

## Check what the runs share

The runs differ only in what the matrix varies:

- Each source is resolved and its view drawn once, before the first run, and every run reads that draw. A view with an
  unseeded `Shuffle` gives every run the same items, so no run looks better for having sampled other ones. With the
  pipeline's `seed:`, the draw is the one a lone task makes. Where `sources` is a key, each source is still drawn once.
- The pipeline's `seed:` is applied again before each run, so no run's result depends on the runs before it. Without a
  `seed:`, nothing is reseeded.
- Statistics, embeddings and clusters are computed once. They come from the cache, which the runs share in memory, and
  on disk with `--cache`. A threshold matrix computes the statistics once, the clusters once per algorithm and cluster
  count, and the embeddings once per extractor and source, then re-judges them in each run.
  [Reuse results with the disk cache](reuse_results_with_cache.md) covers `--cache`.
- A stateful extractor is fitted once per `sources` value. BoVW learns its vocabulary from the data it embeds. The runs
  that read the same sources share one fit, made as a lone task would make it. A matrix over the extractor's own
  settings fits once per setting, as separate extractors would.
- A custom workflow's `view` step is not shared. It draws inside each run, after that run's seed.

Apart from its timestamp and duration, each run's result is the result of the same configuration run as a lone task. Its
`metadata.resolved_config` records that configuration, so each run can be read and reproduced on its own. The runs go
one at a time, and every run's result is held until the matrix ends.

## Check the limits

- The TUI keeps a task's matrix when it edits the task, and shows a matrix's result, but it does not edit matrices.
  Write them in YAML or Python.
- Renaming or deleting a source or extractor in the TUI does not rename it inside a matrix. The next load refuses the
  old name, such as `` `sources` value tran names no source `tran` ``.
- A matrix runs the whole task per combination. It does not fan out one step inside a custom workflow.
- There is no limit on the number of runs, and they do not run in parallel.

## Next steps

- [Tune data cleaning with a matrix](../notebooks/tune_data_cleaning.py) — Choose `quality` thresholds on
  MilitaryVehicles from a matrix's table.
- [Name the setting a matrix key varies](name_a_matrix_key.md) — Vary the sources, the extractor and the settings of
  other entries.
- [Gate a matrix run and read its files](gate_a_matrix_run.md) — Set the exit code, export a dataset for each run and
  read the matrix in the result files.
