# Monitor drift with steps

The `drift-monitoring` workflow type is a preset: a chain of steps that tests each of your test sources against a
reference, with a `drift` check judging each test. This guide shows how to read what it makes, and how to rebuild
parts of it as a custom workflow when you want something else: one merged test set, a comparison of classes or
groups, or drift on the objects in detection images. The [drift tutorial](../notebooks/drift_monitoring.py) runs the
preset on real imagery, and [Chain steps into a workflow of your own](write_a_custom_workflow.md) explains the step
syntax this guide uses.

## 1. Each source on its own

A task with a reference and two test sources runs every detector once for each test source:

```yaml
workflows:
  - name: drift
    type: drift-monitoring
    detectors:
      - {type: drift-univariate, method: ks, p_val: 0.01}
      - {name: mmd_chunked, type: drift-mmd, chunking: {chunk_count: 10, threshold: [zscore, 2.5]}}
      - {type: drift-kneighbors, k: 5}
    health_thresholds:
      drift: {warn_on_drift: true, chunk_percent: 10.0, consecutive_chunks: 3}

tasks:
  - name: cameras
    workflow: drift
    sources: [reference, cam1, cam2]
    extractor: bovw
```

The first source is the reference, and each later source is tested against it. The entries of `detectors:` are drift
evaluator entries, so each takes the fields its evaluator takes in the
[Evaluator Catalog](../reference/evaluators.md). An entry's `name` defaults to its type, and it names the detector's
step. Fitting on the reference is the cost of testing each source on its own, but the reference's embeddings are
derived once, and only the fit repeats for each source.

A chunked entry cuts each test source into chunks of its own, and tests each against the same bounds, which come
from the spread of the reference's chunks. The bounds follow the entry's `threshold`. Unset, they follow the
detector's default, which differs between detectors: the `drift-domain-classifier` judges its AUROC against a
constant band. Set `threshold: [zscore, 3.0]` to judge every chunked detector alike.

The report groups each source's findings under the source's name. From Python, the result is a `ChainResult`, and
each step holds one element per test source:

```python
result = run_task(task, config)

output = result.steps["drift-kneighbors"].elements["cam1"].output  # DataEval's DriftOutput
print(output.drifted, output.distance, output.threshold)

chunks = result.steps["mmd_chunked"].elements["cam1"].output.details  # a polars DataFrame, when chunked
finding = result.steps["mmd_chunked-check"].elements["cam1"].output[0]
print(finding.severity, finding.brief)  # warning, 3/10 chunks drifted
```

The `drift` check is the step `<detector>-check`. Unchunked, it warns where the detector found drift, or reports
`info` when `warn_on_drift` is false. Chunked, it warns where the share of drifted chunks reaches `chunk_percent` or
the longest run of drifted chunks reaches `consecutive_chunks`, and it reports `info` where some chunk drifted but
neither threshold is reached. A threshold of `null` judges nothing.

A detector that raises, such as a chunked one whose reference is too small to split into 3 chunks, fails its own step
and the task. The other detectors still run, and their findings are in the result.

## 2. One merged test set

Testing each source on its own tells you which source drifted. When you would rather ask whether the pooled test
data drifted, merge the sources and run one detector on the result. Where the sources name their classes
differently, conform them to one vocabulary first, as section 1 of
[Chain steps into a workflow of your own](write_a_custom_workflow.md) does:

```yaml
evaluators:
  - {name: mmd, type: drift-mmd}

workflows:
  - name: pooled_drift
    inputs: [reference, cam1, cam2]
    steps:
      - {name: pooled, transform: merge, input: [cam1, cam2]}
      - {name: drift, evaluator: mmd, input: [reference, pooled]}
      - {name: drift-check, check: drift, input: drift}

tasks:
  - name: cameras
    workflow: pooled_drift
    sources: [reference, cam1, cam2]
    extractor: bovw
```

`merge` concatenates its inputs in order, and `drift` tests the pooled Dataset against the reference. This is how the
workflow behaved before it tested each source on its own. The recipe names its sources in `inputs:`, since `merge`
reads a list of addresses.

## 3. By class, and by group

To see which classes drift, name detectors in the preset's `classwise:`. Each named detector also runs once per
class, unchunked, as the step `<detector>-classes`, judged by `<detector>-classes-check`:

```yaml
workflows:
  - name: drift
    type: drift-monitoring
    detectors:
      - {type: drift-mmd}
    classwise: [drift-mmd]
```

A class is tested only where the reference and the test source each hold at least 2 items of it. A class too small
for the detector, such as one with no more reference items than `drift-kneighbors`'s `k`, is left out with the
detector's error. The step lists a class it leaves out under `skipped`, with the reason, and the report names it. The by-class finding is one finding
for the source, titled `Drift (MMD) by class`, briefed such as `2/8 classes warn`, and it names the classes that
warned. In Python, `result.steps["drift-mmd-classes"].elements["cam1"].output` is a `PerClassOutput`: its `outputs`
holds each class's `DriftOutput`, by class name, and its `skipped` holds each class's reason. Detection data has no
single label per item, so the by-class step is skipped for it with that reason. Section 4 shows how to run it on
crops.

A custom workflow takes the same machinery on any evaluate step or check, with `by:`. `by: class` runs the step once
per class, and `groups:` runs it once per group of classes instead:

```yaml
evaluators:
  - {name: mmd, type: drift-mmd}

workflows:
  - name: vehicle_drift
    inputs: [reference, {name: tests, list: true}]
    steps:
      - name: mmd-groups
        evaluator: mmd
        input: [reference, tests]
        by:
          class:
            groups:
              vehicles: [car, truck, van]
              people: [person]
            min_items: 2
      - {name: mmd-groups-check, check: drift, input: mmd-groups, by: class}
```

A group lists classes by name or by index, and a name resolves through the first input's `index2label`. Groups may
overlap, and a class in no group is left out, and listed as skipped. `min_items` is the smallest number of items of
a key that each input must hold, and it defaults to 2. The check's `by: class` takes no settings, because its keys
come from the step it reads.

To compare one group of the reference with a different group of the test data, such as the cats in one against the
dogs in the other, narrow each side with a `view` step and run a detector on the two views:

```yaml
evaluators:
  - {name: mmd, type: drift-mmd}

workflows:
  - name: group_against_group
    inputs: [reference, {name: tests, list: true}]
    steps:
      - name: ref-cats
        transform: view
        input: reference
        operations: [{type: ClassFilter, params: {classes: [0]}}]
      - name: test-dogs
        transform: view
        input: tests
        operations: [{type: ClassFilter, params: {classes: [1]}}]
      - {name: drift, evaluator: mmd, input: [ref-cats, test-dogs]}
      - {name: drift-check, check: drift, input: drift}
```

`ClassFilter` takes class indexes. `test-dogs` and the steps after it run once for each test source.

## 4. Detections

A detection Dataset has several labels per image, so `by: class` cannot key it. `wrap` with DataEval's
`DetectionCrops` turns each box into a classification item labelled with the box's class, so a custom workflow can
run the preset on crops. Wrap the reference and the tests, and run the preset as a step:

```yaml
workflows:
  - name: drift
    type: drift-monitoring
    detectors:
      - {type: drift-mmd}
    classwise: [drift-mmd]

  - name: object_drift
    inputs: [reference, {name: tests, list: true}]
    steps:
      - {name: ref-crops, transform: wrap, input: reference, wrapper: DetectionCrops}
      - {name: test-crops, transform: wrap, input: tests, wrapper: DetectionCrops}
      - {name: drift, workflow: drift, input: [ref-crops, test-crops]}
```

`test-crops` runs once for each test source. The preset's steps are addressed through `drift`, such as
`drift/drift-mmd-classes-check`. Three caveats apply to what the recipe measures:

- **It asks about the objects, not the scenes.** Drift on crops asks whether the objects look different, and a change
  in the scene around them does not register. Crops also vary in size, so the extractor's preprocessor must resize
  every crop to one size.
- **A crop-level p-value reads optimistic.** The boxes from one image are correlated, while drift tests assume
  independent items, so the test sees more evidence than the images give it.
- **The cost scales with the boxes.** The extractor embeds every box, so a dataset of many boxes per image costs many
  times what its images would.

The run also emits DataEval's warning that `source_id` was binned automatically. The result's binning record reads
the metadata that `DetectionCrops` adds to each crop, and DataEval bins its `source_id`. The warning is harmless for
drift.

## 5. Reading many verdicts

Every source, class and group is a separate test at the detector's `p_val`, so each has about that chance of flagging
drift by luck when nothing changed. With 24 classes and 3 test sources, there are 72 by-class tests, and at 0.05 a few
of them flag on unchanged data. Read the size of a flagged distance beside its flag, and compare it with the distances
of the classes that did not flag. A class that flags at a p-value just under `p_val`, with a distance close to the quiet
classes', is likelier chance than drift. To flag less often, lower `p_val`. Judge the classes or sources that warned
together, against how many were tested. The univariate detector's `correction` does not help here: it corrects across
the embedding's features within one test, not across classes or sources. Repeat a surprising result on new data before
you act on it. The [classwise drift tutorial](../notebooks/classwise_drift.py) shows a run where five of 24 classes
flag, three of them the ones that were degraded.

## See also

- [Distribution Shift](../concepts/DistributionShift.md) — what drift and out-of-distribution detection ask
- [Check and Combine Catalog](../reference/checks.md) — the `drift` check's fields, and `by: class`
- [Evaluator Catalog](../reference/evaluators.md) — each drift evaluator's fields
