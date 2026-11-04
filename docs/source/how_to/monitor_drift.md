# Monitor drift with steps

The [`shift`](../reference/presets.md#shift) preset is a chain of steps that tests each of your test sources against a
reference, with a `drift` check judging each drift detector and an `ood` check each OOD detector. The examples here list
drift detectors only, which adds none of the OOD steps. With no `detectors:`, it runs `drift-univariate` and
`ood-kneighbors`. A univariate KS test can miss correlated shifts across embedding dimensions: list `drift-mmd` or
`drift-domain-classifier` for those. This guide shows how to test each source on its own, how to test each class, and
how to read many verdicts. [Monitor incoming data for drift](../notebooks/drift_monitoring.py) runs the preset on real
imagery and teaches chunking, and [Distribution Shift](../concepts/DistributionShift.md) explains what drift asks. Other
guides rebuild parts of the preset as a custom workflow:
[Test merged sources for drift](test_merged_sources_for_drift.md),
[Compare one group of classes with another](compare_class_groups.md),
[Detect drift on detection crops](detect_drift_on_detection_crops.md) and
[Monitor a model's uncertainty](monitor_model_uncertainty.md). You need a reference source, two or more test sources and
the extractor `bovw`.

## Test each source on its own

A task with a reference and two test sources runs every detector once for each test source:

```yaml
workflows:
  - name: drift
    type: shift
    detectors:
      - {type: drift-univariate, method: ks, p_val: 0.01}
      - {name: mmd_chunked, type: drift-mmd, chunking: {chunk_count: 10, threshold: [zscore, 2.5]}}
      - {type: drift-kneighbors, k: 5}
    checks:
      drift: {warn_on_drift: true, chunk_percent: 10.0, consecutive_chunks: 2}

tasks:
  - name: cameras
    workflow: drift
    sources: [reference, cam1, cam2]
    extractor: bovw
```

The first source is the reference, and each later source is tested against it. The entries of `detectors:` are drift
evaluator entries, so each takes the fields its evaluator takes in the [Evaluator Catalog](../reference/evaluators.md).
An entry's `name` defaults to its type, and it names the detector's step. Fitting on the reference is the cost of
testing each source on its own, but the reference's embeddings are derived once, and only the fit repeats for each
source.

A chunked entry cuts each test source into chunks of its own, and tests each against the same bounds, which come from
the spread of the reference's chunks. The bounds follow the entry's `threshold`. Unset, they follow the detector's
default, which differs between detectors: the `drift-domain-classifier` judges its AUROC against a constant band. Set
`threshold: [zscore, 3.0]` to judge every chunked detector alike.

The report groups each source's findings under the source's name. From Python, the result is a `ChainResult`, and each
step holds one element per test source:

```python
result = run_task(config, task)

output = result.steps["drift-kneighbors"].elements["cam1"].output  # DataEval's DriftOutput
print(output.drifted, output.distance, output.threshold)

chunks = result.steps["mmd_chunked"].elements["cam1"].output.details  # a polars DataFrame, when chunked
finding = result.steps["mmd_chunked-check"].elements["cam1"].output[0]
print(finding.severity, finding.brief)  # warning, 3/10 chunks drifted
```

The `drift` check is the step `<detector>-check`. Unchunked, it warns where the detector found drift, or reports `info`
when `warn_on_drift` is false. Chunked, it warns where the share of drifted chunks passes `chunk_percent` or the longest
run of drifted chunks passes `consecutive_chunks`, and it reports `info` where some chunk drifted but neither threshold
is passed. A threshold of `null` judges nothing.

A detector that raises, such as a chunked one whose reference is too small to split into 3 chunks, fails its own step
and the task. The other detectors still run, and their findings are in the result.

## Test each class

To see which classes drift, map detectors to `by:` in the preset's `classwise:`, as `classwise: {drift-mmd: class}`.
Each mapped detector also runs once per class, unchunked, as the step `<detector>-by-class`, judged by
`<detector>-by-class-check`:

```yaml
workflows:
  - name: drift
    type: shift
    detectors:
      - {type: drift-mmd}
    classwise:
      drift-mmd: class
```

A class is tested only where the reference and the test source each hold at least 2 items of it. A class too small for
the detector, such as one with no more reference items than `drift-kneighbors`'s `k`, is left out with the detector's
error. The step lists a class it leaves out under `skipped`, with the reason, and the report names it. The by-class
finding is one finding for the source, titled `Drift (MMD) by class`, briefed such as `2/8 classes warn`, and it names
the classes that warned. In Python, `result.steps["drift-mmd-by-class"].elements["cam1"].output` is a `PerClassOutput`:
its `outputs` holds each class's `DriftOutput`, by class name, and its `skipped` holds each class's reason.
[Detect classwise drift](../notebooks/classwise_drift.py) reads one in a full run.

Detection data has no single label per item, so the by-class step is skipped for it with that reason.
[Detect drift on detection crops](detect_drift_on_detection_crops.md) shows how to run it on crops. To run a detector
once per group of classes, see [Run a step once per class](run_a_step_once_per_class.md).

## Read many verdicts

Every source, class and group is a separate test at the detector's `p_val`, so each has about that chance of flagging
drift by luck when nothing changed. With 24 classes and 3 test sources, there are 72 by-class tests, and at 0.05 a few
of them flag on unchanged data. Read the size of a flagged distance beside its flag, and compare it with the distances
of the classes that did not flag. A class that flags at a p-value just under `p_val`, with a distance close to the
distances of the classes that did not flag, is likelier chance than drift. To flag less often, lower `p_val`. Judge the
classes or sources that warned together, against how many were tested. The univariate detector's `correction` does not
help here: it corrects across the embedding's features within one test, not across classes or sources. Repeat a
surprising result on new data before you act on it. The [classwise drift tutorial](../notebooks/classwise_drift.py)
shows a run where five of 24 classes flag, three of them the ones that were degraded.

## Next steps

- [Monitor incoming data for drift](../notebooks/drift_monitoring.py) — Run the preset on real imagery with chunked
  detectors and a baseline control.
- [Distribution Shift](../concepts/DistributionShift.md) — Learn what drift and out-of-distribution detection ask.
- [Test merged sources for drift](test_merged_sources_for_drift.md) — Pool the test sources and run one detector on
  them.
