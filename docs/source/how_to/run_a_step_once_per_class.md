# Run a step once per class

An evaluate step or a check with `by:` runs once per class, or once per group of classes, and keeps every run in one
Output. It suits an evaluator that compares two sources, such as a drift detector like
[`drift-mmd`](../reference/evaluators.md#drift-mmd), when you want to know which classes differ. The
[`shift`](../reference/presets.md#shift) preset does this for you with `classwise:`:
[Monitor drift with steps](monitor_drift.md) shows it. [Workflows as Chains of Steps](../concepts/WorkflowsAsChains.md)
explains steps, addresses and lists. You need a custom workflow and a Dataset with one label per item, as image
classification has.

## Add by to a step and its check

```yaml
evaluators:
  - {name: mmd, type: drift-mmd}

workflows:
  - name: class_drift
    inputs: [reference, {name: tests, list: true}]
    steps:
      - {name: mmd-classes, evaluator: mmd, input: [reference, tests], by: class}
      - {name: mmd-classes-check, check: drift, input: mmd-classes, by: class}
      - name: mmd-groups
        evaluator: mmd
        input: [reference, tests]
        by:
          class:
            groups: {vehicles: [car, truck, van], people: [person]}
            min_items: 2
      - {name: mmd-groups-check, check: drift, input: mmd-groups, by: class}
```

## Know how the keys are chosen

- `by: class` keys each class by its name, taken from the first input's `index2label`, and runs the step on that class's
  items alone, in ascending class index.
- `groups:` keys each named group instead, in the order written. A group lists classes by name or by index, and a name
  resolves through the first input's `index2label`. A name that resolves to nothing fails the step. Groups may overlap,
  and a class in no group is left out and listed as skipped.
- With several inputs, a key runs only where every input holds at least `min_items` of it, 2 unless you set it.
  Otherwise the key is skipped, and the Output says why.
- A key whose run raises, such as a class too small for the evaluator, is skipped with its error, and the other keys
  still run. The step fails only when every key raises.
- The step's output is a `PerClassOutput`: `outputs` holds each key's own Output, and `skipped` holds each key left out
  with its reason. Only a check with `by:` can read it.
- A check's `by: class` takes no settings, since its keys come from the step it reads. It runs once for each key and
  rolls the findings into one, titled with the per-key title and `by class` or `by group`, which names the keys that
  warned and each key it did not assess.
- `by:` slices embeddings and labels, so it needs a Dataset with one label per item, as image classification has. The
  load refuses an evaluator that reads statistics, metadata or clusters. For detection data, run `wrap` with
  `DetectionCrops` first, as [Detect drift on detection crops](detect_drift_on_detection_crops.md) does.

## Next steps

- [Detect classwise drift](../notebooks/classwise_drift.py) — Find which classes drive a drift signal, then compare the
  degraded classes with the rest as two groups.
- [Monitor drift with steps](monitor_drift.md) — Use `classwise:` in the preset, and read many verdicts with care.
- [Compare one group of classes with another](compare_class_groups.md) — Compare a group of the reference with a
  different group of the test data.
