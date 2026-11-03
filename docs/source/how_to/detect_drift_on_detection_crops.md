# Detect drift on detection crops

A detection Dataset has several labels per image, so `by: class` cannot key it. The
[`wrap`](../reference/transforms.md#wrap) transform with DataEval's `DetectionCrops` turns each box into a
classification item labeled with the box's class, so a custom workflow can run the
[`shift`](../reference/presets.md#shift) preset on crops. [Distribution Shift](../concepts/DistributionShift.md)
explains what drift asks. You need a reference source and one or more test sources that hold detection data, and an
extractor whose preprocessor resizes every crop to one size.

## Wrap the data and run the preset as a step

Wrap the reference and the tests, and run the preset as a step:

```yaml
workflows:
  - name: drift
    type: shift
    detectors:
      - {type: drift-mmd}
    classwise:
      drift-mmd: class

  - name: object_drift
    inputs: [reference, {name: tests, list: true}]
    steps:
      - {name: ref-crops, transform: wrap, input: reference, wrapper: DetectionCrops}
      - {name: test-crops, transform: wrap, input: tests, wrapper: DetectionCrops}
      - {name: drift, workflow: drift, input: [ref-crops, test-crops]}
```

`test-crops` runs once for each test source. The preset's steps are addressed through `drift`, such as
`drift/drift-mmd-by-class-check`.

## Know what the recipe measures

Three caveats apply:

- Drift on crops asks about the objects, not the scenes. It asks whether the objects look different, and a change in the
  scene around them does not register. Crops also vary in size, so the extractor's preprocessor must resize every crop
  to one size.
- A crop-level p-value reads optimistic. The boxes from one image are correlated, while drift tests assume independent
  items, so the test sees more evidence than the images give it.
- The cost scales with the boxes. The extractor embeds every box, so a dataset of many boxes per image costs many times
  what its images would.

The run also emits DataEval's warning that `source_id` was binned automatically. The result's binning record reads the
metadata that `DetectionCrops` adds to each crop, and DataEval bins its `source_id`. The warning is harmless for drift.

## Next steps

- [Monitor drift with steps](monitor_drift.md) — Test each class of a classification source, and read many verdicts with
  care.
- [Detect classwise drift](../notebooks/classwise_drift.py) — Find which classes drive a drift signal.
- [Distribution Shift](../concepts/DistributionShift.md) — Learn what drift and out-of-distribution detection ask.
