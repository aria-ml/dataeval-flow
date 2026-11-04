# Compare one group of classes with another

To compare one group of the reference with a different group of the test data, such as the cats in one against the dogs
in the other, narrow each side with a [`view`](../reference/transforms.md#view) step and run a detector on the two
views. [Distribution Shift](../concepts/DistributionShift.md) explains what drift asks. You need a reference source, one
or more test sources, and the extractor `bovw`.

## Narrow each side and run the detector

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
        operations: [{type: Filter, params: {classes: [0]}}]
      - name: test-dogs
        transform: view
        input: tests
        operations: [{type: Filter, params: {classes: [1]}}]
      - {name: drift, evaluator: mmd, input: [ref-cats, test-dogs]}
      - {name: drift-check, check: drift, input: drift}
```

`Filter`'s `classes` takes class indexes. `test-dogs` and the steps after it run once for each test source.

## Next steps

- [Run a step once per class](run_a_step_once_per_class.md) — Run a detector once per class, or once per group of
  classes in one source.
- [Detect classwise drift](../notebooks/classwise_drift.py) — Find which classes drive a drift signal, and compare
  groups of classes.
- [Monitor drift with steps](monitor_drift.md) — Test each source on its own, and read many verdicts with care.
