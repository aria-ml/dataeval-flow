# Test merged sources for drift

Testing each source on its own tells you which source drifted. When you would rather ask whether the pooled test data
drifted, merge the sources and run one detector on the result. A custom workflow does this: the
[`merge`](../reference/transforms.md#merge) transform concatenates the test sources, and
[`drift-mmd`](../reference/evaluators.md#drift-mmd) tests the pooled Dataset against the reference.
[Distribution Shift](../concepts/DistributionShift.md) explains what drift asks. Where the sources name their classes
differently, conform them to one vocabulary first, as the "Conform two datasets and merge them" section of
[Chain steps into a workflow of your own](write_a_custom_workflow.md) does. You need a reference source, two test
sources named `cam1` and `cam2`, and the extractor `bovw`.

## Merge the sources and test them

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
workflow behaved before it tested each source on its own. The recipe names its sources in `inputs:`, since `merge` reads
a list of addresses.

## Next steps

- [Monitor incoming data for drift](../notebooks/drift_monitoring.py) — Run the `shift` preset on real imagery and read
  its drift report.
- [Monitor drift with steps](monitor_drift.md) — Test each source on its own, and read many verdicts with care.
