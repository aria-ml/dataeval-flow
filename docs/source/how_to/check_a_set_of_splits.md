# Check a set of splits

The [`splits`](../reference/presets.md#splits) preset splits a Dataset and judges each part's class shares. To judge the
parts as an audit does, for leakage, shift, coverage and each split's labels and cleanliness, gather the evaluation
parts with `collect` and run [`audit`](../reference/presets.md#audit) as a step on them.
[Dataset Splitting](../concepts/DatasetSplitting.md) explains how the parts are made and judged. You need a source with
a `scene` factor in its metadata, as the example splits on it, and the extractor `bovw_ext`.

## Split, collect and audit

```yaml
workflows:
  - name: splitting
    type: splits
    split_on: [scene]

  - name: release-audit
    type: audit
    outliers: {flags: [pixel], outlier_threshold: zscore}
    factor-leakage: {factors: [scene]}

  - name: split_and_audit
    inputs: [data]
    steps:
      - {name: splits, workflow: splitting, input: data}
      - {name: evals, transform: collect, input: [splits.val, splits.test]}
      - {name: audit, workflow: release-audit, input: [splits.train, evals]}

tasks:
  - {name: split-and-audit, workflow: split_and_audit, sources: [train], extractor: bovw_ext}
```

Keep these points in mind:

- The task's result carries the audit's verdict, record and five questions, as an `audit` task's does, so
  `dataeval-flow --require ready-with-caveats` gates it (see [Gate training on an audit](gate_training_on_an_audit.md)).
  Its steps run as `audit/label-health-train`, `audit/leakage` and so on, and the verdict names them that way. In the
  entry's `accepted:`, write a step without the prefix: `image-outliers-evals[test]` covers
  `audit/image-outliers-evals[test]`.
- `collect` keys each part by its output's name, so the evaluation splits are `val` and `test`. A train/test split
  (`val_frac: 0`) collects `test` alone: `input: [splits.test]`.
- The record has a column `train` and one per key of `evals`: `val` and `test` here. A key of `evals` that is also the
  name of audit's single slot, `train`, is refused, so do not give `keys:` that name it.
- If the `audit` step never starts, because a step before it failed or was skipped, the task gives no verdict. The
  report and the JSON's `no_verdict` say so, and `--require` refuses the task and names the step that did not run.
- Every part's metadata is encoded like train's, as in an `audit` task.
- A workflow gives one verdict, so it runs one audit step, and the step may not be `optional:`.
- With `folds` of 2 or more, `splits.train` and `splits.val` are lists keyed by fold, and the audit step refuses them.
  Audit one fold by naming its elements, `splits.train[0]` with `collect: [splits.val[0], splits.test]`, or chain the
  checks yourself, as the [Check Catalog](../reference/checks.md#leakage) examples do. That collect keys the evaluation
  splits `0` and `test`, so the verdict reads `audit/...-evals[0]` and `accepted:` says `[0]`; give `keys: [val, test]`
  to name them as above.
- With `rebalance:`, `splits.train` is a rebalanced view that may repeat items, which the audit's duplicates check then
  reports.
- Do not embed `data` before the split with the extractor the audit uses: a stateful extractor such as BoVW is fitted
  once per task, on the first Dataset that asks for embeddings, so it would have seen the evaluation items.

## Audit splits that are already on disk

Where the splits are already on disk, give them to an [`audit`](../reference/presets.md#audit) task as sources, train
first: `sources: [train, val, test]`. That is also the route when a training job will load exported splits, so its
digests match the audit's; whether a part's digest survives export and reload is not yet checked.

## Next steps

- [Audit a set of splits before training](../notebooks/audit.py) — Run an audit on real splits and read its verdict.
- [Gate training on an audit](gate_training_on_an_audit.md) — Gate a run on the audit's verdict with `--require`.
- [Dataset Splitting](../concepts/DatasetSplitting.md) — Learn how the splits are made and what each part is judged on.
