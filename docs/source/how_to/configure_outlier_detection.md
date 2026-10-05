# Configure outlier detection

The `data-cleaning` workflow is flagging too much, too little, or the wrong thing. This guide covers the knobs that
control what counts as an outlier and when a finding becomes a warning.

## Used in these tutorials

- {doc}`Clean a dataset <../notebooks/data_cleaning>`
- {doc}`Tune data cleaning with a matrix <../notebooks/tune_data_cleaning>`

## Pick a statistical method

The method in `outliers.outlier_threshold` selects how far from typical a statistic has to be before the sample is
flagged. There is no universally correct choice. It depends on how heavy-tailed your data is.

| Method | Flags a sample when a statistic is… | Use when |
| --- | --- | --- |
| `adaptive` | beyond a threshold chosen from the observed distribution | you do not yet know the shape of your data — a good first run |
| `modzscore` | far from the *median*, scaled by median absolute deviation | the dataset already contains outliers that would drag a mean |
| `zscore` | far from the *mean*, scaled by standard deviation | statistics are roughly normal and outliers are rare |
| `iqr` | outside the interquartile fences | the distribution is skewed and you want a distribution-free rule |

```yaml
workflows:
  - name: quality_check
    type: data-cleaning
    outliers:
      flags: [dimension, pixel, visual]
      outlier_threshold: modzscore
```

## Choose which statistics to test

`outliers.flags` selects the *groups* of image statistics the method is applied to. At least one is required.

- `dimension` — geometry: width, height, aspect ratio, channel count, value range, total pixel count, and (for detection
  boxes) offsets and distances to the image center and edges.
- `pixel` — the intensity *distribution*: mean, standard deviation, variance, skew, kurtosis, entropy, and the
  fraction of zero or missing/NaN pixels. As of DataEval v1.1 these are reported in the units the data is stored in,
  so their magnitude depends on the encoding — a 12-bit mean reads in the thousands.
- `visual` — perceived appearance derived from intensity percentiles: brightness, contrast, darkness, and
  edge-detection sharpness. These read the 0–255 display range regardless of encoding, so one image answers the same
  whatever it is stored as.

Float imagery needs `value_range` before the `visual` group — and pixel histogram and entropy — can be computed at
all; without it they answer `NaN`. Declare it on the dataset — see {doc}`configure_metadata_binning`.

Narrow the list when you already know what kind of defect you are hunting. A dimension-only run over a freshly
converted dataset is fast and answers one question cleanly.

## Override the threshold

Write `outliers.outlier_threshold` as `[method, bound]` to replace the method's built-in cutoff. Write the method alone
to use the DataEval default for it; raise the bound to flag less, lower it to flag more.

```yaml
    outliers:
      outlier_threshold: [modzscore, 3.5]
```

Because the right value depends on the dataset, this is the parameter most worth sweeping. A `matrix:` on the task
runs it once per value and compares the flag rates in one table: see {doc}`Sweep settings with a matrix
<run_a_matrix>`, and {doc}`Tune data cleaning with a matrix <../notebooks/tune_data_cleaning>` for a worked run.

:::{note}
The pixel rescale that arrived in v0.2 did **not** move any outlier flag, and a threshold tuned under v0.1 is still
correct. Every method offered here is location-scale equivariant: each measures distance in units of the
distribution's own spread, so scaling every value by a constant moves the statistic and the cutoff together.
:::

## Add cluster-based detection

Statistical flags only see per-image statistics. A sample can be statistically unremarkable and still sit far from
every cluster in {term}`embedding <Embedding>` space. Catching that requires an {term}`extractor <Extractor>`.

```yaml
extractors:
  - name: bovw_ext
    model: bovw
    vocab_size: 512

workflows:
  - name: quality_check
    type: data-cleaning
    outliers:
      flags: [dimension, pixel, visual]
      outlier_threshold: adaptive
      cluster_threshold: 3.5              # std devs from a cluster center
      cluster_algorithm: hdbscan          # or kmeans
      n_clusters: 5                       # omit to auto-detect

tasks:
  - name: check
    workflow: quality_check
    sources: my_source
    extractor: bovw_ext                   # required for cluster-based detection
```

Leaving `outliers.cluster_threshold` unset skips cluster-based detection entirely, even when an extractor is
configured. `outliers.n_clusters` is a hint — omit it and the algorithm auto-detects. `hdbscan` handles clusters of
varying density and does not need a cluster count; `kmeans` is faster and predictable when you know roughly how many
groups to expect.

## Decide when a finding becomes a warning

Detection and {term}`severity <Severity>` are separate concerns. `checks` sets the bound past which each finding is
a `warning`, counted in the report's health line. It does not change what is detected.

```yaml
    checks:
      image-duplicates: {exact: 0.0, near: 5.0}  # % of images in exact- and near-duplicate groups
      image-outliers: {warning: 5.0}             # % of images flagged
      target-outliers: {warning: 10.0}           # % of labels/annotations flagged
      classwise-outliers: {warning: 12.0}        # % flagged within any single class
      class-imbalance: {warning: 5.0}            # max:min class count ratio
```

Rough guidance: tighten toward 1–2% for curated benchmarks and safety-critical datasets; loosen toward 10–15% for
large web-scraped or naturally diverse collections. For a class hierarchy with a long tail, raise
`class-imbalance` to 10–20 to avoid a warning that only restates the domain.

## Verify the effect

Every run reports the flag rate alongside the health line. Change one parameter and re-read the report:

```python
result = run_task(config, task)
print(result.report())
```

See {doc}`read_evaluation_outputs` for the report structure and how to pull the flagged indices out of the result for
inspection.

## Related material

- {doc}`configure_metadata_binning` — the `metadata:` policy this workflow names, `intrinsic_factors` for measuring
  statistics into the metadata, and `value_range` for float imagery
- [Data Quality and Cleaning](../concepts/DataQualityAndCleaning.md) — the concepts behind outlier and duplicate
  detection
- [Evaluator Catalog](../reference/evaluators.md#outliers) — the `outliers` evaluator runs the same detection
  alone, and takes every setting of the preset's `outliers` block, spelled the same way
- [DataEval Data Integrity explanation](https://dataeval.readthedocs.io/en/latest/concepts/DataIntegrity.html) — the
  authoritative treatment of the detection methods themselves
- [Preset Catalog: `data-cleaning`](../reference/presets.md#data-cleaning) — every setting and `checks:` default
  of `data-cleaning`, from `DataCleaningConfig` and `DataCleaningChecks`
