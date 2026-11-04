# Configure outlier detection

The [`quality`](../reference/presets.md#quality) workflow is flagging too much, too little, or the wrong thing. Its
`outliers` block controls what counts as an outlier, and its `checks` block controls when a finding becomes a warning.
The [`outliers`](../reference/evaluators.md#outliers) evaluator runs the same detection alone, and it takes every
setting of the preset's `outliers` block, spelled the same way. The
[Data Quality and Cleaning](../concepts/DataQualityAndCleaning.md) explanation covers the concepts behind the
detection, and the
[DataEval Data Integrity explanation](https://dataeval.readthedocs.io/en/latest/concepts/DataIntegrity.html) is the
authoritative treatment of the detection methods themselves. This guide needs a pipeline that runs `quality`.

## Pick a statistical method

The method in `outliers.outlier_threshold` selects how far from typical a statistic has to be before the sample is
flagged. There is no universally correct choice. It depends on how heavy-tailed your data is.

| Method | Flags a sample when a statistic is… | Use when |
| --- | --- | --- |
| `adaptive` | beyond a threshold chosen from the observed distribution | you do not yet know the shape of your data (a good first run) |
| `modzscore` | far from the median, scaled by median absolute deviation | the dataset already contains outliers that would drag a mean |
| `zscore` | far from the mean, scaled by standard deviation | statistics are roughly normal and outliers are rare |
| `iqr` | outside the interquartile fences | the distribution is skewed and you want a distribution-free rule |

```yaml
workflows:
  - name: quality_check
    type: quality
    outliers:
      flags: [dimension, pixel, visual]
      outlier_threshold: modzscore
```

## Choose which statistics to test

`outliers.flags` selects the groups of image statistics the method is applied to. At least one is required.

- `dimension`: geometry. Width, height, aspect ratio, channel count, value range, total pixel count, and (for detection
  boxes) offsets and distances to the image center and edges.
- `pixel`: the intensity distribution. Mean, standard deviation, variance, skew, kurtosis, entropy, and the
  fraction of zero or missing/NaN pixels. As of DataEval v1.1 these are reported in the units the data is stored in,
  so their magnitude depends on the encoding. A 12-bit mean reads in the thousands.
- `visual`: perceived appearance derived from intensity percentiles. Brightness, contrast, darkness, and
  edge-detection sharpness. These read the 0–255 display range regardless of encoding, so one image answers the same
  whatever it is stored as.

Float imagery needs `value_range` before the `visual` group, and pixel histogram and entropy, can be computed at
all. Without it they answer `NaN`. See [Declare the value range of float images](declare_a_value_range.md).

Narrow the list when you already know what kind of defect you are hunting. A dimension-only run over a freshly
converted dataset is fast and answers one question cleanly. To narrow further, name a sub-group in place of its
family: `pixel_basic`, `visual_basic`, `dimension_basic` and the others listed in
[Measure band groups](measure_band_groups.md).

## Override the threshold

Write `outliers.outlier_threshold` as `[method, bound]` to replace the method's built-in cutoff. Write the method alone
to use the DataEval default for it. Raise the bound to flag less, and lower it to flag more.

```yaml
    outliers:
      outlier_threshold: [modzscore, 3.5]
```

Because the right value depends on the dataset, this is the parameter most worth sweeping. A `matrix:` on the task
runs it once per value and compares the flag rates in one table. See
[Sweep settings with a matrix](run_a_matrix.md), and
[Tune data cleaning with a matrix](../notebooks/tune_data_cleaning.py) for a worked run.

:::{note}
The pixel rescale that arrived in v0.2 did not move any outlier flag, and a threshold tuned under v0.1 is still
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
    type: quality
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
configured. `outliers.n_clusters` is a hint: omit it and the algorithm auto-detects. `hdbscan` handles clusters of
varying density and does not need a cluster count. `kmeans` is faster and predictable when you know roughly how many
groups to expect.

## Set when a finding becomes a warning

Detection and {term}`severity <Severity>` are separate concerns. `checks` sets the bound past which each finding is
a `warning`, counted in the report's health line. It does not change what is detected.

```yaml
    checks:
      image-duplicates: {exact: 0.0, near: 5.0}  # % of images in exact- and near-duplicate groups
      image-outliers: {warning: 5.0}             # % of images flagged
      target-outliers: {warning: 10.0}           # % of labels/annotations flagged
      class-outliers: {warning: 12.0}        # % flagged within any single class
```

Rough guidance: tighten toward 1–2% for curated benchmarks and safety-critical datasets, and loosen toward 10–15% for
large web-scraped or naturally diverse collections. Class imbalance is judged by the
[`bias`](../reference/presets.md#bias) preset. For a class hierarchy with a long tail, raise its `class-imbalance` to
10–20 to avoid a warning that only restates the domain.

## Verify the effect

Every run reports the flag rate alongside the health line. Change one parameter and re-read the report:

```python
result = run_task(config, task)
print(result.report())
```

See [Read evaluation outputs](read_evaluation_outputs.md) for the report structure, and [Get the numbers behind a
finding](get_the_numbers_behind_a_finding.md) for how to pull the flagged indices out of the result for inspection.

## Next steps

- [Clean a dataset](../notebooks/data_cleaning.py) — Run `quality` end to end on a sample of a dataset and read its
  report.
- [Data Quality and Cleaning](../concepts/DataQualityAndCleaning.md) — The ideas behind outlier and duplicate detection.
- [Configure metadata binning](configure_metadata_binning.md) — Use `intrinsic_factors` to measure image statistics into
  the `metadata:` policy that the workflow names.
