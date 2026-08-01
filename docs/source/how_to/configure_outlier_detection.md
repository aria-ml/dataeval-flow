# Configure outlier detection

The `data-cleaning` workflow is flagging too much, too little, or the wrong thing. This guide covers the knobs that
control what counts as an outlier and when a finding becomes a warning.

## Used in these tutorials

- {doc}`Clean a dataset <../notebooks/data_cleaning>`
- {doc}`Parameter Sweep for Data Cleaning <../notebooks/parameter_sweep>`

## Pick a statistical method

`outlier_method` selects how far from typical a statistic has to be before the sample is flagged. There is no
universally correct choice — it depends on how heavy-tailed your data is.

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
    mode: advisory
    outlier_method: modzscore
    outlier_flags: [dimension, pixel, visual]
```

## Choose which statistics to test

`outlier_flags` selects the *groups* of image statistics the method is applied to. At least one is required.

- `dimension` — geometry: width, height, aspect ratio, channel count, bit depth, total pixel count, and (for detection
  boxes) offsets and distances to the image center and edges. Catches the wrong-shaped image, the accidental
  thumbnail, the one grayscale file in an RGB set, the degenerate bounding box.
- `pixel` — the intensity *distribution*: mean, standard deviation, variance, skew, kurtosis, entropy, and the
  fraction of zero or missing/NaN pixels. Catches corrupt, truncated, and constant-valued frames.
- `visual` — perceived appearance derived from intensity percentiles: brightness, contrast, darkness, and
  edge-detection sharpness. Catches the blown-out, the underexposed, and the out-of-focus capture.

Narrow the list when you already know what kind of defect you are hunting. A dimension-only run over a freshly
converted dataset is fast and answers one question cleanly.

## Override the threshold

`outlier_threshold` replaces the method's built-in cutoff. Leave it unset to use the DataEval default for the method
you chose; raise it to flag less, lower it to flag more.

```yaml
    outlier_method: modzscore
    outlier_threshold: 3.5
```

Because the right value depends on the dataset, this is the parameter most worth sweeping rather than guessing — see
{doc}`the Parameter Sweep tutorial <../notebooks/parameter_sweep>` to run a grid and compare the flag rates side by
side.

## Add cluster-based detection

Statistical flags only see per-image statistics. A sample can be statistically unremarkable and still sit far from
every cluster in {term}`embedding <Embedding>` space — a picture of something that simply does not belong. Catching
that requires an {term}`extractor <Extractor>`.

```yaml
extractors:
  - name: bovw_ext
    model: bovw
    vocab_size: 512

workflows:
  - name: quality_check
    type: data-cleaning
    mode: advisory
    outlier_method: adaptive
    outlier_flags: [dimension, pixel, visual]
    outlier_cluster_threshold: 3.5        # std devs from a cluster center
    outlier_cluster_algorithm: hdbscan    # or kmeans
    outlier_n_clusters: 5                 # omit to auto-detect

tasks:
  - name: check
    workflow: quality_check
    sources: my_source
    extractor: bovw_ext                   # required for cluster-based detection
```

Leaving `outlier_cluster_threshold` unset skips cluster-based detection entirely, even when an extractor is
configured. `outlier_n_clusters` is a hint — omit it and the algorithm auto-detects. `hdbscan` handles clusters of
varying density and does not need a cluster count; `kmeans` is faster and predictable when you know roughly how many
groups to expect.

## Decide when a finding becomes a warning

Detection and *severity* are separate concerns. `health_thresholds` sets the rate at which each finding is elevated
from `info` to `warning` in the report's health line — it does not change what is detected.

```yaml
    health_thresholds:
      exact_duplicates: 0.0         # any byte-identical image warns
      near_duplicates: 5.0          # % of images in near-duplicate groups
      image_outliers: 5.0           # % of images flagged
      target_outliers: 10.0         # % of labels/annotations flagged
      classwise_outliers: 12.0      # % flagged within any single class
      class_label_imbalance: 5.0    # max:min class count ratio
```

Rough guidance: tighten toward 1–2% for curated benchmarks and safety-critical datasets; loosen toward 10–15% for
large web-scraped or naturally diverse collections. For a class hierarchy with a long tail, raise
`class_label_imbalance` to 10–20 to avoid a warning that only restates the domain.

## Verify the effect

Every run reports the flag rate alongside the health line, so the fastest feedback loop is to change one parameter and
re-read the report:

```python
result = run_task(task, config)
print(result.report())
```

See {doc}`read_evaluation_outputs` for the report structure and how to pull the flagged indices out of the result for
inspection.

## Related material

- [Data Quality and Cleaning](../concepts/DataQualityAndCleaning.md) — the concepts behind outlier and duplicate
  detection
- [DataEval Data Integrity explanation](https://dataeval.readthedocs.io/en/latest/concepts/DataIntegrity.html) — the
  authoritative treatment of the detection methods themselves
- {doc}`API Reference <../reference/autoapi/dataeval_flow/index>` — every field and default on
  `DataCleaningParameters` and `DataCleaningHealthThresholds`
