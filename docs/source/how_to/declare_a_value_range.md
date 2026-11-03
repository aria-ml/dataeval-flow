# Declare the value range of float images

Integer encodings state the interval their values occupy. Float data does not. As of DataEval v1.1 the image statistics
that need an interval answer `NaN` when a float dataset does not declare one. `value_range` governs image statistics,
not metadata factors. Declare it on the dataset, because it is a fact about the imagery and not a setting of any one
workflow. This guide declares the range for a dataset that a [`quality`](../reference/presets.md#quality) workflow
measures. It needs a dataset of float imagery and a workflow that computes image statistics.

## Declare the range on the dataset

```yaml
datasets:
  - name: bathymetry
    format: huggingface
    path: data/bathymetry
    task: image_classification
    value_range: [-50.0, 50.0]      # meters above and below sea level

workflows:
  - name: quality_check
    type: quality
    outliers:
      flags: [dimension, pixel, visual]
      outlier_threshold: modzscore
```

Every workflow reading that dataset then measures against the same interval, including the `intrinsic_factors` pass
described in [Configure metadata binning](configure_metadata_binning.md). A declared range then means the same thing to
the injection and to the workflow's own statistics. A channel group can declare a range of its own. See
[Measure band groups](measure_band_groups.md).

`value_range` participates in the cache key, so two runs declaring different ranges never share a cached entry.

## Check which statistics need a range

Affected without a declared range: the whole `visual` group, pixel histogram and entropy, and dimension depth.
`PIXEL_MISSING` always answers, because it measures the presence of data. Leave `value_range` unset for ordinary
integer imagery. The `[0, 1]` and `0–255` float conventions are still detected automatically.

## Next steps

- [Configure outlier detection](configure_outlier_detection.md) — Choose the statistic families and the method that flag
  outliers in your imagery.
- [Measure band groups](measure_band_groups.md) — Measure channels separately, each against a range of its own.
