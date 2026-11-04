# Move from a parameter sweep to a matrix

`type: parameter-sweep` now fails the load as any unknown type does. A task's `matrix:` replaces it, as
[Sweep settings with a matrix](run_a_matrix.md) explains. Its fields become a
[`quality`](../reference/presets.md#quality) entry and a matrix on its task. You need a config with a `parameter-sweep`
entry and its task.

## Rewrite the sweep as a matrix

A sweep such as:

```text
workflows:
  - name: sweep
    type: parameter-sweep
    outlier_threshold: [2.5, 3.5, 4.5]
    duplicate_cluster_sensitivity: [0.5, 2.0]
    duplicate_cluster_algorithm: [hdbscan]

tasks:
  - name: tune
    workflow: sweep
    sources: train
    extractor: bovw_ext
```

becomes:

```yaml
workflows:
  - name: cleaning_tune
    type: quality
    outliers:
      flags: [dimension, pixel, visual]
      outlier_threshold: adaptive
    duplicates:
      cluster_algorithm: hdbscan

tasks:
  - name: tune
    workflow: cleaning_tune
    sources: train
    extractor: bovw_ext
    matrix:
      outliers.outlier_threshold: [[adaptive, 2.5], [adaptive, 3.5], [adaptive, 4.5]]
      duplicates.cluster_sensitivity: [0.5, 2.0]
```

Apply these rules and differences:

- Set `outliers.outlier_threshold` and `outliers.flags` on the entry. The sweep defaulted them to `[adaptive]` and all
  three flag groups, and `quality` has no default for either. The sweep's `outlier_method` and `outlier_threshold` are
  one setting here: `adaptive` alone, or `[adaptive, 2.5]`.
- A field the sweep held one value for is a setting of the entry. A field with several values is a key of the matrix,
  `null` included where the sweep tried DataEval's default.
- The sweep's statistics ignored `value_range`. `quality` reads the dataset's `value_range`, so on float imagery that
  declares one, its statistics can differ from the sweep's.
- On detection data, `quality` also judges boxes, and its outlier counts include them, where the sweep counted whole
  images. On classification data, the count of images flagged and of near-duplicate groups match what the sweep reported
  for each combination, where `duplicates.merge_near_duplicates` is left at its default: the sweep merged near-duplicate
  groups whatever it said.
- The sweep printed one pivot table per count. The matrix prints one table of every run's findings, and each run's full
  result is in `result.runs`, with every step's output.

## Next steps

- [Tune data cleaning with a matrix](../notebooks/tune_data_cleaning.py) — Run the same sweep on MilitaryVehicles and
  choose `quality` thresholds from its table.
- [Sweep settings with a matrix](run_a_matrix.md) — Write the values and keys of a matrix and read its table.
- [Configure outlier detection](configure_outlier_detection.md) — Pick the outlier settings most worth varying.
