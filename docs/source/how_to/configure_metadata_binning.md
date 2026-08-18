# Configure metadata binning

Bias, balance, diversity, and parity do not read your metadata factors as measured. They read them as **codes** — a
continuous factor is cut into intervals first, and a categorical one is mapped to ordinals. Where those cuts fall
changes the numbers those evaluators report, so binning is a parameter of the evaluation, not an implementation
detail. This guide covers choosing the binning, excluding factors, and reading back what the run actually did.

## Used in these tutorials

- {doc}`Assess dataset coverage <../notebooks/data_coverage>`
- {doc}`Analyze dataset quality across splits <../notebooks/data_analysis>`

## Which workflows this applies to

The three `metadata_*` settings are accepted by every workflow that builds metadata:

| Workflow | Reads metadata for |
| --- | --- |
| `data-analysis` | balance, diversity, per-factor summaries |
| `data-cleaning` | classwise outlier attribution and label statistics |
| `data-coverage` | class balance, factor gap analysis, factor-to-class mutual information |
| `ood-detection` | factor deviation and predictors for flagged samples |

## Let the method choose the cuts

`metadata_auto_bin_method` picks how a continuous factor is discretized when you have not said otherwise.

```yaml
workflows:
  - name: coverage_check
    type: data-coverage
    metadata_auto_bin_method: uniform_width
```

| Method | Cuts the range into | Use when |
| --- | --- | --- |
| `uniform_width` | equal-width intervals | the scale is meaningful and you want bins that read as ranges |
| `uniform_count` | intervals holding equal counts | the distribution is skewed and equal-width bins would leave most bins near-empty |
| `clusters` | intervals found from the data's own structure | the values group naturally and you want the groups, not a grid |

Leave it unset to take DataEval's default (`uniform_width`).

:::{important}
The bin **count** an automatic method lands on is derived from the data, not taken as a setting. Two runs over
different samples of the same population can therefore produce different bin counts for the same factor — which is
enough to move a `Balance` score. Pin the count with `metadata_continuous_factor_bins` for any factor whose numbers
you intend to compare across runs.
:::

## Pin the cuts for a specific factor

`metadata_continuous_factor_bins` overrides the automatic method per factor. Give it a bin **count**, or explicit
**edges** when the boundaries carry domain meaning:

```yaml
    metadata_continuous_factor_bins:
      elevation: 8                          # eight bins, placed by the auto method
      temperature: [-40, 0, 20, 40, 60]     # explicit edges — four bins
```

Explicit edges are the stronger choice for anything comparative: they are a property of your configuration rather
than of the sample, so the same edges apply to every run and every split. Naming a factor the dataset does not carry
is not an error — DataEval ignores it and warns — and the run records it as an unmatched request (see below).

## Drop factors that are not evidence

`metadata_exclude` removes factors before any evaluator sees them:

```yaml
    metadata_exclude: [id, filename, width, height]
```

The usual candidates are identifiers and bookkeeping columns. An `id` is unique per sample, so it correlates
perfectly with everything and reports as maximally informative while telling you nothing. Image geometry is worth
excluding when it is a property of your pipeline rather than of the scene.

Exclusion is recorded in the result, because an excluded factor otherwise leaves no trace — it is simply absent, and
absent-because-excluded looks identical to absent-because-never-collected.

## Declare the range of float image data

`value_range` is a separate setting that governs **image statistics**, not metadata factors — but it fails the same
quiet way, so it belongs here.

Integer encodings state the interval their values occupy. Float data does not. As of DataEval v1.1 the statistics
that need one answer `NaN` rather than inferring it:

```yaml
workflows:
  - name: quality_check
    type: data-cleaning
    outlier_method: modzscore
    outlier_flags: [dimension, pixel, visual]
    value_range: [-50.0, 50.0]      # metres above and below sea level
```

Affected without a declared range: the whole `visual` group, pixel histogram and entropy, and dimension depth.
`PIXEL_MISSING` always answers, because it measures the presence of data rather than the data. Leave `value_range`
unset for ordinary integer imagery — the `[0, 1]` and `0–255` float conventions are still detected automatically.

`value_range` participates in the cache key, so two runs declaring different ranges never share a cached entry.

## Read back what the run did

Every result records its binning decisions, so you do not have to reason about them from the configuration. The
record appears in the text report under **METADATA FACTORS** and in the result envelope at
`metadata.metadata_binning`:

```text
--------------------------------------------------------------------------------
  METADATA FACTORS
--------------------------------------------------------------------------------
  Auto-bin method: uniform_width
  Excluded:        id
  Unmatched bins:  not_a_factor
    elevation [continuous @ unit] — binned, 4 (requested)
        bin 1: n=4   [41.87, 68.52]
        bin 2: n=19  [68.85, 94.77]
        bin 3: n=25  [96.02, 119.6]
        bin 4: n=12  [122.6, 149]
    sensor [categorical @ unit] — 3 categories
        0: a (n=15)
        1: b (n=22)
        2: c (n=23)
    file_name [categorical @ unit] — 250 categories (one per sample)
```

Per factor the envelope records the type, the {term}`level <Metadata Level>` it was binned at, whether it was binned
or digitized, and the decision itself — the observed range and population of every bin, or the ordinal-to-value map
of every category. Observed ranges are reported rather than nominal edges: they describe what the run actually did
and stay meaningful for a bin that ended up empty or clipped.

The text report shows that detail only for a factor with 12 or fewer bins or categories. Above that it gives the
count and how the buckets were populated — `40 categories, n=3–19 per category`, or the overall span for a binned
factor — so one high-cardinality factor cannot bury the rest. A factor holding exactly one category per sample is an
identifier rather than a grouping, and is labelled `(one per sample)`; it contributes nothing to balance or
diversity, so it is a candidate for `metadata_exclude`. The envelope is unaffected by the cap — read
`binning["factors"][name]["categories"]` for the full map.

From Python:

```python
result = run_task(task, config)

binning = result.metadata.metadata_binning
for name, info in binning["factors"].items():
    if info["is_binned"]:
        print(name, info["bin_count"], [(b["min"], b["max"]) for b in info["bins"]])
```

For `data-analysis`, splits are binned independently — two splits of one dataset can land on different edges — so the
record is nested one level deeper, under `binning["per_split"][split_name]`.

### Diagnostics

`metadata.diagnostics` carries the warnings DataEval raised during the run: the ranges it could not resolve, bin
requests it ignored, and factors it dropped. These used to reach only the console and `result.log`; they are now part
of the envelope, so an archived result can answer for itself why a statistic came back `NaN`.

## Why this matters for comparison

Binning is the reason two evaluations of "the same" data can disagree. A `Balance` score is an association between a
factor's codes and the class labels, and a factor's codes depend on where its cuts fell. Change the sample, and an
automatic method may place different cuts; change the bin count, and the entropy the association is normalized
against changes with it.

Two rules keep runs comparable:

1. **Pin the bins** for any factor you will compare across runs — explicit edges by preference.
2. **Compare the records, not just the scores.** If two runs disagree, diff their `metadata_binning` blocks before
   concluding the data moved.

This is the same property that makes a result reproducible — see [Reproducibility](../concepts/Reproducibility.md).

## Related material

- {doc}`read_evaluation_outputs` — the result envelope the binning record lives in
- {doc}`configure_outlier_detection` — `value_range` also affects which image statistics are computable
- [Provenance](../concepts/Provenance.md) — why a result has to carry the decisions made on its behalf
- [DataEval Dataset Bias and Coverage explanation](https://dataeval.readthedocs.io/en/latest/concepts/DatasetBias.html)
  — the balance and diversity measures that read binned codes
- {doc}`API Reference <../reference/autoapi/dataeval_flow/index>` — `MetadataConfigMixin` and `StatsConfigMixin`
