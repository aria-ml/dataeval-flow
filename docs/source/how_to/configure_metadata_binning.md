# Configure metadata binning

Bias, balance, diversity, and parity read your metadata factors as **codes** — a continuous factor cut into
intervals, a categorical one mapped to ordinals. Where the cuts fall changes the numbers those evaluators report.
Binning is a parameter of the evaluation. This guide covers choosing the binning, excluding factors, and reading
back what the run did.

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

## Define the policy once and share it

The encoding is a decision, not a per-workflow setting. Two workflows over one dataset that cut it differently
produce numbers that land in the same result file and cannot be compared. Define it once under `metadata:` and
reference it by name, the same way `datasets`, `views`, `sources` and `extractors` work:

```yaml
metadata:
  - name: standard
    encoding: policy/factor_bins.json     # a committed descriptor
    exclude: [id, filename]
    intrinsic_factors: [visual, pixel]    # measure these off the imagery and bin them too
    continuous_factor_bins:
      elevation: 8
      brightness: 4

workflows:
  - name: coverage_check
    type: data-coverage
    metadata: standard
  - name: quality_check
    type: data-cleaning
    metadata: standard        # same policy — and the digests prove it
```

A policy carries everything that decides how a factor becomes a code: `encoding`, `factor_levels`, `strict`,
`auto_bin_method`, `exclude`, `continuous_factor_bins`, `intrinsic_factors`, `factor_source`, and
`reference_split`.

The older per-workflow `metadata_*` fields still work and mean the same things. Naming a policy *and* setting one of
them on the same workflow is an error rather than a merge — two sources disagreeing about one factor has no good
resolution.

Everything a policy says is checked before the dataset is read, and a mistake costs only a message:

| The config says | What happens |
| --- | --- |
| `metadata:` names no policy in the pool | Error, listing the policies that do exist |
| `encoding` points at a file that is not there | Error — a descriptor matching nothing silently reverts every factor it meant to pin |
| One factor named by both `encoding` and `continuous_factor_bins` | Error |
| `strict: true` over a descriptor with unreviewed vocabularies | Error, naming them (see below) |
| `intrinsic_factors` names something that is not a family | Error, listing the families that exist |
| A statistic pinned by both a level-prefixed `encoding` entry and a bare `continuous_factor_bins` name | Error |
| `include_image_stats: true` alongside an `intrinsic_factors` that says something else | Error |

### Get a descriptor out of a run

The descriptor is not written by hand. Every run records the encoding it used, and a run with `-o` writes it
beside the results:

```text
output/results/
  result.json      # the record, per factor
  result.txt
  encoding.json    # the same record, as the artifact you commit
```

Any archived `result.json` yields one too. The value of pinning an encoding is usually noticed weeks after
the run:

```console
dataeval-flow encoding output/results/result.json -o policy/factor_bins.json
git add policy/factor_bins.json
```

Then reference it, and every later run is cut the same way:

```yaml
metadata:
  - name: standard
    encoding: policy/factor_bins.json
```

Review it before committing. `provenance` is the field to read. An entry still saying `derived` is one nobody has
looked at, and pinning it locks in a cut DataEval chose from one sample. Editing `provenance` to `accepted` records
that you read it and it is right. That is what lets `strict` be set later.

A run whose tasks encoded the dataset differently writes no `encoding.json`, because no single descriptor describes
it; pass `--task <name>` to take one. Splits work the same way — see `reference_split`.

### Apply a committed encoding

`encoding` points at a descriptor — the artifact `dataeval-flow encoding` writes — held under the data root and
committed alongside the config. It applies the recorded cuts and vocabularies. That is what makes two runs over
different data comparable:

```yaml
metadata:
  - name: locked
    encoding: policy/factor_bins.json
```

Factors the descriptor does not name are encoded normally. A vocabulary it does name grows by **appending**: a
category the descriptor never saw takes the next free code, so codes already assigned keep meaning what they meant.

### Close a vocabulary you have reviewed

`strict: true` makes a value outside a declared vocabulary an error. Use it for a fixed taxonomy: the vocabulary
stays closed, and new data is reported.

It is refused over a descriptor whose vocabularies still read `provenance: "derived"` — what a descriptor
exported from an exploratory run contains. `strict` does not consult provenance. Allowing it there would enforce a
taxonomy **nobody decided on** and fail the run on the first new category:

```text
Metadata policy 'locked' sets strict, which closes every vocabulary in its descriptor, but
['weather'] still read provenance="derived" — nobody reviewed them. Ratify them in the
descriptor (set provenance to "accepted" or "declared"), or drop strict.
```

Ratify the entries you have actually looked at by editing `provenance` to `accepted` in the committed descriptor.
The pull request is the point.

## Let the method choose the cuts

`metadata_auto_bin_method` picks how an un-pinned continuous factor is discretized.

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
The bin **count** an automatic method lands on is derived from the data. Two runs over
different samples of the same population can therefore produce different bin counts for the same factor. That is
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

Explicit edges are the stronger choice for anything comparative. They are a property of the configuration, and
the same edges apply to every run and every split. Naming a factor the dataset does not carry is not an error.
DataEval ignores it and warns, and the run records it as an unmatched request (see below).

## Drop factors that are not evidence

`metadata_exclude` removes factors before any evaluator sees them:

```yaml
    metadata_exclude: [id, filename, width, height]
```

The usual candidates are identifiers and bookkeeping columns. An `id` is unique per sample, so it correlates
perfectly with everything and reports as maximally informative while telling you nothing. Image geometry is worth
excluding when it is a property of your pipeline rather than of the scene.

Exclusion is recorded in the result. Without the record, an excluded factor is indistinguishable from one never
collected.

## Measure factors off the imagery itself

Most factors arrive with the dataset. Some are properties of the images and have to be measured — brightness,
contrast, sharpness, entropy. `intrinsic_factors` names the **families** to measure and add to the metadata, so
they bin, encode and report like any factor the dataset shipped with:

```yaml
metadata:
  - name: standard
    intrinsic_factors: [visual, pixel]
    continuous_factor_bins:
      brightness: 4
```

Families are lowercase: `visual`, `pixel`, `dimension`, `hash`. Naming anything else is an error before the dataset
is read, listing the ones that exist. `hash` is accepted and measured but never injected — a digest is not a
quantity, and binning one would produce a factor whose codes mean nothing.

Because it lives on the policy, every workflow sharing that policy measures the same set. That is the point: a
statistic injected for one workflow and not another produces two results that look comparable and are not.

### A bare name reaches both levels on detection data

On object detection data a statistic exists twice — once per image and once per detected object — and dataeval names
them `unit_brightness` and `instance_brightness`. Declare the bin on the **bare** statistic and it binds both:

```yaml
    intrinsic_factors: [visual]
    continuous_factor_bins:
      brightness: 4          # binds unit_brightness and instance_brightness alike
```

Declaring `unit_brightness` directly also works, and binds only that one. Pinning the same statistic from both
channels is an error — a level-prefixed entry in a committed `encoding` alongside a bare
`continuous_factor_bins` declaration. The two disagree, and neither is the obvious winner.

A declared bin that matches nothing is not silently dropped. The result envelope records it under
`unmatched_bin_requests`. That is how a typo — `brightnes` — shows up as a fact in the output.

## Declare the range of float image data

`value_range` governs **image statistics**, not metadata factors — but it fails the same quiet way, so it belongs
here.

Integer encodings state the interval their values occupy. Float data does not. As of DataEval v1.1 the statistics
that need one answer `NaN`. Declare it on the **dataset**: it is a fact about the imagery, not a setting of any one
workflow:

```yaml
datasets:
  - name: bathymetry
    format: huggingface
    path: data/bathymetry
    task: image_classification
    value_range: [-50.0, 50.0]      # meters above and below sea level

workflows:
  - name: quality_check
    type: data-cleaning
    outlier_method: modzscore
    outlier_flags: [dimension, pixel, visual]
```

Every workflow reading that dataset then measures against the same interval, including the `intrinsic_factors`
pass above. A declared range then means the same thing to the injection and to the workflow's own
statistics.

Affected without a declared range: the whole `visual` group, pixel histogram and entropy, and dimension depth.
`PIXEL_MISSING` always answers, because it measures the presence of data. Leave `value_range`
unset for ordinary integer imagery — the `[0, 1]` and `0–255` float conventions are still detected automatically.

`value_range` participates in the cache key, so two runs declaring different ranges never share a cached entry.

```{warning}
**Deprecated.** `value_range` on a workflow, and `include_image_stats` on `data-analysis`, are the older spellings
of the two settings above. Both still work and both are removed in the next minor version.
`include_image_stats: true` means `intrinsic_factors: [visual, pixel]`. Setting a workflow's `value_range` alongside
a disagreeing one on the dataset is an error rather than a merge.
```

## Read back what the run did

Every result records its binning decisions. The record appears in the text report under **METADATA FACTORS** and in
the result envelope at `metadata.metadata_binning`:

```text
--------------------------------------------------------------------------------
  METADATA FACTORS
--------------------------------------------------------------------------------
  Auto-bin method: uniform_width
  Excluded:        id
  Unmatched bins:  not_a_factor
    elevation [continuous @ unit] — 4 bins, count declared
        < 68.66         n=4    occupied [41.87, 68.52]
        [68.66, 95.44)  n=19   occupied [68.85, 94.77]
        [95.44, 122.2)  n=25   occupied [96.02, 119.6]
        >= 122.2        n=12   occupied [122.6, 149]
    temp_c [continuous @ unit] — 3 bins, edges declared, 2 empty
        < 0      n=0    empty
        [0, 10)  n=0    empty
        >= 10    n=60   occupied [12.9, 25.07]
    sensor [categorical @ unit] — 3 levels, derived
        a (0): n=15
        b (1): n=22
        c (2): n=23
    file_name [categorical @ unit] — 250 levels, derived (one per sample)
```

Each factor carries two things, and the distinction is the point:

`encoding` — **the policy.** The cut points or the vocabulary, who chose them, and how they were placed. `provenance`
is the field to read: `edges declared` means you said where to cut, `count declared` means you said how many and
DataEval placed them, and `derived` means nobody decided: DataEval chose both from this sample. A factor still
reading `derived` is one nobody has reviewed.

`fit` — **the observation.** How many rows reached each bin in this run, the span they occupied, and which declared
bins nothing reached at all. `2 empty` above says the freezing and cold bins are unpopulated: the cut still applies
and the codes are unchanged, but the data has moved out from under the policy.

Bins are named from their edges. A declared cutoff survives into its own label:
`{"temp_c": [-inf, 0.0, 10.0, inf]}` reads as `< 0>`. The same policy prints the same names over a different draw.

The text report shows per-bucket detail only for a factor with 12 or fewer bins or levels. Above that it gives the
count and how the buckets were populated — `40 levels, derived, n=3–19 per level`, or the occupied span for a binned
factor — so one high-cardinality factor does not bury the rest. A factor holding exactly one level per sample is an
identifier, not a grouping, and is labeled `(one per sample)`. It contributes nothing to balance or
diversity, so it is a candidate for `metadata_exclude`. The envelope is unaffected by the cap.

From Python:

```python
result = run_task(task, config)

binning = result.metadata.metadata_binning
for name, info in binning["factors"].items():
    encoding = info.get("encoding")
    if encoding and encoding["kind"] == "bins":
        print(name, encoding["provenance"], encoding["edges"], "empty:", info["fit"]["empty"])
```

Note `requested_bins` records what was *asked for* and `encoding` records what was *applied*. A request of `10` is a
count; where its nine interior cuts landed is in `encoding["edges"]`.

For `data-analysis` the record is nested one level deeper, under `binning["per_split"][split_name]`.

### Give every split the same cuts

`data-analysis` reads several splits. Encoded independently they land on different cuts for the same factor, because
an automatic bin count comes from each split's own draw. The per-factor statistics then sit side by side in one
report under different alphabets. The **reference split** is encoded first, and every other split takes its
encoding:

```yaml
metadata:
  - name: standard
    reference_split: train      # default: the first split the task names
```

The report says which split set the policy and whether the result holds:

```text
  Splits share one encoding — factor statistics are comparable across them.
```

A vocabulary still grows: a category the reference never saw takes the next free code in the split that has it, so
shared codes keep meaning what they meant and the splits stay comparable. Only a genuinely different cut, or a
reordered vocabulary, makes them not. The report then names the factors responsible.

### Tell whether two results are comparable

Every result carries `metadata.encoding_digest`, a fingerprint of the encoding every factor was read under. It makes
comparing two runs sound: a `Balance` score that moved between them is otherwise unattributable — between *my
override worked* and *the data changed*.

```python
before.metadata.encoding_digest == after.metadata.encoding_digest
# True  -> same cuts; a difference in the numbers is a difference in the data
# False -> different cuts; the numbers are not measuring the same thing
```

The digest covers the policy, not the rows: it stays put when only the data changes, and moves when a cutoff is
declared or a vocabulary grows.

For a multi-split workflow it is set only where **every split shares one encoding**, and is `None` otherwise. The
per-split digests under `binning["per_split"]` say which differed. The text report states it either way:

```text
  Splits were encoded differently — factor statistics are NOT comparable across them.
  Declare cutoffs, or apply one committed encoding to every split.
```

That is the common case with automatic binning: the bin count is derived from each split's own draw. Pinning
the cuts with `metadata_continuous_factor_bins` makes the splits share one encoding and the message change.

### Diagnostics

`metadata.diagnostics` carries the warnings DataEval raised during the run: the ranges it could not resolve, bin
requests it ignored, and factors it dropped. These used to reach only the console and `result.log`. They are now
part of the envelope, so an archived result records why a statistic came back `NaN`.

## Why this matters for comparison

Binning is the reason two evaluations of "the same" data can disagree. A `Balance` score is an association between a
factor's codes and the class labels, and a factor's codes depend on where its cuts fell. Change the sample and an
automatic method may place different cuts. Change the bin count and the entropy the association is normalized
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
