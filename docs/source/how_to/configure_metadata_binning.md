# Configure metadata binning

Bias, balance, diversity and parity read your metadata factors as **codes**: a continuous factor cut into intervals, a
categorical one mapped to ordinals. Where the cuts fall changes the numbers those evaluators report. A `Balance` score,
for example, is the association between a factor's codes and the class labels, and a change of bin count also changes
the entropy that association is normalized against. Binning is therefore a parameter of the evaluation, and a
`metadata:` policy sets it.

The [`bias`](../reference/presets.md#bias) and [`audit`](../reference/presets.md#audit) workflows read the policy. This
guide defines a policy, pins the cuts, excludes factors, measures factors off the imagery and gives every split the
same cuts. To check what a run did with the policy, see [Read the binning a run used](read_metadata_binning.md). The
[DataEval Dataset Bias and Coverage explanation](https://dataeval.readthedocs.io/en/latest/concepts/DatasetBias.html)
covers the balance and diversity measures that read the codes.

## Find the workflows that read a policy

These workflows read metadata factors under the policy their `metadata:` names:

| Workflow | Reads metadata for |
| --- | --- |
| `audit` | its [`factor-triage`](../reference/evaluators.md#factor-triage), [`factor-summary`](../reference/evaluators.md#factor-summary), [`balance`](../reference/evaluators.md#balance), [`diversity`](../reference/evaluators.md#diversity), [`factor-gaps`](../reference/combines.md#factor-gaps) and [`factor-leakage`](../reference/evaluators.md#factor-leakage) steps: unreadable factors, per-factor summaries, shortcut risk, diversity, factor gaps and group leakage, with every split encoded like train |
| `bias` | its [`label-health`](../reference/evaluators.md#label-health), `factor-summary`, `balance`, `diversity`, [`parity`](../reference/evaluators.md#parity) and `factor-gaps` steps: class counts, per-factor summaries, shortcut risk, diversity, factor-class association, and the factor gap analysis with factor-to-class mutual information |

`quality`, `splits` and `shift` take a policy's name in `metadata:` too. `scope` reads no factors and refuses one. The
[Assess dataset coverage](../notebooks/data_coverage.py) tutorial excludes factors and pins bin counts in a policy
for `bias`.

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
  - name: bias_check
    type: bias
    metadata: standard
  - name: release_audit
    type: audit
    metadata: standard        # same policy; the encoding digests prove it
    outliers: {flags: [dimension, pixel, visual], outlier_threshold: adaptive}
```

A policy carries everything that decides how a factor becomes a code: `encoding`, `factor_levels`, `strict`,
`auto_bin_method`, `exclude`, `continuous_factor_bins`, `intrinsic_factors`, `factor_source`, and
`reference_split`. The {doc}`API Reference <../reference/autoapi/dataeval_flow/index>` lists every field on
`MetadataConfigMixin` and `StatsConfigMixin`.

Everything a policy says is checked before the dataset is read, and a mistake costs only a message:

| The config says | What happens |
| --- | --- |
| `metadata:` names no policy in the pool | Error, listing the policies that do exist |
| `encoding` points at a file that is not there | Error, because a descriptor matching nothing silently reverts every factor it meant to pin |
| One factor named by both `encoding` and `continuous_factor_bins` | Error |
| `strict: true` over a descriptor with unreviewed vocabularies | Error, naming them (see the "Close a vocabulary you have reviewed" section) |
| `intrinsic_factors` names something that is not a family | Error, listing the families that exist |
| A statistic pinned by both a level-prefixed `encoding` entry and a bare `continuous_factor_bins` name | Error |

### Get a descriptor out of a run

You do not write the descriptor by hand. Every run records the encoding it used, and a run with `-o` writes it
beside the results:

```text
output/results/
  result.json      # the record, per factor
  result.txt
  result.html
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
it. Pass `--task <name>` to take one. Splits work the same way; see `reference_split` in the "Give every split the same
cuts" section.

### Start from a recommended policy

[`triage`](../reference/presets.md#triage) recommends a policy as its last finding. It holds the fixes triage suggests
plus a pin for every factor your policy left unpinned, read from this data: explicit edges for each cut, and the
vocabulary, as `factor_levels`, for each categorical factor. A remap value triage could not read is dropped to missing
and marked for you to decide.

```yaml
# Every cut and vocabulary below was read from this data. Pinning them keeps later runs comparable, and it also
# commits them to this data's assumptions. If this data does not represent the data you expect, results computed under
# this policy can be invalid or misleading. Review each factor before you commit it.
metadata:
- name: standard
  continuous_factor_bins:
    altitude: [-.inf, 265.8536348826109, 504.1481565221535, 742.442678161696, .inf]  # seen 27.5591 to 980.737; the outer bins are open, so values beyond this range join them
```

The caveat matters. The outer bins are open, so a value far beyond anything this data held lands in the outermost bin
beside the largest values it did hold, and a category it never saw takes a new code. Pinning a policy read from a sample
that does not represent your data fixes that sample's assumptions for every later run.

The recommendation is inline YAML you review in the config, one line per factor. A descriptor from
`dataeval-flow encoding` holds the same cuts in a file, which suits large vocabularies and sharing one encoding
across configs. The [Triage a dataset's metadata](../notebooks/metadata_triage.py) tutorial reviews and applies a
recommended policy.

### Apply a committed encoding

`encoding` points at a descriptor, the artifact `dataeval-flow encoding` writes, held under the data root and
committed alongside the config. It applies the recorded cuts and vocabularies. That is what makes two runs over
different data comparable:

```yaml
metadata:
  - name: locked
    encoding: policy/factor_bins.json
```

Factors the descriptor does not name are encoded normally. A vocabulary it does name grows by appending: a
category the descriptor never saw takes the next free code, so codes already assigned keep meaning what they meant.

### Close a vocabulary you have reviewed

`strict: true` makes a value outside a declared vocabulary an error. Use it for a fixed taxonomy: the vocabulary
stays closed, and new data is reported.

It is refused over a descriptor whose vocabularies still read `provenance: "derived"`, which is what a descriptor
exported from an exploratory run contains. `strict` does not consult provenance. Allowing it there would enforce a
taxonomy nobody decided on and fail the run on the first new category:

```text
Metadata policy 'locked' sets strict, which closes every vocabulary in its descriptor, but
['weather'] still read provenance="derived" — nobody reviewed them. Ratify them in the
descriptor (set provenance to "accepted" or "declared"), or drop strict.
```

Ratify the entries you have looked at by editing `provenance` to `accepted` in the committed descriptor. Make that
edit in a pull request.

## Let the method choose the cuts

A policy's `auto_bin_method` picks how an un-pinned continuous factor is discretized.

```yaml
metadata:
  - name: standard
    auto_bin_method: uniform_width

workflows:
  - name: bias_check
    type: bias
    metadata: standard
```

| Method | Cuts the range into | Use when |
| --- | --- | --- |
| `uniform_width` | equal-width intervals | the scale is meaningful and you want bins that read as ranges |
| `uniform_count` | intervals holding equal counts | the distribution is skewed and equal-width bins would leave most bins near-empty |
| `clusters` | intervals found from the data's own structure | the values group naturally and you want the groups, not a grid |

Leave it unset to take DataEval's default (`uniform_width`).

:::{important}
The bin count an automatic method lands on is derived from the data. Two runs over
different samples of the same population can therefore produce different bin counts for the same factor. That is
enough to move a `Balance` score. Pin the count with the policy's `continuous_factor_bins` for any factor whose numbers
you intend to compare across runs.
:::

## Pin the cuts for a specific factor

A policy's `continuous_factor_bins` overrides the automatic method per factor. Give it a bin count, or explicit
edges when the boundaries carry domain meaning:

```yaml
    continuous_factor_bins:
      elevation: 8                          # eight bins, placed by the auto method
      temperature: [-40, 0, 20, 40, 60]     # explicit edges, four bins
```

Explicit edges are the stronger choice for anything comparative. They are a property of the configuration, and
the same edges apply to every run and every split. Naming a factor the dataset does not carry is not an error.
DataEval ignores it and warns, and the run records it as an unmatched request (see the "Bin both levels of a
statistic on detection data" section).

## Drop factors that are not evidence

A policy's `exclude` removes factors before any evaluator sees them:

```yaml
    exclude: [id, filename, width, height]
```

The usual candidates are identifiers and bookkeeping columns. An `id` is unique per sample, so it correlates
perfectly with everything and reports as maximally informative while telling you nothing. Image geometry is worth
excluding when it is a property of your pipeline rather than of the scene.

Exclusion is recorded in the result. Without the record, an excluded factor is indistinguishable from one never
collected.

## Measure factors off the imagery itself

Most factors arrive with the dataset. Some are properties of the images and have to be measured: brightness,
contrast, sharpness, entropy. `intrinsic_factors` names the families to measure and add to the metadata, so
they bin, encode and report like any factor the dataset shipped with:

```yaml
metadata:
  - name: standard
    intrinsic_factors: [visual, pixel]
    continuous_factor_bins:
      brightness: 4
```

Families are lowercase: `visual`, `pixel`, `dimension`, `hash`. Naming anything else is an error before the dataset
is read, listing the ones that exist. `hash` is accepted and measured but never injected, because a digest is not a
quantity, and binning one would produce a factor whose codes mean nothing.

Because it lives on the policy, every workflow sharing that policy measures the same set. A statistic injected for one
workflow and not another produces two results that look comparable and are not. To measure named channel groups or the
background as well, see [Measure band groups](measure_band_groups.md).

## Bin both levels of a statistic on detection data

On object detection data a statistic exists twice, once per image and once per detected object, and dataeval names
them `unit_brightness` and `instance_brightness`. Declare the bin on the bare statistic and it binds both:

```yaml
    intrinsic_factors: [visual]
    continuous_factor_bins:
      brightness: 4          # binds unit_brightness and instance_brightness alike
```

Declaring `unit_brightness` directly also works, and binds only that one. Pinning the same statistic from both
channels is an error: a level-prefixed entry in a committed `encoding` alongside a bare
`continuous_factor_bins` declaration. The two disagree, and neither is the obvious winner.

A declared bin that matches nothing is not silently dropped. The result envelope records it under
`unmatched_bin_requests`. That is how a typo, such as `brightnes`, shows up as a fact in the output.

## Give every split the same cuts

`audit` reads several splits. Encoded independently they land on different cuts for the same factor, because
an automatic bin count comes from each split's own draw. The per-factor statistics then sit side by side in one
report under different alphabets. The reference split is encoded first, and every other split takes its
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
shared codes keep meaning what they meant and the splits stay comparable. Only a different cut, or a
reordered vocabulary, makes them not. The report then names the factors responsible.

A preset that names a reference split encodes every split like it too, and a policy's `reference_split` names a
different one. Custom workflows encode each Dataset on its own draw. The
[Audit a set of splits before training](../notebooks/audit.py) tutorial sets `reference_split` to `train`.

## Next steps

- [Triage a dataset's metadata](../notebooks/metadata_triage.py) — Review the policy that `triage` recommends and apply
  it.
- [Read the binning a run used](read_metadata_binning.md) — Confirm the cuts a run applied and whether two results are
  comparable.
- [Reproducibility](../concepts/Reproducibility.md) — Why pinned binning is the same property that makes a result
  reproducible.
