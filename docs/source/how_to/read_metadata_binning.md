# Read the binning a run used

Every result that built metadata records its binning decisions. This guide reads that record in the text report and from
Python, then uses it to tell whether two results are comparable. It needs a result from a workflow that reads metadata
factors. [Configure metadata binning](configure_metadata_binning.md) sets the decisions the record shows.

## Find the record

The record appears in the text report under METADATA FACTORS and in the result envelope at
`metadata.metadata_binning`:

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

A custom workflow's or preset's result records the encodings its steps read in `metadata_binning`: one record, or
`per_split` keyed by Dataset address where the steps read several Datasets or one Dataset two ways. A step that reads
labels alone, as `label-health` does, adds nothing to it, so a split's parts are not recorded when only their labels
are read. A chain whose steps read no factors records none, and its `metadata_binning` and `encoding_digest` are
`null`.

## Read a factor's policy and fit

Each factor carries two things:

- `encoding` is the policy: the cut points or the vocabulary, who chose them, and how they were placed. `provenance` is
  the field to read. `edges declared` means you said where to cut, `count declared` means you said how many and DataEval
  placed them, and `derived` means nobody decided: DataEval chose both from this sample. A factor still reading
  `derived` is one nobody has reviewed.
- `fit` is the observation: how many rows reached each bin in this run, the span they occupied, and which declared
  bins nothing reached at all. `2 empty` above says the freezing and cold bins are unpopulated. The cut still applies
  and the codes are unchanged, but the data no longer matches the cuts the policy set.

Bins are named from their edges. A declared cutoff survives into its own label:
`{"temp_c": [-inf, 0.0, 10.0, inf]}` reads as `< 0>`. The same policy prints the same names over a different draw.

The text report shows per-bucket detail only for a factor with 12 or fewer bins or levels. Above that it gives the
count and how the buckets were populated, such as `40 levels, derived, n=3–19 per level`, or the occupied span for a
binned factor, so one high-cardinality factor does not bury the rest. A factor holding exactly one level per sample is
an identifier, not a grouping, and is labeled `(one per sample)`. It contributes nothing to balance or diversity, so it
is a candidate for the policy's `exclude`. The envelope is unaffected by the cap.

## Read the record from Python

```python
result = run_task(config, task)

binning = result.metadata.metadata_binning
for name, info in binning["factors"].items():
    encoding = info.get("encoding")
    if encoding and encoding["kind"] == "bins":
        print(name, encoding["provenance"], encoding["edges"], "empty:", info["fit"]["empty"])
```

`requested_bins` records what you asked for, and `encoding` records what was applied. A request of `10` is a count.
Where its nine interior cuts landed is in `encoding["edges"]`.

For a preset that reads several splits, such as `audit`, the record is nested one level deeper, under
`binning["per_split"]`, keyed by each split's address: `train`, then `evals[val]` and so on.

## Tell whether two results are comparable

Every result carries `metadata.encoding_digest`, a fingerprint of the encoding every factor was read under. It makes
comparing two runs sound. Without it, you cannot attribute a `Balance` score that moved between them to your override
or to a change in the data.

```python
before.metadata.encoding_digest == after.metadata.encoding_digest
# True  -> same cuts; a difference in the numbers is a difference in the data
# False -> different cuts; the numbers are not measuring the same thing
```

The digest covers the policy, not the rows: it stays put when only the data changes, and moves when a cutoff is
declared or a vocabulary grows. If two runs disagree, diff their `metadata_binning` blocks before concluding the data
moved.

For a multi-split workflow the digest is set only where every split shares one encoding, and is `None` otherwise. The
per-split digests under `binning["per_split"]` say which differed. The text report states it either way:

```text
  Splits were encoded differently — factor statistics are NOT comparable across them.
  Declare cutoffs, or apply one committed encoding to every split.
```

That is the common case with automatic binning: the bin count is derived from each split's own draw. Pinning the cuts
with `continuous_factor_bins` makes the splits share one encoding, and the message changes.

## Read the diagnostics

`metadata.diagnostics` carries the warnings DataEval raised during the run: the ranges it could not resolve, bin
requests it ignored, and factors it dropped. An archived result records why a statistic came back `NaN`.

## Next steps

- [Read the result envelope](read_the_result_envelope.md) — The envelope that holds the binning record.
- [Configure metadata binning](configure_metadata_binning.md) — Pin the bins for any factor you compare across runs,
  with explicit edges by preference.
- [Provenance](../concepts/Provenance.md) — Why a result carries the decisions that shaped it.
