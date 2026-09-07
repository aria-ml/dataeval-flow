# Measure band groups

m3fd pairs a visible-light frame with a thermal one in the same record — four channels: three visible, one
infrared. Left alone, every statistic reduces over all four channels together, so a colour statistic and a
thermal one are the same number. Declare `channel_groups:` to measure each one on its own, and a `stats:` policy to
say which statistic families each group gets, whether the image background is measured too, and which views drive
outlier detection and metadata factors.

## Declare channel_groups on the dataset

```yaml
datasets:
  - name: m3fd
    format: coco
    path: m3fd
    value_range: [0, 255]
    channel_groups: {rgb: [0, 1, 2], ir: 3}
```

A group becomes a set of `<name>_<statistic>` columns alongside the unprefixed whole-image ones — `rgb_brightness`
beside `brightness`. Write one band as a bare index (`ir: 3`) or several as a list (`rgb: [0, 1, 2]`).

Declare it on the dataset, not on a workflow, because that channel 3 is infrared is a fact about the sensor and not
a decision any one workflow makes. Every workflow reading m3fd then sees the same groups. `value_range` and an
ontology belong on the dataset for the same reason — see {doc}`configure_metadata_binning` and
{doc}`declare_an_ontology`.

`channel_groups:` alone measures nothing. It only makes the groups available to name from a `stats:` policy.

## Declare a stats policy

```yaml
stats:
  - name: multispectral
    measure:
      - {bands: ~,   families: [dimension, hash]}
      - {bands: rgb, families: [visual]}
      - {bands: ir,  families: [pixel]}
    background: true
    outliers_from: [~]
    factors_from:  [~, rgb, ir]

workflows:
  - name: clean
    type: data-cleaning
    stats: multispectral
    outlier_method: modzscore
    outlier_flags: [visual]
```

Reference a policy by name from any workflow that computes image statistics — `data-cleaning`, `data-analysis`,
`data-coverage`, `ood-detection`, `data-prioritization`, and `parameter-sweep` all take a `stats:` name. Two
workflows naming the same policy measure the same things, so their results are comparable.

### measure is a complete statement

`measure` lists every view that gets computed, and nothing else. A view with no entry is not measured — nothing is
inferred for a missing whole-image entry, even though every workflow measured the whole image before band groups
existed. Write out `{bands: ~, ...}` if you want it.

`bands: ~` names the whole image. `bands: rgb` and `bands: ir` name groups the dataset declares under
`channel_groups`; naming a group the dataset does not declare is refused before the dataset is read.

Geometry does not vary with a band subset, so `rgb_width` is never produced. A `measure` entry asking only
`dimension` of a band group is refused, because it would compute nothing — ask `dimension` of the whole image
instead.

### Measure the background

`background: true` adds a further pass over every pixel none of the item's boxes cover, describing the scene it was
captured in rather than the things annotated within it. It composes with `channel_groups:` — `background_rgb`
columns are the visible-light background, `background_ir` the thermal one.

Only `pixel` and `visual` are measured for the background. Hash and dimension are computed for the image and its
boxes as usual and skipped for the background, which has no meaningful hash and no geometry of its own.

### The two consumer lists

`outliers_from` and `factors_from` each name the views whose columns one consumer reads, by the prefix those
columns carry. Both default to `[~]` — the whole image alone.

- `outliers_from` decides which columns `data-cleaning`'s outlier detection sees.
- `factors_from` decides which columns become metadata factors, for `Balance` and `Diversity` to read.

Which statistic *families* get injected as factors at all is a separate decision — `intrinsic_factors` on the
metadata policy (see {doc}`configure_metadata_binning`). `factors_from` decides which *views* those families are
read from.

## The view namespace

A view is named by the prefix its columns carry, and channel groups compose with the background: for `n` groups
plus `background: true`, the full naming vocabulary has `2n + 2` forms. m3fd's two groups give six:

| Prefix | Columns from | Example |
| --- | --- | --- |
| `~` | the whole image | `brightness`, `width` |
| `rgb` | the rgb group | `rgb_brightness` |
| `ir` | the ir group | `ir_mean` |
| `background` | the scene behind every box, of the whole image | `background_fraction` |
| `background_rgb` | the scene behind every box, of the rgb group | `background_rgb_brightness` |
| `background_ir` | the scene behind every box, of the ir group | `background_ir_mean` |

Two band groups plus a background already reach this six-view vocabulary, and each view can carry more than one
statistic family, so the columns a `measure` block computes multiply far past what any one consumer reads. Call
`produced_views()` on your own policy to see which views it actually produces, rather than assuming a count. Ask
for what each consumer in `outliers_from` and `factors_from` actually reads, not more.

`measure` is a complete statement, so a policy only produces the views its entries ask for. The `multispectral`
policy above produces five of the six: `~`, `rgb`, `ir`, `background_rgb`, and `background_ir`. Its whole-image
entry asks for `dimension` and `hash` only, and neither is measured for the background, so it has no plain
`background` view. The section on reading `background_fraction`, below, shows what that means and how to get one.

## Enabling a band group moves nothing until you name it

Adding a channel group to a dataset, or a new entry to `measure`, changes nothing `data-cleaning` flags and nothing
`Balance` or `Diversity` reads. `outliers_from` and `factors_from` both default to `[~]`, so until a view appears in
one of them, its columns are computed and cached but nothing downstream reads them.

That is deliberate. A group appearing in `channel_groups:` must not move a cleaning result or a bias number until a
config names it. In the `multispectral` policy above, `outliers_from: [~]` keeps `data-cleaning`'s outlier
detection exactly what it flagged before band groups existed. `factors_from: [~, rgb, ir]` opts the two groups in
for bias analysis instead. Drop them from `factors_from` and `Balance` and `Diversity` go back to reading the
whole-image factors alone, whatever `measure` computes.

## Duplicate detection reads the unprefixed hashes only

DataEval's duplicate detector looks up `xxhash`, `phash`, `dhash`, `phash_d4`, and `dhash_d4` by those exact names.
A `hash` family measured on a band group produces `rgb_phash`, not `phash` — a column the detector never looks at.

Ask `hash` of a group and it is measured and reported: the column is computed, cached, and available to
`factors_from` like any other. It detects no duplicate, because nothing reads a prefixed name as a hash. There is
no `duplicates_from` field. Duplicate detection is always checked against the whole image alone, so a field naming
views for it could only ever say `[~]`.

## Read background_fraction first

`background: true` always computes `background_fraction` — the share of the image left unmasked — whatever
families you ask for elsewhere. Read it before any other `background_*` number: a background statistic measured
over a few percent of an image is noise.

`background_fraction` belongs to the plain `background` view, and that view is only produced if the whole-image
entry itself measures `pixel` or `visual`. The `multispectral` policy above asks `dimension` and `hash` of the
whole image, so it has no plain `background` view to name — only `background_rgb` and `background_ir`, from the
groups that do measure something background-eligible. Ask `pixel` or `visual` of `~` too, and name `background`
alongside the views you already read:

```yaml
stats:
  - name: multispectral
    measure:
      - {bands: ~,   families: [dimension, pixel, hash]}
      - {bands: rgb, families: [visual]}
      - {bands: ir,  families: [pixel]}
    background: true
    outliers_from: [~]
    factors_from:  [~, rgb, ir, background]
```

`Balance` and `Diversity` now see `background_fraction` alongside the whole-image and group factors, so you can
read it against `background_mean` and `background_std` before trusting either one.

## Related material

- {doc}`configure_outlier_detection` — the statistic families `outlier_flags` names, and the methods that turn a
  measurement into a flag
- {doc}`configure_metadata_binning` — `intrinsic_factors`, `value_range`, and the metadata policy `factors_from`
  composes with
- {doc}`reuse_results_with_cache` — what a stats cache entry keys on, and why a narrower `measure` still shares one
- {doc}`API Reference <../reference/autoapi/dataeval_flow/index>` — every field on `StatsPolicyConfig` and
  `StatsConfigMixin`
