# Export a dataset

Declare an `exports:` block to write a {term}`source <Source>` out as a dataset on disk, together with the provenance
that produced it. It takes a conformed dataset out of the tool: merged, relabeled, and filtered as the config
describes. This guide covers declaring an export, the available formats, what the emitted dataset carries, and what it
does not.

## Declare an export

An export names a source and a format:

```yaml
exports:
  - name: conformed_dataset
    source: merged
    format: coco
```

Each export is written to `output/datasets/<name>/`, beside `output/results/`. A run with no output directory writes
no exports.

An export names a source, not a task, so it is written whether or not any task reads that source. A config that
declares no tasks at all still writes its exports.

`name:` must be a single directory segment. A `/` or a `\` is refused, as are `.` and `..`.

An export that fails is logged and does not cost the run its other exports. The run then exits non-zero.

## Choose a format

Every format writes an object-detection dataset. An export naming a classification source fails.

| `format` | Written as |
| --- | --- |
| `coco` (default) | A COCO detection root: an annotation JSON plus the images |
| `yolo` | A YOLO/Ultralytics dataset root |
| `huggingface_vision` | A Hugging Face ImageFolder repository with `objects` metadata |
| `visdrone` | Official VisDrone-DET split roots |

Each sample keeps its split, so a format that lays out per-split directories keeps `val` out of `train`.

## Choose what happens to an occupied destination

```yaml
exports:
  - name: conformed_dataset
    source: merged
    mode: replace
```

| `mode` | Effect |
| --- | --- |
| `error` (default) | Refuses a destination that already holds a dataset |
| `replace` | Clears the destination first |
| `append` | Writes into what is already there |

The default refuses so a re-run cannot overwrite a dataset somebody is using. `replace` empties the destination
before the write and does not restore it if the write then fails. `append` may leave stale files behind that a reload
of the destination would pick up.

## Declare the ontology on the export

An export names no workflow, so it inherits no {term}`ontology <Ontology>`. Declare one on the export itself, by name
under the top-level `ontologies:` key or as a path:

```yaml
ontologies:
  - name: vehicles
    source: config/label_ontology.jsonld

exports:
  - name: conformed_dataset
    source: merged
    ontology: vehicles
```

If you omit it while a workflow declares one, the digests will not match: the dataset carries a label-space identity
the run's envelope does not. The run warns and names the workflows that declare one, so you can copy the value
across.

## Read the provenance the export writes

Every export writes a `provenance.json` beside the dataset, shaped `{"runs": [...]}` with one entry per write. Each
entry records:

- the tool and its version, and when the write happened,
- the source that was written, the ontology it was read under, and that ontology's digest,
- one entry per merge operand: its source name, its dataset, the view it read through, and the `class_remap` that
  conformed it,
- the `label_space` records, each with its `digest`.

`mode: append` adds an entry rather than replacing the file, so a directory written twice records both writes.

Only the COCO writer also embeds the same mapping in its own `info` block. The other three drop it, which is why the
sidecar is written for every format.

The `label_space` digests are the values the {term}`result envelope <Result Envelope>` carries and a `data-coverage`
audit stamps. Declare the same ontology on the audit, on the workflows, and on the export, and one digest match ties
an emitted dataset to the run that produced it and to the audit that justified its vocabulary.

## Know what an export drops

An export is not lossless. Detections are rebuilt from the realized dataset, because a view can filter and relabel
boxes and nothing then maps an output detection back to the annotation it came from. Each box keeps its geometry and
its class. Dropped are:

- the source annotation id,
- `area`, `segmentation`, and `iscrowd`,
- the per-detection attributes.

Read the source's own files where you need those.

## Know when the imagery is copied and when it is re-encoded

Images are referenced by path where the source is file-backed and its view left the imagery alone, so a write copies
files. That keeps an export cheap.

The realized pixels are encoded in two cases: a source that is not file-backed, and a view that changed the image
size or the channel count. `Crop`, `Resize`, and `SelectChannels` all produce imagery the file on disk no longer
matches.

Encoded imagery must be 8-bit, and grayscale or three-channel. An export refuses any other dtype or channel count.
Drop the view operation that normalizes the pixel range or changes the channel count.

## Export a merged source

A merged source exports like any other, with two differences.

Each emitted `file_name` is prefixed with its operand's position, as `0/img_000.png`. Two operands numbering their
images independently would otherwise overwrite each other on disk.

The merged datum id is carried as `source_id` in the per-image metadata, so `0:17` in the dataset stays traceable to
item `17` of the first operand.

## Export the parts of a split

A `data-splitting` entry, run as a step of a custom workflow, hands on its parts as `<step>.train`, `<step>.val` and
`<step>.test`; `train` is the rebalanced one where the entry sets `rebalance:`. An `export` step writes any of them.
The workflow below splits an object-detection source and writes its train and test:

```yaml
workflows:
  - name: splits
    type: data-splitting
    test_frac: 0.2
    val_frac: 0.1

  - name: split_and_export
    inputs: [data]
    steps:
      - {name: split, workflow: splits, input: data}
      - {name: train_set, transform: export, input: split.train, format: coco}
      - {name: test_set, transform: export, input: split.test, format: coco}

tasks:
  - name: build_splits
    workflow: split_and_export
    sources: [train]
```

Each is written under `datasets/<task>.<step>/`, here `datasets/build_splits.train_set/`. With `folds: 3` the entry
runs `kfold`, and `train` and `val` are lists keyed by fold, `"0"` to `"2"`; an `export` step handed one writes each
element under its key, as `datasets/build_splits.train_set/0/`. `test` is one Dataset shared by every fold.

```yaml
workflows:
  - name: splits
    type: data-splitting
    folds: 3
    test_frac: 0.2
```

Only object-detection Datasets can be exported. A classification source's parts can be read as `split.train` by other
steps, such as an evaluator, but `export` does not yet write them.

## Related material

- {doc}`build_dataset_views` — the views that conform and merge a dataset before it is exported
- {doc}`declare_an_ontology` — declaring the label space an export records
- [Provenance](../concepts/Provenance.md) — the label-space digest that joins an export, a run, and an audit
- {doc}`API Reference <../reference/autoapi/dataeval_flow/index>` — every field on `ExportConfig`
