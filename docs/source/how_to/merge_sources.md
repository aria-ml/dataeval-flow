# Merge sources into one dataset

Two datasets almost never number their classes the same way. Merging composes sources, not datasets, because each
operand needs its own `Relabel`. You conform each operand to a shared vocabulary in a view that its source references,
then name the conformed sources under `merge:` on a new source. The [`merge`](../reference/transforms.md#merge)
transform concatenates datasets inside a chain of steps. This guide needs two datasets and a pipeline that can define a
view for each.

## Conform each dataset and merge them

Name the operands under `merge:` on a source. A source names either `dataset:` or `merge:`, never both, and a `merge:`
names at least two sources:

```yaml
datasets:
  - name: m3fd
    format: coco
    path: m3fd
    annotations_file: annotations/instances_train.json
    images_dir: images
  - name: drone
    format: yolo
    path: drone
    split: train

views:
  - name: conform_m3fd
    operations:
      - type: Relabel
        params:
          class_remap: { people: Person, car: Vehicle, truck: Vehicle }
          target: [Person, Vehicle]
  - name: conform_drone
    operations:
      - type: Relabel
        params:
          class_remap: { pedestrian: Person, van: Vehicle }
          target: [Person, Vehicle]

sources:
  - name: m3fd_conformed
    dataset: m3fd
    view: conform_m3fd
  - name: drone_conformed
    dataset: drone
    view: conform_drone
  - name: merged
    merge: [m3fd_conformed, drone_conformed]
```

An operand's view is applied before the merge, so it is part of the dataset. Everything else a view can do applies to
an operand too: filter, limit, crop.

## Give every operand the same target

`target` is the integer indexing of the merged label space. Operands conformed against differently ordered targets
carry labels that denote different classes. The merge is refused:

```text
merge_datasets requires all datasets to share the same 'index2label'.
```

Fix it in the config. Pass every operand's `Relabel` the same `target` list, in the same order. The `class_remap`
differs per operand. The `target` does not.

## Match merged ids to the original ids

A merge re-keys every datum's id to `<position>:<id>`, where `position` is the operand's place in the `merge:` list.
Two operands that both number an item `0` stay distinguishable. A merged run's ids therefore differ from the ids the
same dataset produces on its own. Strip the prefix to match them back.

## Apply a view after the merge

A merged source may carry its own `view:`, applied to the concatenated dataset. This one names the
`two_classes_sample` view from [Narrow a dataset with views](build_dataset_views.md):

```yaml
sources:
  - name: merged
    merge: [m3fd_conformed, drone_conformed]
    view: two_classes_sample
```

Use it to sample, limit or shuffle across both operands at once, or to `Relabel` the shared target into a coarser one.
Operand views run first, then the merge, then this one.

A merged source's cache key covers every operand and the view it was merged under. See
[Reuse results with the disk cache](reuse_results_with_cache.md).

## Nest a merge inside a merge

An operand may itself be a merged source. A source that merges itself through any path is refused:

```text
Source 'a' merges itself through: a -> b -> a.
```

Nesting more than eight deep is refused too. Flatten a config that hits the bound: name the leaf sources in one
`merge:`.

## Next steps

- [Export a dataset](export_a_dataset.md) — Write a conformed or merged source out as a dataset on disk, with the
  provenance that produced it.
- [Chain steps into a workflow of your own](write_a_custom_workflow.md) — Conform and merge two datasets, remove their
  duplicates, and check what is left, as one chain of steps.
- [Provenance](../concepts/Provenance.md) — What the result envelope records about a merged dataset and the label space
  it was read under.
