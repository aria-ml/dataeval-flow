# What's New in v0.2

v0.2 upgrades the underlying library to **DataEval v1.1**, introduces a **data coverage** workflow, and adds
committed **metadata encoding policies**. The v0.2.1 patch renames the **CUDA container variants** and restores
**per-split loading** for YOLO datasets.

---

## At a Glance

| Feature / Change | Impact | Action Required |
| --- | --- | --- |
| **DataEval v1.1 Upgrade** | Stored pixel units, chance-corrected `Balance`, and shared split encodings | Recompute baselines before comparing. |
| **New Data Coverage Workflow** | Pre-training gap analysis for classes & embeddings | Configure a `data-coverage` workflow. |
| **Metadata Encoding Policies** | Declared, reviewable, and pinned factor encodings | Define a `metadata:` policy block. |
| **Float Value Range** | Float imagery requires an explicit range to avoid `NaN` | Set `value_range` in your config. |
| **Workflow Cache v1** | v0.1 cache entries are not reusable and everything recomputes | Run `rm -rf <cache_dir>/v0`. |
| **Renamed Config Keys** | `selections`, `steps`, `selection` renamed | Rename to `views`, `operations`, `view`. |
| **CUDA Variants Renamed** (v0.2.1) | `cu118` / `cu128` image tags no longer published | Pull `cu126` / `cu130` instead. |
| **YOLO Split Selection** (v0.2.1) | Per-split loading restored | Add `split:` to YOLO dataset configs. |

---

## 1. New Data Coverage Workflow

Assess label-space, embedding-space, and metadata factor gaps before training.

### How to use

1. Configure the workflow. `ontology` takes an inline mapping or a path to an RDF artifact; without it, a flat
   ontology is synthesized from the dataset's `index2label`:

   ```yaml
   workflows:
     - name: coverage_check
       type: data-coverage
       ontology: config/taxonomy.ttl
   ```

2. Reading an ontology from an RDF file needs the extra:

   ```bash
   pip install "dataeval-flow[ontology]"
   ```

See {doc}`the coverage tutorial <../notebooks/data_coverage>` and {doc}`Declare an ontology <../how_to/declare_an_ontology>`.

---

## 2. Metadata Encoding Policies

Encodings — how continuous factors are cut into bins and categorical factors are mapped to codes — are now
nameable, exportable, and verifiable.

### How to use

1. Define a top-level `metadata:` policy in your configuration:

   ```yaml
   metadata:
     - name: standard
       encoding: policy/factor_bins.json
       continuous_factor_bins:
         elevation: 8

   workflows:
     - name: coverage_check
       type: data-coverage
       metadata: standard
   ```

2. Extract and pin the encoding from an existing run:

   ```bash
   dataeval-flow encoding output/results/result.json -o policy/factor_bins.json
   ```

3. Review the descriptor. Entries still reading `provenance: "derived"` are ones nobody has checked; ratify them
   by setting `provenance` to `"accepted"` or `"declared"`.

See {doc}`Configure metadata binning <../how_to/configure_metadata_binning>`.

---

## 3. Explicit Value Range for Float Imagery

Float imagery has no inherent range. Without one, the visual statistics, pixel histogram and entropy, and
dimension depth answer `NaN`.

### How to use

Set `value_range` directly on any workflow that computes image statistics:

```yaml
workflows:
  - name: quality_check
    type: data-cleaning
    value_range: [-50.0, 50.0]
```

---

## 4. Renames and Removals

### Configuration Keys

The old keys are deprecated but still function. Please update at your convenience:

- `selections` → `views`
- `steps` → `operations`
- `selection` → `view`

```yaml
views:                          # was: selections
  - name: first_5k
    operations:                 # was: steps
      - type: Limit
        n: 5000
```

### Result Fields

- Classwise outlier pivots: `level` is now `count_basis` (`image` or `annotation`).
- Per-factor metadata: `bins`, `bin_count`, `bins_requested`, `binned_by`, `categories`, `category_count`,
  `is_binned`, and `is_digitized` are replaced by `encoding` (the cut) and `fit` (what this run's rows did
  against it).

---

## 5. CUDA Container Variants Renamed (v0.2.1)

The published image tags now track the CUDA builds PyTorch actually publishes. `cu118` and `cu128` are no longer
built, so a `docker pull` naming either will fail.

| Was | Now | Base |
| --- | --- | --- |
| `cu118` | `cu126` | Ubuntu 24.04 |
| `cu128` | `cu130` | Ubuntu 24.04 |

The `cpu` tag is unchanged. The `onnx-gpu` extra split the same way, into `onnx-cu126` and `onnx-cu130`, because an
`onnxruntime-gpu` wheel links against one specific CUDA major and has to match the PyTorch build:

```bash
uv sync --extra cu130 --extra onnx-cu130
```

### How to use

```bash
docker pull harbor.jatic.net/aria/dataeval:cu130   # was: :cu128
```

Host requirements follow the CUDA major: a 525-series or newer driver for `cu126`, 580-series or newer for `cu130`.
The CUDA runtime libraries ship inside the image, so no host CUDA install is needed. See
{doc}`Run workflows in containers <../how_to/containerized_workflows>`.

---

## 6. YOLO Split Selection (v0.2.1)

Selecting one split of a YOLO dataset — lost when the loaders moved from `maite-datasets` to `datamaite` — works
again through a `split` field on the dataset config. Keep `path` on the dataset root so `data.yaml` stays in scope;
pointing it at a split subdirectory falls back to numeric class names.

### How to use

```yaml
datasets:
  - name: yolo_train
    format: yolo
    path: yolo-data          # dataset root: data.yaml + image/label trees
    split: train             # train | val | test; omit to load every split
```

Two optional fields cover non-standard layouts: `yaml_file` for a config that is not at the root under a
conventional name, and `ann_dir` for labels kept outside the `labels/` sibling of `images/`. Both are relative to
`path`.

:::{note}
`images_dir`, `labels_dir`, and `classes_file` on YOLO configs, and `classes_file` on COCO configs, were removed in
v0.2.0 and are rejected by the schema. The shipped example config advertised them by mistake until v0.2.1.
:::

---

## 7. Other Enhancements & Platform Updates

- **Enriched Results**: Result envelopes now include `metadata_binning`, `encoding_digest`, and `diagnostics`.
- **Python 3.14**: Fully supported.
- **Dataset Loaders**: Moved from `maite-datasets` to `datamaite`.
- **New View Operations**: `Crop`, `Resize`, `SelectChannels`, and `TorchvisionTransform` are now supported.
  See {doc}`Narrow a dataset with views <../how_to/build_dataset_views>`.

---

## Upgrade Checklist

1. **Clean the cache**: `rm -rf <cache_dir>/v0` (see
   {doc}`Reuse results with the disk cache <../how_to/reuse_results_with_cache>`).
2. **Recompute baselines**: Recompute stored image statistics and chance-corrected `Balance` rankings.
3. **Set float range**: Add `value_range` for workflows on float imagery.
4. **Update config keys**: Rename `selections` / `steps` / `selection` → `views` / `operations` / `view`.
5. **Pin metadata encodings**: Run `dataeval-flow encoding` to pin and review your metadata encodings.
6. **Update result consumers**: Adjust code that reads `level` or old bin/category fields from output JSONs.
7. **Repoint container pulls**: Replace `cu118` / `cu128` image tags with `cu126` / `cu130`, and swap the `onnx-gpu`
   extra for `onnx-cu126` / `onnx-cu130`.

---

### Related

- {doc}`Change Log <changelog>`
- [DataEval's v1.1 migration guide](https://dataeval.readthedocs.io/en/latest/migration/v1.1.html)
