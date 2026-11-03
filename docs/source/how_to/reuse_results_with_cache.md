# Reuse results with the disk cache

Embedding extraction and image statistics dominate the runtime of most workflows, and re-running an unchanged pipeline
should not pay for them twice. The disk-backed {term}`cache <Caching>` keeps them between runs. This guide turns it on,
lists what it stores and what invalidates it, and cleans it up.

## Turn it on

Without a cache directory, DataEval Flow caches in memory only: artifacts are reused within a process, but nothing
survives the run. Point it at a directory to persist across runs.

From the CLI:

```bash
dataeval-flow --config params.yaml --data . --output ./results --cache ./cache
```

By environment variable:

```bash
export DATAEVAL_CACHE=./cache
```

In the container, mount a writable volume at `/cache` and it is used automatically:

```bash
docker run \
  --mount type=bind,source=/path/to/data,target=/dataeval,readonly \
  --mount type=bind,source=/path/to/output,target=/output \
  --mount type=bind,source=/path/to/cache,target=/cache \
  harbor.jatic.net/aria/dataeval-flow:latest-cu130
```

The [Container Reference](../reference/containers.md) lists the `/cache` mount and the `DATAEVAL_CACHE` defaults.

From Python:

```python
from pathlib import Path

result = run_task(config, task, cache_dir=Path("./cache"))
```

## Check what gets cached

Four artifact types, each the expensive part of some workflow:

| Artifact | Stored as | Produced by |
| --- | --- | --- |
| Embeddings | `.npy` | Any extractor |
| Cluster results | `.npz` | Cluster-based outlier and duplicate detection |
| Metadata | `.parquet` + `.json` | Metadata factor analysis |
| Image statistics | `.parquet` + `.json` | Statistical outlier detection, analysis, coverage |

Statistics accumulate incrementally. Two workflows that request different `ImageStats` flags share one cache entry and
only the missing metrics are computed. Adding `visual` to a run that already computed `dimension` and `pixel` pays
for `visual` alone. The same holds for [band groups and the background](measure_band_groups.md): a policy adding
an `ir` group or `background: true` to a dataset already measured computes the `ir_*` or `background_*` columns alone.
The entry records the bands and range each group's columns were measured over, and a group whose definition changed is
measured again, replacing its old columns.

## Check what invalidates it

The cache is keyed on the things that would change the answer, so a hit means the inputs were identical:

- The dataset: a content fingerprint of the resolved dataset.
- The view: a hash of the {term}`view <View>` applied to it, so the same dataset under two views keeps two
  separate entries. Changing a view invalidates only that view's artifacts. A merged source's key covers every operand
  and the view it was merged under, so narrowing one operand's view invalidates that dataset and nothing else.
- The artifact configuration: for embeddings, the extractor's model, parameters, preprocessor, and the content
  hash of the model file itself. For statistics, whether images or targets are measured and the dataset's
  `value_range`.

An extractor that fits itself to the data, such as BoVW, is fitted once per task on the first source to ask. Its
embeddings, and the clusters built from them, are keyed by that source, the batch size that picked its fitting images,
and the `seed` it was fitted under. They are only reused under the same vocabulary. Without a `seed` each fit learns a
different vocabulary, so they are kept only for the task that computed them and recomputed for every task.

Change any of those and the next run recomputes. Change nothing and it does not.

Two practical consequences:

- Seed your shuffles. An unseeded `Shuffle` produces a different view every run, so the view hash changes and
  nothing is ever reused. See [Narrow a dataset with views](build_dataset_views.md).
- Swapping a model file invalidates its embeddings, even at the same path, because the key includes the file's
  content hash.

## Clean up the cache

```text
cache_dir/
  v{CACHE_VERSION}/
    {dataset_name}_{dataset_config_hash}/
      sel_{view_hash}/
        embeddings_{config_hash}.npy
        clusters_{config_hash}.npz
        metadata.parquet
        stats_{scope_hash}.parquet
```

Artifacts live under a version directory so that formats can coexist. When the cache format changes incompatibly the
version is bumped and older data is ignored. It is not deleted, so it is worth removing stale versions
yourself.

The version tracks releases rather than individual format changes, so it advances at most once per release and the
numbering matches what you can have on disk. The v0.1 line shipped `v0`; the current version is `v1`.
v0.2 moved to it because pixel statistics are now cached in the units the data is stored in, which made every `v0`
statistic incomparable with a fresh one, and because metadata archives now serve only the binning configuration
that wrote them.

```bash
rm -rf ./cache/v0        # drop the superseded v0.1 cache
rm -rf ./cache           # start completely clean
```

Removing the cache is always safe; the next run recomputes.

## Keep one dataset per cache

All artifacts within a single cache instance must come from the same dataset. DataEval Flow keys the on-disk layout by
dataset so this holds automatically for normal use. The constraint matters only if you are constructing a
`DatasetCache` directly, where mixing datasets in one instance would make invalidation unsound.

## Next steps

- [Assess dataset coverage](../notebooks/data_coverage.py) — Reuse image statistics and embeddings across runs with
  `cache_dir`.
- [Tune data cleaning with a matrix](../notebooks/tune_data_cleaning.py) — Nine runs over one draw of the data share its
  statistics, embeddings and clusters.
- [Reproducibility](../concepts/Reproducibility.md) — Why config-keyed caching and reproducible results are the same
  mechanism.
