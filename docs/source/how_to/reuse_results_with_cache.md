# Reuse results with the disk cache

Embedding extraction and image statistics dominate the runtime of most workflows, and re-running an unchanged pipeline
should not pay for them twice. This guide covers turning on the disk-backed {term}`cache <Caching>`, what it stores,
what invalidates it, and how to clean it up.

## Used in these tutorials

- {doc}`Assess dataset coverage <../notebooks/data_coverage>`
- {doc}`Parameter Sweep for Data Cleaning <../notebooks/parameter_sweep>`

## Turn it on

Without a cache directory, DataEval Flow caches **in memory only**: artifacts are reused within a process, but nothing
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
  harbor.jatic.net/aria/dataeval:cu128
```

From Python:

```python
from pathlib import Path

result = run_task(task, config, cache_dir=Path("./cache"))
```

## What gets cached

Four artifact types, each the expensive part of some workflow:

| Artifact | Stored as | Produced by |
| --- | --- | --- |
| Embeddings | `.npy` | Any extractor |
| Cluster results | `.npz` | Cluster-based outlier and duplicate detection |
| Metadata | `.parquet` + `.json` | Metadata factor analysis |
| Image statistics | `.parquet` + `.json` | Statistical outlier detection, analysis, coverage |

Statistics accumulate incrementally. Two workflows that request different `ImageStats` flags share one cache entry and
only the missing metrics are computed — so adding `visual` to a run that already computed `dimension` and `pixel` pays
for `visual` alone.

## What invalidates it

The cache is keyed on the things that would change the answer, so a hit means the inputs really were identical:

- **The dataset** — a content fingerprint of the resolved dataset.
- **The view** — a hash of the {term}`view <View>` applied to it, so the same dataset under two views keeps two
  separate entries.
- **The artifact configuration** — for embeddings, the extractor's model, parameters, preprocessor, and the content
  hash of the model file itself; for statistics, the set of metrics requested.

Change any of those and the next run recomputes. Change nothing and it does not. This is the same property that makes
a result reproducible — see [Reproducibility](../concepts/Reproducibility.md).

Two practical consequences:

- **Seed your shuffles.** An unseeded `Shuffle` produces a different view every run, so the view hash changes and
  nothing is ever reused. See {doc}`build_dataset_views`.
- **Swapping a model file invalidates its embeddings**, even at the same path, because the key includes the file's
  content hash.

## Layout and cleanup

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
version is bumped and older data is simply ignored — it is not deleted, so it is worth removing stale versions
yourself:

```bash
rm -rf ./cache/v1        # drop a superseded cache version
rm -rf ./cache           # start completely clean
```

Removing the cache is always safe; the next run recomputes.

## One dataset per cache

All artifacts within a single cache instance must come from the same dataset. DataEval Flow keys the on-disk layout by
dataset so this holds automatically for normal use — the constraint matters only if you are constructing a
`DatasetCache` directly, where mixing datasets in one instance would make invalidation unsound.

## Related material

- [Reproducibility](../concepts/Reproducibility.md) — why config-keyed caching and reproducible results are the same
  mechanism
- {doc}`build_dataset_views` — how a view participates in the cache key
- {doc}`Container Reference <../reference/containers>` — the `/cache` mount and `DATAEVAL_CACHE` defaults
