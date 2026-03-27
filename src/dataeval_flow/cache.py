"""Workflow cache for expensive computations — optionally disk-backed.

.. warning::

   All cached values within a single ``DatasetCache`` instance **must** be
   sourced from the same dataset.  Do **not** mix artifacts from different
   datasets in one cache — this prevents cross-dataset dependencies and
   ensures cache invalidation is straightforward.

When ``cache_dir`` is provided, ``DatasetCache`` persists artifacts to
disk so that subsequent runs reuse previously computed results.  When
``cache_dir`` is ``None``, the cache operates in **memory-only** mode:
disk loads always miss and disk saves are no-ops, but computed values are
still held in the per-instance ``_memory`` dict and the
``load_or_compute_*`` methods still provide a single code-path for
obtaining each artifact.

A global singleton registry (keyed by ``dataset_name``) ensures that the
same **non-disk-backed** in-memory cache is reused across multiple
``run_task`` calls for the same dataset configuration — use
:meth:`DatasetCache.get_or_create` to take advantage of this.

Provides load/save methods for four component types:

- **Embeddings** — Dense numpy arrays stored as ``.npy``
- **Cluster results** — Clustering output stored as ``.npz``
- **Metadata** — Polars DataFrame as ``.parquet`` + auxiliary attributes as ``.json``
- **Stats** — Unified ``StatsResult`` stored as ``.parquet`` + ``.json``.
  Metrics accumulate incrementally: different workflows requesting different
  ``ImageStats`` flags share the same cache entry and only compute the
  missing metrics.

Cache layout (disk-backed mode)::

    cache_dir/
      v{CACHE_VERSION}/
        {dataset_name}_{dataset_config_hash}/
          sel_{selection_hash}/
            embeddings_{config_hash}.npy
            clusters_{config_hash}.npz
            metadata.parquet
            metadata.json
            stats_{scope_hash}.parquet
            stats_{scope_hash}.json

Cache artifacts are stored under a ``v{CACHE_VERSION}`` subdirectory so
that different versions can coexist side-by-side.  When the cache format
changes in a backwards-incompatible way, bump ``CACHE_VERSION`` and old
data is simply ignored.  Users can clean up stale versions by removing the
old ``v*/`` directories (e.g. ``rm -rf /cache/v1``).
"""

__all__ = [
    "CACHE_VERSION",
    "FLAG_TO_METRIC",
    "METRIC_TO_FLAG",
    "DatasetCache",
    "active_cache",
    "get_or_compute_cluster_result",
    "get_or_compute_embeddings",
    "get_or_compute_metadata",
    "get_or_compute_stats",
    "missing_flags",
    "scope_key",
    "dataset_fingerprint",
    "selection_repr",
]

import contextvars
import hashlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl
from dataeval.core import ClusterResult
from dataeval.flags import ImageStats
from dataeval.protocols import AnnotatedDataset, Array
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dataeval import Metadata

_logger = logging.getLogger(__name__)

# Bump this when the on-disk cache format changes in a backwards-incompatible
# way.  Cached artifacts are stored under ``v{CACHE_VERSION}/`` so different
# versions coexist and users can ``rm -rf`` old directories to reclaim space.
CACHE_VERSION = "0"

# Default for the ``persist_memory`` constructor parameter.  When ``True``
# (the default), ``DatasetCache`` instances hold computed artifacts in an
# in-memory dict so that repeated requests for the same artifact within a
# process are served instantly.
DEFAULT_PERSIST_MEMORY: bool = True

# Active cache context — set via :func:`active_cache` so that downstream
# convenience functions can discover the cache without explicit parameters.
_active_cache: contextvars.ContextVar[tuple["DatasetCache", str] | None] = contextvars.ContextVar(
    "_active_cache", default=None
)


@contextmanager
def active_cache(cache: "DatasetCache", selection_key: str) -> Generator[None]:
    """Set the active cache context for the duration of the block.

    Convenience functions (:func:`get_or_compute_stats`, etc.) will
    automatically use this cache and selection key when called inside
    the ``with`` block.  Outside an ``active_cache`` block (or when
    ``cache`` is ``None``), they compute directly without caching.

    Parameters
    ----------
    cache : DatasetCache
        The cache instance to activate.
    selection_key : str
        Selection key (from :func:`selection_repr`).
    """
    token = _active_cache.set((cache, selection_key))
    try:
        yield
    finally:
        _active_cache.reset(token)


# ---------------------------------------------------------------------------
# Flag ↔ metric-name mappings
# ---------------------------------------------------------------------------

METRIC_TO_FLAG: dict[str, ImageStats] = {
    # Pixel
    "mean": ImageStats.PIXEL_MEAN,
    "std": ImageStats.PIXEL_STD,
    "var": ImageStats.PIXEL_VAR,
    "skew": ImageStats.PIXEL_SKEW,
    "kurtosis": ImageStats.PIXEL_KURTOSIS,
    "entropy": ImageStats.PIXEL_ENTROPY,
    "missing": ImageStats.PIXEL_MISSING,
    "zeros": ImageStats.PIXEL_ZEROS,
    "histogram": ImageStats.PIXEL_HISTOGRAM,
    # Visual
    "brightness": ImageStats.VISUAL_BRIGHTNESS,
    "contrast": ImageStats.VISUAL_CONTRAST,
    "darkness": ImageStats.VISUAL_DARKNESS,
    "sharpness": ImageStats.VISUAL_SHARPNESS,
    "percentiles": ImageStats.VISUAL_PERCENTILES,
    # Dimension
    "offset_x": ImageStats.DIMENSION_OFFSET_X,
    "offset_y": ImageStats.DIMENSION_OFFSET_Y,
    "width": ImageStats.DIMENSION_WIDTH,
    "height": ImageStats.DIMENSION_HEIGHT,
    "channels": ImageStats.DIMENSION_CHANNELS,
    "size": ImageStats.DIMENSION_SIZE,
    "aspect_ratio": ImageStats.DIMENSION_ASPECT_RATIO,
    "depth": ImageStats.DIMENSION_DEPTH,
    "center": ImageStats.DIMENSION_CENTER,
    "distance_center": ImageStats.DIMENSION_DISTANCE_CENTER,
    "distance_edge": ImageStats.DIMENSION_DISTANCE_EDGE,
    "invalid_box": ImageStats.DIMENSION_INVALID_BOX,
    # Hash
    "xxhash": ImageStats.HASH_XXHASH,
    "phash": ImageStats.HASH_PHASH,
    "dhash": ImageStats.HASH_DHASH,
    "phash_d4": ImageStats.HASH_PHASH_D4,
    "dhash_d4": ImageStats.HASH_DHASH_D4,
}

FLAG_TO_METRIC: dict[ImageStats, str] = {v: k for k, v in METRIC_TO_FLAG.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_hash(config_data: str) -> str:
    """Generate an 8-char hex hash from a config string."""
    return hashlib.sha256(config_data.encode()).hexdigest()[:8]


def _file_content_hash(path: str | Path) -> str:
    """Return an 8-char hex hash of a file's contents, or 'missing' if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:8]
    except OSError:
        return "missing"


def _extractor_config_key(extractor_config: Any) -> str:
    """Build a cache key from an extractor config, hashing model file contents.

    The Pydantic ``model_dump_json`` only serializes config fields (path,
    layer name, etc.) — not the model weights.  If the user retrains and
    overwrites the file at the same path, the JSON is identical but the
    embeddings should differ.  We append a content hash of the model file
    so the cache key changes when the weights change.
    """
    config_json = extractor_config.model_dump_json(exclude_defaults=False)
    model_path = getattr(extractor_config, "model_path", None)
    if model_path is not None:
        config_json += f"|file_hash={_file_content_hash(model_path)}"
    return config_json


def _atomic_write(target: Path, data_fn: Callable[[Path], Any], *, suffix: str = ".tmp") -> None:
    """Write to a temp file in the same directory, then atomically rename."""
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=suffix)
    tmp_path = Path(tmp)
    try:
        os.close(fd)
        data_fn(tmp_path)
        tmp_path.rename(target)  # atomic on POSIX same-filesystem
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_pair(
    target_a: Path,
    data_fn_a: Callable[[Path], Any],
    target_b: Path,
    data_fn_b: Callable[[Path], Any],
) -> None:
    """Write two files atomically as a pair.

    Both files are written to temp locations first.  Only after both
    writes succeed are they renamed into place.  If the process crashes
    between the two renames, the next load will see a missing partner
    file and treat it as a cache miss (both files must exist for a hit).
    """
    fd_a, tmp_a = tempfile.mkstemp(dir=target_a.parent, suffix=".tmp")
    tmp_path_a = Path(tmp_a)
    fd_b, tmp_b = tempfile.mkstemp(dir=target_b.parent, suffix=".tmp")
    tmp_path_b = Path(tmp_b)
    try:
        os.close(fd_a)
        os.close(fd_b)
        data_fn_a(tmp_path_a)
        data_fn_b(tmp_path_b)
        # Both writes succeeded — rename into place.
        tmp_path_a.rename(target_a)
        tmp_path_b.rename(target_b)
    except BaseException:
        tmp_path_a.unlink(missing_ok=True)
        tmp_path_b.unlink(missing_ok=True)
        # Clean up any already-renamed target if the second rename failed
        # but the first succeeded — removes the orphaned file so the next
        # load sees a clean miss rather than a mismatched pair.
        if not tmp_path_a.exists() and target_a.exists():
            target_a.unlink(missing_ok=True)
        raise


_MAX_DS_ID_BYTES = 100  # Conservative limit (ext4 NAME_MAX = 255 bytes)


def _make_dataset_id(name: str, cache_key: str) -> str:
    """Build a cache-safe dataset identifier.

    Hashes *cache_key* so that changes to any config field produce a
    distinct cache directory.  The result is ``{name_prefix}_{hash}``
    where the prefix keeps it human-readable and the hash guarantees
    uniqueness.

    Parameters
    ----------
    name : str
        Human-readable dataset name (used as prefix).
    cache_key : str
        Opaque string whose content uniquely identifies the dataset
        configuration.  Provided by :func:`resolve_dataset`.
    """
    config_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]

    max_prefix = _MAX_DS_ID_BYTES - 17  # 17 = 1 ("_") + 16 (hash)
    prefix = name.encode("utf-8")[:max_prefix].decode("utf-8", errors="ignore")
    return f"{prefix}_{config_hash}"


def selection_repr(dataset: Any) -> str:
    """Build a deterministic cache key from the (possibly selected) dataset.

    For datasets wrapped with ``Select``, hashes the *resolved* indices so
    that non-deterministic selections (e.g. Shuffle) produce distinct cache
    keys.  For unwrapped datasets, returns ``"sel:all"``.

    Parameters
    ----------
    dataset : MaiteDataset | Select
        The dataset *after* selection has been applied.

    Returns
    -------
    str
        Deterministic string suitable for cache key hashing.
    """
    resolve = getattr(dataset, "resolve_indices", None)
    if resolve is not None:
        indices = resolve()
        idx_str = ",".join(str(i) for i in indices)
        idx_hash = hashlib.sha256(idx_str.encode()).hexdigest()[:16]
        return f"sel:n={len(indices)}:{idx_hash}"
    return "sel:all"


def dataset_fingerprint(dataset: Any) -> str:
    """Build a content-based fingerprint by hashing a sample of datum tuples.

    Hashes the first 5, middle 5, and last 5 datum tuples (or all data
    if the dataset has 15 or fewer items) plus the dataset length using
    xxHash.  Each element of the tuple (image, target, metadata) is
    hashed so that label or metadata changes also invalidate the cache.

    Parameters
    ----------
    dataset : Dataset
        Any object supporting ``__len__`` and ``__getitem__`` that
        returns ``(image, target, ...)`` tuples where elements are
        array-like or have a bytes-serialisable ``repr``.

    Returns
    -------
    str
        Hex digest fingerprint of the sampled data.
    """
    import xxhash as xxh
    from dataeval.utils._internal import as_numpy

    n = len(dataset)
    hasher = xxh.xxh3_64()

    # Include dataset length so additions/removals are detected even if
    # the sampled items happen to remain unchanged.
    hasher.update(n.to_bytes(8, "little"))

    # Sample indices: first 5 + middle 5 + last 5, or all if <= 15.
    if n <= 15:
        indices = list(range(n))
    else:
        mid = n // 2
        indices = list(range(5)) + list(range(mid - 2, mid + 3)) + list(range(n - 5, n))

    for idx in indices:
        datum = dataset[idx]
        datum = datum if isinstance(datum, tuple) else (datum,)
        for element in datum:
            if isinstance(element, Array):
                hasher.update(as_numpy(element).ravel().tobytes())
            else:
                hasher.update(repr(element).encode("utf-8"))

    return hasher.hexdigest()


def scope_key(
    per_image: bool = True,
    per_target: bool = True,
    per_channel: bool = False,
) -> str:
    """Build a deterministic scope key from per_image/per_target/per_channel settings.

    Stats computed with different scope settings have incompatible
    ``source_index`` arrays and cannot be merged.  The scope key ensures
    they are cached separately.
    """
    parts: list[str] = []
    if per_image:
        parts.append("img")
    if per_target:
        parts.append("tgt")
    if per_channel:
        parts.append("ch")
    return "+".join(parts) or "none"


def missing_flags(cached_metrics: set[str], desired_flags: ImageStats) -> ImageStats:
    """Compute the ``ImageStats`` flags not yet covered by cached metrics.

    Parameters
    ----------
    cached_metrics : set[str]
        Metric names already present in the cache.
    desired_flags : ImageStats
        The flags the workflow wants computed.

    Returns
    -------
    ImageStats
        Flags that still need to be computed.  ``ImageStats.NONE`` if all
        desired metrics are already cached.
    """
    uncovered = ImageStats.NONE
    for flag in ImageStats:
        # Only consider individual (atomic, single-bit) flags
        if flag.value and (flag.value & (flag.value - 1)) == 0 and flag in desired_flags:
            metric_name = FLAG_TO_METRIC.get(flag)
            if metric_name and metric_name not in cached_metrics:
                uncovered |= flag
    # Re-resolve so that dependencies of missing flags are included
    return uncovered if uncovered else ImageStats.NONE


# ---------------------------------------------------------------------------
# Shared compute helpers (used by both convenience functions and DatasetCache)
# ---------------------------------------------------------------------------


def _do_compute_stats(
    dataset: AnnotatedDataset[Any],
    desired_flags: ImageStats,
    per_image: bool = True,
    per_target: bool = True,
    per_channel: bool = False,
) -> dict[str, Any]:
    """Compute stats and return as a plain dict."""
    from dataeval.core._compute_stats import compute_stats

    return dict(
        compute_stats(
            dataset,
            stats=desired_flags,
            per_image=per_image,
            per_target=per_target,
            per_channel=per_channel,
            normalize_pixel_values=True,
        )
    )


def _do_compute_metadata(
    dataset: AnnotatedDataset[Any],
    auto_bin_method: Any = None,
    exclude: Sequence[str] | None = None,
    continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = None,
) -> "Metadata":
    """Build metadata from a dataset."""
    from dataeval_flow.metadata import build_metadata

    return build_metadata(
        dataset, auto_bin_method=auto_bin_method, exclude=exclude, continuous_factor_bins=continuous_factor_bins
    )


def _do_compute_embeddings(
    dataset: AnnotatedDataset[Any],
    extractor_config: Any,
    transforms: Any = None,
    batch_size: int | None = None,
) -> NDArray[Any]:
    """Extract embeddings and flatten to 2-D."""
    from dataeval_flow.embeddings import build_embeddings

    embeddings = build_embeddings(dataset, extractor_config, transforms, batch_size)
    array: NDArray[Any] = np.asarray(embeddings)
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return array


def _do_compute_clusters(
    embeddings: NDArray[Any],
    algorithm: Literal["kmeans", "hdbscan"],
    n_clusters: int | None,
) -> ClusterResult:
    """Run clustering on embeddings."""
    from dataeval.core._clusterer import cluster

    return cluster(embeddings, algorithm=algorithm, n_clusters=n_clusters)


# ---------------------------------------------------------------------------
# Convenience functions (delegate to DatasetCache via active_cache context)
# ---------------------------------------------------------------------------


def get_or_compute_stats(
    desired_flags: ImageStats,
    dataset: AnnotatedDataset[Any],
    per_image: bool = True,
    per_target: bool = True,
    per_channel: bool = False,
) -> dict[str, Any]:
    """Centralized stats computation with context-aware caching.

    Uses the :func:`active_cache` context when set, otherwise computes
    directly without any caching.
    """
    ctx = _active_cache.get()
    if ctx is not None:
        cache, sel_key = ctx
        return cache.load_or_compute_stats(
            sel_key,
            scope_key(per_image, per_target, per_channel),
            desired_flags,
            dataset,
            per_image=per_image,
            per_target=per_target,
            per_channel=per_channel,
        )
    _logger.info("Computing stats (no cache)")
    return _do_compute_stats(dataset, desired_flags, per_image, per_target, per_channel)


def get_or_compute_metadata(
    dataset: AnnotatedDataset[Any],
    auto_bin_method: Any = None,
    exclude: Sequence[str] | None = None,
    continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = None,
) -> "Metadata":
    """Build metadata with context-aware caching.

    Uses the :func:`active_cache` context when set, otherwise computes
    directly without any caching.
    """
    ctx = _active_cache.get()
    if ctx is not None:
        cache, sel_key = ctx
        return cache.load_or_compute_metadata(
            sel_key,
            dataset,
            auto_bin_method=auto_bin_method,
            exclude=exclude,
            continuous_factor_bins=continuous_factor_bins,
        )
    _logger.info("Building metadata (no cache)")
    return _do_compute_metadata(dataset, auto_bin_method, exclude, continuous_factor_bins)


def get_or_compute_embeddings(
    dataset: AnnotatedDataset[Any],
    extractor_config: Any,
    transforms: Any = None,
    batch_size: int | None = None,
) -> NDArray[Any]:
    """Extract embeddings with context-aware caching.

    Uses the :func:`active_cache` context when set, otherwise computes
    directly without any caching.
    """
    ctx = _active_cache.get()
    if ctx is not None:
        cache, sel_key = ctx
        config_json = _extractor_config_key(extractor_config)
        transforms_key = repr(transforms) if transforms is not None else "none"
        return cache.load_or_compute_embeddings(
            sel_key,
            config_json,
            transforms_key,
            dataset,
            extractor_config,
            transforms,
            batch_size,
        )
    _logger.info("Computing embeddings (no cache)")
    return _do_compute_embeddings(dataset, extractor_config, transforms, batch_size)


def get_or_compute_cluster_result(
    embeddings: NDArray[Any],
    algorithm: Literal["kmeans", "hdbscan"],
    n_clusters: int | None,
    extractor_config: Any = None,
    transforms: Any = None,
) -> ClusterResult:
    """Compute cluster result with context-aware caching.

    Uses the :func:`active_cache` context when set, otherwise computes
    directly without any caching.
    """
    ctx = _active_cache.get()
    if ctx is not None:
        cache, sel_key = ctx
        config_json = _extractor_config_key(extractor_config) if extractor_config is not None else "none"
        transforms_key = repr(transforms) if transforms is not None else "none"
        return cache.load_or_compute_cluster_result(
            sel_key,
            config_json,
            transforms_key,
            embeddings,
            algorithm,
            n_clusters,
        )
    _logger.info("Computing clusters (no cache)")
    return _do_compute_clusters(embeddings, algorithm, n_clusters)


# ---------------------------------------------------------------------------
# DatasetCache
# ---------------------------------------------------------------------------


class DatasetCache:
    """Cache for dataset computations — optionally disk-backed.

    When *cache_dir* is a :class:`~pathlib.Path`, artifacts are persisted
    under::

        cache_dir / v{CACHE_VERSION} / dataset_name / sel_{hash} / {component}_{cfg}.{ext}

    When *cache_dir* is ``None``, the cache operates in **memory-only**
    mode: disk loads always miss and disk saves are no-ops.  The
    ``load_or_compute_*`` methods still work — they just always compute
    on first access and then serve from the in-memory ``_memory`` dict.

    Use :meth:`get_or_create` to reuse an existing instance for the same
    ``(cache_dir, dataset_name)`` pair — this preserves the in-memory
    cache across multiple ``run_task`` calls.

    Parameters
    ----------
    cache_dir : Path | None
        Root directory for cache storage.  ``None`` disables disk persistence.
    dataset_name : str
        Dataset identifier (from ``TaskConfig.dataset``).
    """

    # Global singleton registry for **non-disk-backed** instances only.
    # Disk-backed caches already have durable storage; singletons would
    # just hold duplicate references and complicate lifetime management.
    _instances: dict[str, "DatasetCache"] = {}
    _instances_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_or_create(cls, cache_dir: Path | None, name: str, cache_key: str) -> "DatasetCache":
        """Return an existing instance for this key, or create a new one.

        Derives the dataset identifier from *name* and *cache_key* via
        :func:`_make_dataset_id`.

        For **non-disk-backed** caches (``cache_dir is None``), a global
        singleton keyed by *dataset_name* is returned so that computed
        artifacts are reused across multiple ``run_task`` calls.

        For **disk-backed** caches, a fresh instance is always created
        (the disk itself provides persistence).
        """
        dataset_name = _make_dataset_id(name, cache_key)
        if cache_dir is not None:
            return cls(cache_dir, dataset_name)
        with cls._instances_lock:
            instance = cls._instances.get(dataset_name)
            if instance is not None:
                return instance
            instance = cls(None, dataset_name)
            cls._instances[dataset_name] = instance
            return instance

    @classmethod
    def clear_instances(cls) -> None:
        """Remove all cached singleton instances.

        Call this to release memory held by non-disk-backed caches
        (e.g. between test runs or when datasets are no longer needed).
        """
        with cls._instances_lock:
            cls._instances.clear()

    def __init__(
        self,
        cache_dir: Path | None,
        dataset_name: str,
        *,
        persist_memory: bool = DEFAULT_PERSIST_MEMORY,
    ) -> None:
        """Initialize the cache with the root directory and dataset name."""
        if cache_dir is not None and ("/" in dataset_name or "\\" in dataset_name or dataset_name in (".", "..")):
            raise ValueError(
                f"Invalid dataset_name for cache (must not contain path separators or be '.'/'..'): {dataset_name!r}"
            )
        self._cache_dir = cache_dir
        self._dataset_name = dataset_name
        self._dataset_dir: Path | None = None
        self._persist_memory = persist_memory
        # Nested in-memory cache: {selection_id: {object_id: value}}
        # The dataset_id level is implicit (one DatasetCache per dataset).
        self._memory: dict[str, dict[str, Any]] = {}

    # =====================================================================
    # In-memory helpers
    # =====================================================================

    def _mem_get(self, selection_key: str, object_key: str) -> Any | None:
        """Return a cached value from memory, or ``None``."""
        if not self._persist_memory:
            return None
        sel = self._memory.get(selection_key)
        if sel is None:
            return None
        return sel.get(object_key)

    def _mem_set(self, selection_key: str, object_key: str, value: Any) -> None:
        """Store a value in the in-memory cache."""
        if self._persist_memory:
            self._memory.setdefault(selection_key, {})[object_key] = value

    # =====================================================================
    # Properties
    # =====================================================================

    @property
    def disk_backed(self) -> bool:
        """Whether this cache persists artifacts to disk."""
        return self._cache_dir is not None

    @property
    def cache_dir(self) -> Path | None:
        """Root cache directory, or ``None`` if memory-only."""
        return self._cache_dir

    @property
    def dataset_name(self) -> str:
        """Dataset identifier."""
        return self._dataset_name

    @property
    def dataset_dir(self) -> Path | None:
        """Dataset-specific cache directory (version-namespaced).

        Returns ``None`` when not disk-backed.
        """
        if self._cache_dir is None:
            return None
        if self._dataset_dir is None:
            d = self._cache_dir / f"v{CACHE_VERSION}" / self._dataset_name
            d.mkdir(parents=True, exist_ok=True)
            self._dataset_dir = d
        return self._dataset_dir

    # =====================================================================
    # Embeddings (.npy)
    # =====================================================================

    def _selection_dir(self, selection_repr: str) -> Path | None:
        """Return (and create) the selection-specific subdirectory."""
        dd = self.dataset_dir
        if dd is None:
            return None
        s = _config_hash(selection_repr)
        d = dd / f"sel_{s}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _embeddings_path(
        self,
        selection_repr: str,
        extractor_config_json: str,
        transforms_repr: str,
    ) -> Path | None:
        sel_dir = self._selection_dir(selection_repr)
        if sel_dir is None:
            return None
        h = _config_hash(extractor_config_json + "|" + transforms_repr)
        return sel_dir / f"embeddings_{h}.npy"

    def load_embeddings(
        self,
        selection_repr: str,
        extractor_config_json: str,
        transforms_repr: str = "none",
    ) -> NDArray[Any] | None:
        """Load cached embedding array, or ``None`` on miss."""
        obj_key = f"embeddings_{_config_hash(extractor_config_json + '|' + transforms_repr)}"
        cached = self._mem_get(selection_repr, obj_key)
        if cached is not None:
            _logger.debug("Memory hit: embeddings for %s/%s", self._dataset_name, selection_repr)
            return cached

        path = self._embeddings_path(selection_repr, extractor_config_json, transforms_repr)
        if path is None or not path.exists():
            return None
        _logger.info("Cache hit: embeddings for %s/%s", self._dataset_name, selection_repr)
        try:
            arr = np.load(path, allow_pickle=False)
            self._mem_set(selection_repr, obj_key, arr)
            return arr
        except Exception:  # noqa: BLE001
            _logger.warning(
                "Failed to load embeddings from cache for %s/%s — recomputing",
                self._dataset_name,
                selection_repr,
                exc_info=True,
            )
            return None

    def save_embeddings(
        self,
        selection_repr: str,
        extractor_config_json: str,
        transforms_repr: str,
        array: NDArray[Any],
    ) -> None:
        """Persist embedding array to cache (no-op on disk when not disk-backed)."""
        obj_key = f"embeddings_{_config_hash(extractor_config_json + '|' + transforms_repr)}"
        self._mem_set(selection_repr, obj_key, array)

        path = self._embeddings_path(selection_repr, extractor_config_json, transforms_repr)
        if path is None:
            return
        _atomic_write(path, lambda p: np.save(p, array), suffix=".npy")
        _logger.info("Cache save: embeddings for %s/%s (%s)", self._dataset_name, selection_repr, path.name)

    def load_or_compute_embeddings(
        self,
        selection_repr: str,
        extractor_config_json: str,
        transforms_repr: str,
        dataset: AnnotatedDataset[Any],
        extractor_config: Any,
        transforms: Any = None,
        batch_size: int | None = None,
    ) -> NDArray[Any]:
        """Load cached embeddings or compute, cache, and return them.

        Parameters
        ----------
        selection_repr : str
            Selection key (from :func:`selection_repr`).
        extractor_config_json : str
            JSON representation of the extractor config (for cache key).
        transforms_repr : str
            String representation of transforms (for cache key).
        dataset
            The dataset to extract embeddings from on cache miss.
        extractor_config
            ``ExtractorConfig`` used to build the extractor.
        transforms
            Optional preprocessing transforms.
        batch_size : int | None
            Batch size for extraction.

        Returns
        -------
        NDArray
            2-D embedding array of shape ``(N, D)``.
        """
        cached = self.load_embeddings(selection_repr, extractor_config_json, transforms_repr)
        if cached is not None:
            return cached

        _logger.info("Computing embeddings for %s/%s", self._dataset_name, selection_repr)
        array = _do_compute_embeddings(dataset, extractor_config, transforms, batch_size)
        self.save_embeddings(selection_repr, extractor_config_json, transforms_repr, array)
        return array

    # =====================================================================
    # Cluster results (.npz)
    # =====================================================================

    def _cluster_path(
        self,
        selection_repr: str,
        extractor_config_json: str,
        transforms_repr: str,
        algorithm: Literal["kmeans", "hdbscan"],
        n_clusters: int | None,
    ) -> Path | None:
        sel_dir = self._selection_dir(selection_repr)
        if sel_dir is None:
            return None
        h = _config_hash(f"{extractor_config_json}|{transforms_repr}|{algorithm}|{n_clusters}")
        return sel_dir / f"clusters_{h}.npz"

    def load_cluster_result(
        self,
        selection_repr: str,
        extractor_config_json: str,
        transforms_repr: str,
        algorithm: Literal["kmeans", "hdbscan"],
        n_clusters: int | None,
    ) -> dict[str, Any] | None:
        """Load cached ClusterResult, or ``None`` on miss."""
        obj_key = f"clusters_{_config_hash(f'{extractor_config_json}|{transforms_repr}|{algorithm}|{n_clusters}')}"
        cached = self._mem_get(selection_repr, obj_key)
        if cached is not None:
            _logger.debug("Memory hit: cluster result for %s/%s", self._dataset_name, selection_repr)
            return cached

        path = self._cluster_path(selection_repr, extractor_config_json, transforms_repr, algorithm, n_clusters)
        if path is None or not path.exists():
            return None
        _logger.info("Cache hit: cluster result for %s/%s", self._dataset_name, selection_repr)
        try:
            data = np.load(path, allow_pickle=False)
            result = {
                "clusters": data["clusters"],
                "mst": data["mst"],
                "linkage_tree": data["linkage_tree"],
                "membership_strengths": data["membership_strengths"],
                "k_neighbors": data["k_neighbors"],
                "k_distances": data["k_distances"],
            }
            self._mem_set(selection_repr, obj_key, result)
            return result
        except Exception:  # noqa: BLE001
            _logger.warning(
                "Failed to load cluster result from cache for %s/%s — recomputing",
                self._dataset_name,
                selection_repr,
                exc_info=True,
            )
            return None

    def save_cluster_result(
        self,
        selection_repr: str,
        extractor_config_json: str,
        transforms_repr: str,
        algorithm: Literal["kmeans", "hdbscan"],
        n_clusters: int | None,
        result: dict[str, Any],
    ) -> None:
        """Persist ClusterResult to cache as .npz (no-op on disk when not disk-backed)."""
        obj_key = f"clusters_{_config_hash(f'{extractor_config_json}|{transforms_repr}|{algorithm}|{n_clusters}')}"
        self._mem_set(selection_repr, obj_key, result)

        path = self._cluster_path(selection_repr, extractor_config_json, transforms_repr, algorithm, n_clusters)
        if path is None:
            return
        _atomic_write(
            path,
            lambda p: np.savez(
                p,
                clusters=result["clusters"],
                mst=result["mst"],
                linkage_tree=result["linkage_tree"],
                membership_strengths=result["membership_strengths"],
                k_neighbors=result["k_neighbors"],
                k_distances=result["k_distances"],
            ),
            suffix=".npz",
        )
        _logger.info("Cache save: cluster result for %s/%s (%s)", self._dataset_name, selection_repr, path.name)

    def load_or_compute_cluster_result(
        self,
        selection_repr: str,
        extractor_config_json: str,
        transforms_repr: str,
        embeddings: NDArray[Any],
        algorithm: Literal["kmeans", "hdbscan"],
        n_clusters: int | None,
    ) -> ClusterResult:
        """Load cached cluster result or compute, cache, and return it."""
        cached = self.load_cluster_result(
            selection_repr,
            extractor_config_json,
            transforms_repr,
            algorithm,
            n_clusters,
        )
        if cached is not None:
            return ClusterResult(**cached)

        _logger.info("Computing clusters for %s/%s (algorithm=%s)", self._dataset_name, selection_repr, algorithm)
        cluster_result = _do_compute_clusters(embeddings, algorithm, n_clusters)

        self.save_cluster_result(
            selection_repr,
            extractor_config_json,
            transforms_repr,
            algorithm,
            n_clusters,
            dict(cluster_result),
        )
        return cluster_result

    # =====================================================================
    # Metadata (.parquet + .json sidecar)
    # =====================================================================

    def _metadata_paths(self, selection_repr: str) -> tuple[Path, Path] | tuple[None, None]:
        sel_dir = self._selection_dir(selection_repr)
        if sel_dir is None:
            return None, None
        return sel_dir / "metadata.parquet", sel_dir / "metadata.json"

    def load_metadata(
        self,
        selection_repr: str,
        dataset: AnnotatedDataset[Any],
        auto_bin_method: Any = None,
        exclude: Sequence[str] | None = None,
        continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = None,
    ) -> "Metadata | None":
        """Load cached raw Metadata, or ``None`` on miss.

        Reconstructs a ``dataeval.Metadata`` instance from the cached
        parquet + JSON files with ``_is_binned = False``.  The caller's
        binning configuration is applied so that ``_bin()`` runs lazily
        with the current settings when factor data is first accessed.
        """
        obj_key = "metadata"
        cached = self._mem_get(selection_repr, obj_key)
        if cached is not None:
            _logger.debug("Memory hit: metadata for %s/%s", self._dataset_name, selection_repr)
            return cached

        from dataeval import Metadata as MetadataClass

        pq_path, json_path = self._metadata_paths(selection_repr)
        if pq_path is None or json_path is None or not pq_path.exists() or not json_path.exists():
            return None

        _logger.info("Cache hit: metadata for %s/%s", self._dataset_name, selection_repr)

        try:
            df = pl.read_parquet(pq_path)
            with open(json_path, encoding="utf-8") as f:
                aux = json.load(f)

            # Reconstruct Metadata without calling __init__ (which requires a dataset).
            # We bypass __init__ and set internal attributes directly so that
            # _structure() is never invoked (it requires a live dataset).
            # _is_binned is False so _bin() will run lazily with the caller's config.
            meta = object.__new__(MetadataClass)
            meta._dataframe = df  # noqa: SLF001
            meta._is_structured = True  # noqa: SLF001  # skip _structure()
            meta._is_binned = False  # noqa: SLF001  # _bin() will run lazily
            meta._dataset = dataset  # noqa: SLF001
            meta._has_targets = aux.get("has_targets")  # noqa: SLF001
            meta._count = aux["item_count"]  # noqa: SLF001
            meta._class_labels = np.asarray(aux["class_labels"], dtype=np.intp)  # noqa: SLF001
            meta._index2label = {int(k): v for k, v in aux["index2label"].items()}  # noqa: SLF001
            meta._item_indices = np.asarray(aux["item_indices"], dtype=np.intp)  # noqa: SLF001
            meta._dropped_factors = {}  # noqa: SLF001
            meta._image_factors = set(aux.get("image_factors", []))  # noqa: SLF001
            meta._target_factors = set(aux.get("target_factors", []))  # noqa: SLF001
            meta._raw = []  # noqa: SLF001
            meta._exclude = set(exclude or ())  # noqa: SLF001
            meta._include = set()  # noqa: SLF001
            meta._continuous_factor_bins = dict(continuous_factor_bins) if continuous_factor_bins else {}  # noqa: SLF001
            meta._auto_bin_method = auto_bin_method or "uniform_width"  # noqa: SLF001
            meta._target_factors_only = False  # noqa: SLF001
            # Build _factors dict from image/target factor sets
            meta._build_factors()  # noqa: SLF001

            # Smoke-test the reconstructed object: access commonly-used
            # public properties so that missing attributes surface here
            # (at cache-load time) rather than causing a late AttributeError
            # if upstream Metadata adds new internal state.
            meta.class_labels  # noqa: B018
            meta.index2label  # noqa: B018
            meta.item_count  # noqa: B018

            self._mem_set(selection_repr, obj_key, meta)
            return meta
        except Exception:  # noqa: BLE001
            _logger.warning(
                "Failed to load Metadata from cache for %s/%s — recomputing",
                self._dataset_name,
                selection_repr,
                exc_info=True,
            )
            return None

    def save_metadata(
        self,
        selection_repr: str,
        metadata: "Metadata",
    ) -> None:
        """Persist raw (pre-binned) Metadata to cache (no-op on disk when not disk-backed).

        Only the structured DataFrame is saved — binned/digitized columns
        (``↕`` / ``#`` suffixes) are stripped so that a single cache entry
        can be reused across different binning configurations.
        """
        self._mem_set(selection_repr, "metadata", metadata)

        pq_path, json_path = self._metadata_paths(selection_repr)
        if pq_path is None or json_path is None:
            return

        # .dataframe triggers _structure() but NOT _bin(), giving us
        # the raw structured data.  Drop any binned/digitized columns
        # that may exist if _bin() was already called on this object.
        df = metadata.dataframe
        drop_cols = [c for c in df.columns if c.endswith("↕") or c.endswith("#")]
        if drop_cols:
            df = df.drop(drop_cols)

        # Auxiliary attributes — factor sets instead of factor_info
        aux = {
            "class_labels": metadata.class_labels.tolist(),
            "index2label": {str(k): v for k, v in metadata.index2label.items()},
            "item_indices": metadata.item_indices.tolist(),
            "item_count": metadata.item_count,
            "image_factors": sorted(metadata._image_factors),  # noqa: SLF001
            "target_factors": sorted(metadata._target_factors),  # noqa: SLF001
            "has_targets": metadata._has_targets,  # noqa: SLF001
        }
        _atomic_write_pair(
            pq_path,
            lambda p: df.write_parquet(p),
            json_path,
            lambda p: p.write_text(json.dumps(aux, sort_keys=True), encoding="utf-8"),
        )

        _logger.info("Cache save: metadata for %s/%s (%s)", self._dataset_name, selection_repr, pq_path.name)

    def load_or_compute_metadata(
        self,
        selection_repr: str,
        dataset: AnnotatedDataset[Any],
        auto_bin_method: Any = None,
        exclude: Sequence[str] | None = None,
        continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = None,
    ) -> "Metadata":
        """Load cached metadata or build, cache, and return it.

        The cache stores only raw (pre-binned) metadata keyed by dataset
        selection.  On hit, the caller's binning configuration is applied
        so that ``_bin()`` runs lazily.  On miss, the metadata is built,
        raw data is saved, and the full object is returned.

        Parameters
        ----------
        selection_repr : str
            Selection key (from :func:`selection_repr`).
        dataset
            The dataset to build metadata from on cache miss.
        auto_bin_method
            Method for automatic binning of continuous values.
        exclude : list[str] | None
            Metadata columns to exclude.
        continuous_factor_bins : dict[str, int | list[float]] | None
            Number of uniform bins (int) or explicit bin edges (list[float])
            for specific continuous factors.

        Returns
        -------
        Metadata
            DataEval Metadata instance.
        """
        cached = self.load_metadata(
            selection_repr,
            dataset,
            auto_bin_method=auto_bin_method,
            exclude=exclude,
            continuous_factor_bins=continuous_factor_bins,
        )
        if cached is not None:
            return cached

        _logger.info("Building metadata for %s/%s", self._dataset_name, selection_repr)
        metadata = _do_compute_metadata(dataset, auto_bin_method, exclude, continuous_factor_bins)
        self.save_metadata(selection_repr, metadata)
        return metadata

    # =====================================================================
    # Unified stats cache (.parquet + .json sidecar)
    # =====================================================================

    def _stats_paths(self, selection_repr: str, scope: str) -> tuple[Path, Path] | tuple[None, None]:
        sel_dir = self._selection_dir(selection_repr)
        if sel_dir is None:
            return None, None
        h = _config_hash(scope)
        return sel_dir / f"stats_{h}.parquet", sel_dir / f"stats_{h}.json"

    def load_stats(
        self,
        selection_repr: str,
        scope: str,
    ) -> dict[str, Any] | None:
        """Load cached ``StatsResult``, or ``None`` on miss.

        Returns the full dict with whatever metrics are currently cached.
        Inspect ``result["stats"].keys()`` to see which metrics are present.
        """
        obj_key = f"stats_{_config_hash(scope)}"
        cached = self._mem_get(selection_repr, obj_key)
        if cached is not None:
            _logger.debug("Memory hit: stats for %s/%s (scope=%s)", self._dataset_name, selection_repr, scope)
            return cached

        from dataeval.types import SourceIndex

        pq_path, json_path = self._stats_paths(selection_repr, scope)
        if pq_path is None or json_path is None or not pq_path.exists() or not json_path.exists():
            return None

        _logger.info("Cache hit: stats for %s/%s (scope=%s)", self._dataset_name, selection_repr, scope)

        try:
            df = pl.read_parquet(pq_path)
            with open(json_path, encoding="utf-8") as f:
                aux = json.load(f)

            stats: dict[str, NDArray[Any]] = {}
            for col in df.columns:
                series = df[col]
                if series.dtype == pl.Utf8:
                    # Hash string columns
                    stats[col] = series.to_numpy(writable=False).astype(object)
                elif series.dtype.base_type() == pl.List:
                    # 2D array columns (histogram, percentiles, center)
                    stats[col] = np.array(series.to_list())
                else:
                    stats[col] = series.to_numpy(writable=False)

            source_index = [SourceIndex(item=s[0], target=s[1], channel=s[2]) for s in aux["source_index"]]

            result = {
                "source_index": source_index,
                "object_count": aux["object_count"],
                "invalid_box_count": aux["invalid_box_count"],
                "image_count": aux["image_count"],
                "stats": stats,
            }
            self._mem_set(selection_repr, obj_key, result)
            return result
        except Exception:  # noqa: BLE001
            _logger.warning(
                "Failed to load stats from cache for %s/%s (scope=%s) — recomputing",
                self._dataset_name,
                selection_repr,
                scope,
                exc_info=True,
            )
            return None

    def save_stats(
        self,
        selection_repr: str,
        scope: str,
        stats: dict[str, Any],
    ) -> None:
        """Persist ``StatsResult`` to cache (no-op on disk when not disk-backed).

        Handles scalar arrays, object-dtype hash strings, and 2D arrays
        (histogram, percentiles, center) via Polars list columns.
        """
        obj_key = f"stats_{_config_hash(scope)}"
        self._mem_set(selection_repr, obj_key, stats)

        pq_path, json_path = self._stats_paths(selection_repr, scope)
        if pq_path is None or json_path is None:
            return

        stat_arrays: dict[str, NDArray[Any]] = dict(stats["stats"])
        series_dict: dict[str, pl.Series] = {}
        for name, arr in stat_arrays.items():
            if arr.dtype == object:
                # Hash string arrays → Utf8 columns
                series_dict[name] = pl.Series(name, [str(v) for v in arr])
            elif arr.ndim == 2:
                # 2D arrays (histogram, percentiles, center) → list columns
                series_dict[name] = pl.Series(name, arr.tolist())
            else:
                series_dict[name] = pl.Series(name, arr)
        df = pl.DataFrame(series_dict)

        aux = {
            "source_index": [[int(si.item), si.target, si.channel] for si in stats["source_index"]],
            "object_count": [int(v) for v in stats["object_count"]],
            "invalid_box_count": [int(v) for v in stats["invalid_box_count"]],
            "image_count": int(stats["image_count"]),
        }
        _atomic_write_pair(
            pq_path,
            lambda p: df.write_parquet(p),
            json_path,
            lambda p: p.write_text(json.dumps(aux), encoding="utf-8"),
        )

        _logger.info(
            "Cache save: stats for %s/%s (scope=%s, metrics=%s)",
            self._dataset_name,
            selection_repr,
            scope,
            sorted(stat_arrays.keys()),
        )

    def load_or_compute_stats(
        self,
        selection_repr: str,
        scope: str,
        desired_flags: ImageStats,
        dataset: AnnotatedDataset[Any],
        per_image: bool = True,
        per_target: bool = True,
        per_channel: bool = False,
    ) -> dict[str, Any]:
        """Load cached stats, compute any missing metrics, merge, and save.

        Parameters
        ----------
        selection_repr : str
            Selection key (from :func:`selection_repr`).
        scope : str
            Scope key (from :func:`scope_key`).
        desired_flags : ImageStats
            All flags the caller needs.
        dataset
            The dataset to pass to ``compute_stats()`` on miss.
        per_image, per_target, per_channel
            Scope settings for ``compute_stats()``.

        Returns
        -------
        dict[str, Any]
            A ``StatsResult``-shaped dict with at least all the
            requested metrics.
        """
        cached = self.load_stats(selection_repr, scope)
        if cached is not None:
            cached_metric_names = set(cached["stats"].keys())
            to_compute = missing_flags(cached_metric_names, desired_flags)
        else:
            to_compute = desired_flags

        if to_compute == ImageStats.NONE and cached is not None:
            _logger.info("Full cache hit for stats (scope=%s)", scope)
            return cached

        if cached is not None:
            _logger.info(
                "Partial cache hit: computing missing metrics (have %s)",
                sorted(cached["stats"].keys()),
            )

        # Compute the missing stats
        fresh = _do_compute_stats(dataset, to_compute, per_image, per_target, per_channel)

        if cached is None:
            fresh_dict = dict(fresh)
            self.save_stats(selection_repr, scope, fresh_dict)
            return fresh_dict

        # Merge: cached structural fields + merged stats dict
        merged_stats = dict(cached["stats"])
        merged_stats.update(fresh["stats"])

        merged: dict[str, Any] = {
            "source_index": cached["source_index"],
            "object_count": cached["object_count"],
            "invalid_box_count": cached["invalid_box_count"],
            "image_count": cached["image_count"],
            "stats": merged_stats,
        }

        self.save_stats(selection_repr, scope, merged)
        return merged
