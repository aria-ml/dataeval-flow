"""Disk-backed cache for expensive workflow computations.

Provides load/save methods for three component types:
- **Embeddings** — Dense numpy arrays stored as ``.npy``
- **Metadata** — Polars DataFrame as ``.parquet`` + auxiliary attributes as ``.json``
- **Stats** — Unified ``CalculationResult`` stored as ``.parquet`` + ``.json``.
Metrics accumulate incrementally: different workflows requesting different ``ImageStats``
flags share the same cache entry and only compute the missing metrics.

Cache layout::

    cache_dir/
      v{CACHE_VERSION}/
        {dataset_name}/
          embeddings_{selection_hash}_{config_hash}.npy
          metadata_{selection_hash}.parquet
          metadata_{selection_hash}.json
          stats_{selection_hash}_{scope_hash}.parquet
          stats_{selection_hash}_{scope_hash}.json

Cache artefacts are stored under a ``v{CACHE_VERSION}`` subdirectory so
that different versions can coexist side-by-side.  When the cache format
changes in a backwards-incompatible way, bump ``CACHE_VERSION`` and old
data is simply ignored.  Users can clean up stale versions by removing the
old ``v*/`` directories (e.g. ``rm -rf /cache/v1``).
"""

__all__ = [
    "CACHE_VERSION",
    "FLAG_TO_METRIC",
    "METRIC_TO_FLAG",
    "WorkflowCache",
    "get_or_compute_embeddings",
    "get_or_compute_metadata",
    "get_or_compute_stats",
    "missing_flags",
    "scope_key",
    "selection_repr",
]

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from dataeval.flags import ImageStats, resolve_dependencies
from dataeval.protocols import AnnotatedDataset
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dataeval import Metadata

_logger = logging.getLogger(__name__)

# Bump this when the on-disk cache format changes in a backwards-incompatible
# way.  Cached artefacts are stored under ``v{CACHE_VERSION}/`` so different
# versions coexist and users can ``rm -rf`` old directories to reclaim space.
CACHE_VERSION = "0"


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
    resolved = resolve_dependencies(desired_flags)
    uncovered = ImageStats.NONE
    for flag in ImageStats:
        # Only consider individual (atomic, single-bit) flags
        if flag.value and (flag.value & (flag.value - 1)) == 0 and flag in resolved:
            metric_name = FLAG_TO_METRIC.get(flag)
            if metric_name and metric_name not in cached_metrics:
                uncovered |= flag
    # Re-resolve so that dependencies of missing flags are included
    return ImageStats(resolve_dependencies(uncovered)) if uncovered else ImageStats.NONE


def get_or_compute_stats(
    desired_flags: ImageStats,
    dataset: AnnotatedDataset[Any],
    per_image: bool = True,
    per_target: bool = True,
    per_channel: bool = False,
    *,
    cache: "WorkflowCache | None" = None,
    selection_key: str | None = None,
) -> dict[str, Any]:
    """Centralized stats computation with optional disk caching.

    When *cache* and *selection_key* are provided, consults the disk cache
    first: only metrics not yet cached are computed, and the merged result
    is saved back.  Without a cache, stats are computed directly via
    ``calculate_stats()``.

    This is the single entry-point that workflows should use to obtain
    a ``CalculationResult``-shaped dict.

    Parameters
    ----------
    desired_flags : ImageStats
        All the metric flags the caller needs.
    dataset
        The dataset to compute stats from (must conform to DataEval protocol).
    per_image, per_target, per_channel
        Scope settings passed to ``calculate_stats()``.
    cache : WorkflowCache | None
        Optional disk cache.  When ``None``, stats are always computed fresh.
    selection_key : str | None
        Selection key (from :func:`selection_repr`).  Required when *cache*
        is provided.

    Returns
    -------
    dict[str, Any]
        A ``CalculationResult``-shaped dict with at least all the
        requested metrics.
    """
    if cache is not None and selection_key is not None:
        scope = scope_key(per_image, per_target, per_channel)
        return cache.load_or_compute_stats(
            selection_key,
            scope,
            desired_flags,
            dataset,
            per_image=per_image,
            per_target=per_target,
            per_channel=per_channel,
        )

    from dataeval.core._calculate_stats import calculate_stats

    return dict(
        calculate_stats(
            dataset,
            None,
            desired_flags,
            per_image=per_image,
            per_target=per_target,
            per_channel=per_channel,
        )
    )


def get_or_compute_metadata(
    dataset: AnnotatedDataset[Any],
    auto_bin_method: Any = None,
    exclude: list[str] | None = None,
    continuous_factor_bins: dict[str, int | list[float]] | None = None,
    *,
    cache: "WorkflowCache | None" = None,
    selection_key: str | None = None,
) -> "Metadata":
    """Build metadata with optional disk caching.

    When *cache* and *selection_key* are provided, checks the disk cache
    first.  The cache stores only raw (pre-binned) metadata keyed by
    dataset selection — binning configuration is applied lazily on load.
    Without a cache, metadata is built directly.

    Parameters
    ----------
    dataset
        The dataset to build metadata from.
    auto_bin_method
        Method for automatic binning of continuous values.
    exclude : list[str] | None
        Metadata columns to exclude.
    continuous_factor_bins : dict[str, int | list[float]] | None
        Number of uniform bins (int) or explicit bin edges (list[float])
        for specific continuous factors.
    cache : WorkflowCache | None
        Optional disk cache.
    selection_key : str | None
        Selection key (from :func:`selection_repr`).  Required when
        *cache* is provided.

    Returns
    -------
    Metadata
        DataEval Metadata instance.
    """
    if cache is not None and selection_key is not None:
        return cache.load_or_compute_metadata(
            selection_key,
            dataset,
            auto_bin_method=auto_bin_method,
            exclude=exclude,
            continuous_factor_bins=continuous_factor_bins,
        )

    from dataeval_app.metadata import build_metadata

    return build_metadata(
        dataset,
        auto_bin_method=auto_bin_method,
        exclude=exclude,
        continuous_factor_bins=continuous_factor_bins,
    )


def get_or_compute_embeddings(
    dataset: AnnotatedDataset[Any],
    extractor_config: Any,
    transforms: Any = None,
    batch_size: int | None = None,
    *,
    cache: "WorkflowCache | None" = None,
    selection_key: str | None = None,
) -> NDArray[Any]:
    """Extract embeddings with optional disk caching.

    When *cache* and *selection_key* are provided, checks the disk cache
    first.  On miss, builds an ``Embeddings`` instance, computes the
    array, saves it to cache, and returns it.  Without a cache, embeddings
    are computed directly.

    The cache key is derived from the dataset *selection_key*, the
    extractor config (JSON), and the transforms representation.

    Parameters
    ----------
    dataset
        The dataset to extract embeddings from.
    extractor_config
        ``ExtractorConfig`` (Pydantic model with ``.model_dump_json()``).
    transforms
        Optional preprocessing transforms.
    batch_size : int | None
        Batch size for extraction.
    cache : WorkflowCache | None
        Optional disk cache.
    selection_key : str | None
        Selection key (from :func:`selection_repr`).  Required when
        *cache* is provided.

    Returns
    -------
    NDArray
        2-D embedding array of shape ``(N, D)``.
    """
    config_json = extractor_config.model_dump_json(exclude_defaults=False)
    transforms_key = repr(transforms) if transforms is not None else "none"

    if cache is not None and selection_key is not None:
        return cache.load_or_compute_embeddings(
            selection_key,
            config_json,
            transforms_key,
            dataset,
            extractor_config,
            transforms,
            batch_size,
        )

    from dataeval_app.embeddings import build_embeddings

    embeddings = build_embeddings(dataset, extractor_config, transforms, batch_size)
    array: NDArray[Any] = np.asarray(embeddings)
    if array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    return array


# ---------------------------------------------------------------------------
# WorkflowCache
# ---------------------------------------------------------------------------


class WorkflowCache:
    """Disk-backed cache for workflow computations.

    Organises cached artefacts under::

        cache_dir / v{CACHE_VERSION} / dataset_name / {component}_{sel}_{cfg}.{ext}

    Parameters
    ----------
    cache_dir : Path
        Root directory for cache storage.
    dataset_name : str
        Dataset identifier (from ``TaskConfig.dataset``).
    """

    def __init__(self, cache_dir: Path, dataset_name: str) -> None:
        """Initialize the cache with the root directory and dataset name."""
        if "/" in dataset_name or "\\" in dataset_name or dataset_name in (".", ".."):
            raise ValueError(
                f"Invalid dataset_name for cache (must not contain path separators or be '.'/'..'): {dataset_name!r}"
            )
        self._cache_dir = cache_dir
        self._dataset_name = dataset_name
        self._dataset_dir = None

    # =====================================================================
    # Properties
    # =====================================================================

    @property
    def cache_dir(self) -> Path:
        """Root cache directory."""
        return self._cache_dir

    @property
    def dataset_name(self) -> str:
        """Dataset identifier."""
        return self._dataset_name

    @property
    def dataset_dir(self) -> Path:
        """Dataset-specific cache directory (version-namespaced)."""
        if self._dataset_dir is None:
            d = self._cache_dir / f"v{CACHE_VERSION}" / self._dataset_name
            d.mkdir(parents=True, exist_ok=True)
            self._dataset_dir = d
        return self._dataset_dir

    # =====================================================================
    # Embeddings (.npy)
    # =====================================================================

    def _embeddings_path(
        self,
        selection_repr: str,
        extractor_config_json: str,
        transforms_repr: str,
    ) -> Path:
        s = _config_hash(selection_repr)
        h = _config_hash(extractor_config_json + "|" + transforms_repr)
        return self.dataset_dir / f"embeddings_{s}_{h}.npy"

    def load_embeddings(
        self,
        selection_repr: str,
        extractor_config_json: str,
        transforms_repr: str = "none",
    ) -> NDArray[Any] | None:
        """Load cached embedding array, or ``None`` on miss."""
        path = self._embeddings_path(selection_repr, extractor_config_json, transforms_repr)
        if not path.exists():
            return None
        _logger.info("Cache hit: embeddings for %s/%s", self._dataset_name, selection_repr)
        try:
            return np.load(path, allow_pickle=False)
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
        """Persist embedding array to cache."""
        path = self._embeddings_path(selection_repr, extractor_config_json, transforms_repr)
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

        from dataeval_app.embeddings import build_embeddings

        _logger.info("Computing embeddings for %s/%s", self._dataset_name, selection_repr)
        embeddings = build_embeddings(dataset, extractor_config, transforms, batch_size)
        array: NDArray[Any] = np.asarray(embeddings)
        if array.ndim > 2:
            array = array.reshape(array.shape[0], -1)

        self.save_embeddings(selection_repr, extractor_config_json, transforms_repr, array)
        return array

    # =====================================================================
    # Metadata (.parquet + .json sidecar)
    # =====================================================================

    def _metadata_paths(self, selection_repr: str) -> tuple[Path, Path]:
        s = _config_hash(selection_repr)
        return self.dataset_dir / f"metadata_{s}.parquet", self.dataset_dir / f"metadata_{s}.json"

    def load_metadata(
        self,
        selection_repr: str,
        dataset: AnnotatedDataset[Any],
        auto_bin_method: Any = None,
        exclude: list[str] | None = None,
        continuous_factor_bins: Mapping[str, int | Sequence[float]] | None = None,
    ) -> "Metadata | None":
        """Load cached raw Metadata, or ``None`` on miss.

        Reconstructs a ``dataeval.Metadata`` instance from the cached
        parquet + JSON files with ``_is_binned = False``.  The caller's
        binning configuration is applied so that ``_bin()`` runs lazily
        with the current settings when factor data is first accessed.
        """
        from dataeval import Metadata as MetadataClass

        pq_path, json_path = self._metadata_paths(selection_repr)
        if not pq_path.exists() or not json_path.exists():
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
        """Persist raw (pre-binned) Metadata to cache as parquet + JSON sidecar.

        Only the structured DataFrame is saved — binned/digitized columns
        (``↕`` / ``#`` suffixes) are stripped so that a single cache entry
        can be reused across different binning configurations.
        """
        pq_path, json_path = self._metadata_paths(selection_repr)

        # .dataframe triggers _structure() but NOT _bin(), giving us
        # the raw structured data.  Drop any binned/digitized columns
        # that may exist if _bin() was already called on this object.
        df = metadata.dataframe
        drop_cols = [c for c in df.columns if c.endswith("↕") or c.endswith("#")]
        if drop_cols:
            df = df.drop(drop_cols)
        _atomic_write(pq_path, lambda p: df.write_parquet(p))

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
        _atomic_write(json_path, lambda p: p.write_text(json.dumps(aux, sort_keys=True), encoding="utf-8"))

        _logger.info("Cache save: metadata for %s/%s (%s)", self._dataset_name, selection_repr, pq_path.name)

    def load_or_compute_metadata(
        self,
        selection_repr: str,
        dataset: AnnotatedDataset[Any],
        auto_bin_method: Any = None,
        exclude: list[str] | None = None,
        continuous_factor_bins: dict[str, int | list[float]] | None = None,
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

        from dataeval_app.metadata import build_metadata

        _logger.info("Building metadata for %s/%s", self._dataset_name, selection_repr)
        metadata = build_metadata(
            dataset,
            auto_bin_method=auto_bin_method,
            exclude=exclude,
            continuous_factor_bins=continuous_factor_bins,
        )

        self.save_metadata(selection_repr, metadata)
        return metadata

    # =====================================================================
    # Unified stats cache (.parquet + .json sidecar)
    # =====================================================================

    def _stats_paths(self, selection_repr: str, scope: str) -> tuple[Path, Path]:
        s = _config_hash(selection_repr)
        h = _config_hash(scope)
        return self.dataset_dir / f"stats_{s}_{h}.parquet", self.dataset_dir / f"stats_{s}_{h}.json"

    def load_stats(
        self,
        selection_repr: str,
        scope: str,
    ) -> dict[str, Any] | None:
        """Load cached ``CalculationResult``, or ``None`` on miss.

        Returns the full dict with whatever metrics are currently cached.
        Inspect ``result["stats"].keys()`` to see which metrics are present.
        """
        from dataeval.types import SourceIndex

        pq_path, json_path = self._stats_paths(selection_repr, scope)
        if not pq_path.exists() or not json_path.exists():
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

            return {
                "source_index": source_index,
                "object_count": aux["object_count"],
                "invalid_box_count": aux["invalid_box_count"],
                "image_count": aux["image_count"],
                "stats": stats,
            }
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
        """Persist ``CalculationResult`` to cache as parquet + JSON sidecar.

        Handles scalar arrays, object-dtype hash strings, and 2D arrays
        (histogram, percentiles, center) via Polars list columns.
        """
        pq_path, json_path = self._stats_paths(selection_repr, scope)

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
        _atomic_write(pq_path, lambda p: df.write_parquet(p))

        aux = {
            "source_index": [[int(si.item), si.target, si.channel] for si in stats["source_index"]],
            "object_count": [int(v) for v in stats["object_count"]],
            "invalid_box_count": [int(v) for v in stats["invalid_box_count"]],
            "image_count": int(stats["image_count"]),
        }
        _atomic_write(json_path, lambda p: p.write_text(json.dumps(aux), encoding="utf-8"))

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
            The dataset to pass to ``calculate_stats()`` on miss.
        per_image, per_target, per_channel
            Scope settings for ``calculate_stats()``.

        Returns
        -------
        dict[str, Any]
            A ``CalculationResult``-shaped dict with at least all the
            requested metrics.
        """
        from dataeval.core._calculate_stats import calculate_stats

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
        fresh = calculate_stats(
            dataset,
            None,
            to_compute,
            per_image=per_image,
            per_target=per_target,
            per_channel=per_channel,
        )

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
