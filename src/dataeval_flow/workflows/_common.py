"""Helpers shared across workflow implementations.

Private to :mod:`dataeval_flow.workflows`. Each consuming module imports these under
its own ``_``-prefixed local name, so the historical per-workflow spellings still
resolve while there is a single definition to maintain.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from dataeval import Metadata

__all__ = [
    "compute_metadata_summary",
    "normalize_unit_interval",
    "serialize_coverage",
    "to_serializable",
]

#: Maximum number of most-frequent values reported per discrete factor.
_TOP_VALUES = 10


def to_serializable(obj: Any) -> Any:
    """Convert non-JSON-serializable types to plain Python types recursively."""
    if isinstance(obj, dict):
        return {to_serializable(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_serializable(obj.tolist())
    if isinstance(obj, frozenset | set):
        return sorted(str(v) for v in obj)
    return obj


def serialize_coverage(coverage_result: Any) -> dict[str, Any]:
    """Convert a dataeval ``CoverageResult`` to a plain dict."""
    result: dict[str, Any] = {}
    for key in ("uncovered_indices", "critical_value_radii", "coverage_radius"):
        val = coverage_result.get(key, None) if hasattr(coverage_result, "get") else getattr(coverage_result, key, None)
        if val is not None:
            if hasattr(val, "tolist"):
                val = val.tolist()
            result[key] = val
    return result


def compute_metadata_summary(metadata: "Metadata") -> dict[str, dict[str, Any]]:
    """Compute per-factor summary statistics from metadata."""
    summary: dict[str, dict[str, Any]] = {}
    df = metadata.image_data
    factor_info = metadata.factor_info

    for name, info in factor_info.items():
        stats: dict[str, Any] = {"type": info.factor_type}

        if name not in df.columns:
            summary[name] = stats
            continue

        col = df[name]
        stats["null_count"] = col.null_count()

        if info.factor_type == "continuous":
            stats["min"] = col.min()
            stats["max"] = col.max()
            stats["mean"] = col.mean()
            stats["std"] = col.std()
        else:
            stats["unique_values"] = col.n_unique()
            vc = col.value_counts().sort("count", descending=True)
            if len(vc) > 0:
                top = min(_TOP_VALUES, len(vc))
                values = vc[name].head(top).to_list()
                counts = vc["count"].head(top).to_list()
                stats["top_values"] = dict(zip(values, counts, strict=True))

        summary[name] = stats

    return to_serializable(summary)


def normalize_unit_interval(embeddings: np.ndarray) -> np.ndarray:
    """Rescale each embedding dimension to [0, 1], as the coverage functions require.

    Constant dimensions are left at zero rather than dividing by a zero range.
    """
    emb_min = embeddings.min(axis=0, keepdims=True)
    emb_max = embeddings.max(axis=0, keepdims=True)
    emb_range = emb_max - emb_min
    emb_range[emb_range == 0] = 1.0
    return (embeddings - emb_min) / emb_range
