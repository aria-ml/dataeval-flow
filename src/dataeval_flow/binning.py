"""Record of how metadata factors were typed and binned.

DataEval decides each factor's type on the caller's behalf and, for a continuous
one, where its bin edges fall.  Those decisions change what every downstream
evaluator sees — balance, diversity and parity read binned codes rather than the
values measured — but DataEval reports them only as log records, which do not
survive into a result envelope.  A result archived today therefore cannot answer
"was this factor binned, and over what range?", which is exactly what a reviewer
comparing two runs needs to know.

This module reconstructs those decisions from the companion columns DataEval
writes alongside each factor, so the envelope carries them.  Observed per-bin
ranges are recorded rather than the nominal edges: they describe what the run
actually did, and they remain meaningful for an empty or clipped bin.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import polars as pl

from dataeval_flow.workflows._common import to_serializable

if TYPE_CHECKING:
    from dataeval import Metadata

    from dataeval_flow.config.schemas._metadata import ResultMetadata
    from dataeval_flow.workflow.base import MetadataConfigMixin

__all__ = ["attach_binning", "describe_binning"]

_logger: logging.Logger = logging.getLogger(__name__)

# DataEval writes one companion column per factor: bin indices for a binned
# continuous factor, category ordinals for a digitized one.  The suffixes come
# from dataeval's private ``_metadata._columns``; they are mirrored here rather
# than imported so that a private rename upstream costs us the per-bin detail
# rather than raising ImportError on a working install.
_BINNED_SUFFIX = "↕"
_DIGITIZED_SUFFIX = "#"


def _default_factor_source() -> str | None:
    """What ``factor_source`` a workflow leaving it unset actually gets.

    Read off a constructed evaluator rather than imported: the constant lives in a
    private module, and this record is worth less than the run it would take down on a
    release that spells it differently.  Constructing a ``Balance`` runs no statistics.
    """
    try:
        from dataeval.bias import Balance

        return str(Balance().factor_source)
    except Exception:  # noqa: BLE001 - a release without the setting records nothing for it
        return None


def _bin_ranges(df: pl.DataFrame, name: str, companion: str) -> list[dict[str, Any]] | None:
    """Observed value range and population of each bin, in bin order."""
    if name not in df.columns or companion not in df.columns:
        return None

    # Both inputs are renamed to fixed internal names first: a factor is free to
    # be called "count", "min" or "max", and aggregating into those aliases
    # alongside it raises a duplicate-column error.
    grouped = (
        df.select([pl.col(companion).alias("_code"), pl.col(name).alias("_value")])
        .drop_nulls()
        .group_by("_code")
        .agg(
            pl.len().alias("_n"),
            pl.col("_value").min().alias("_min"),
            pl.col("_value").max().alias("_max"),
        )
        .sort("_code")
    )
    return [
        {
            "code": int(row["_code"]),
            "count": int(row["_n"]),
            "min": row["_min"],
            "max": row["_max"],
        }
        for row in grouped.to_dicts()
    ]


def _categories(df: pl.DataFrame, name: str, companion: str) -> list[dict[str, Any]] | None:
    """Ordinal-to-value mapping and population of each category, in ordinal order."""
    if name not in df.columns or companion not in df.columns:
        return None

    # Renamed first for the same reason as _bin_ranges: a factor may be called
    # "count" or "value".
    grouped = (
        df.select([pl.col(companion).alias("_code"), pl.col(name).alias("_value")])
        .drop_nulls()
        .group_by(["_code", "_value"])
        .agg(pl.len().alias("_n"))
        .sort("_code")
    )
    return [
        {
            "code": int(row["_code"]),
            "value": row["_value"],
            "count": int(row["_n"]),
        }
        for row in grouped.to_dicts()
    ]


def _factor_entry(
    name: str,
    info: Any,
    df: pl.DataFrame,
    requested_bins: Mapping[str, int | Sequence[float]],
    auto_bin_method: str | None,
) -> dict[str, Any]:
    """Build one factor's record: its type, level, and what discretizing did."""
    entry: dict[str, Any] = {
        "type": info.factor_type,
        "level": info.level,
        "is_binned": info.is_binned,
        "is_digitized": info.is_digitized,
    }
    if getattr(info, "aggregated_from", None) is not None:
        entry["aggregated_from"] = info.aggregated_from

    # A factor the caller asked to bin explicitly is distinguished from one
    # binned by the automatic method, since only the latter can move when the
    # data changes.
    if name in requested_bins:
        entry["bins_requested"] = requested_bins[name]
    elif info.is_binned:
        entry["binned_by"] = auto_bin_method

    if info.is_binned:
        bins = _bin_ranges(df, name, f"{name}{_BINNED_SUFFIX}")
        if bins is not None:
            entry["bins"] = bins
            entry["bin_count"] = len(bins)
    elif info.is_digitized:
        categories = _categories(df, name, f"{name}{_DIGITIZED_SUFFIX}")
        if categories is not None:
            entry["categories"] = categories
            entry["category_count"] = len(categories)

    return entry


def describe_binning(
    metadata: "Metadata",
    *,
    excluded: Sequence[str] | None = None,
    requested_bins: Mapping[str, int | Sequence[float]] | None = None,
    factor_source: str | None = None,
) -> dict[str, Any]:
    """Describe how every factor was typed and discretized.

    Parameters
    ----------
    metadata : Metadata
        A bound DataEval ``Metadata``.  Reading ``factor_info`` forces binning,
        so the companion columns this reads are guaranteed to exist by the time
        the frames are fetched.
    excluded : Sequence[str] | None
        Factor names the configuration excluded.  Recorded because an excluded
        factor leaves no other trace — it is simply absent from the results.
    requested_bins : Mapping[str, int | Sequence[float]] | None
        The configured ``continuous_factor_bins``.  Defaults to what the
        ``Metadata`` was constructed with; pass explicitly to record a request
        that named a factor the dataset does not carry.
    factor_source : str | None
        The configured ``factor_source``, or None to record DataEval's default.
        Recorded because it decides whether each bias statistic read a factor's
        codes or its measured values, which moves every number those statistics
        report — and it leaves no other trace in the result.

    Returns
    -------
    dict
        JSON-serializable record with ``auto_bin_method``, ``factor_source``,
        ``requested_bins``, ``excluded``, per-factor ``factors``, and ``dropped``.
    """
    # Read factor_info first: it forces _bin(), which is what writes the
    # companion columns the frames below are read for.
    factor_info = metadata.factor_info

    if requested_bins is None:
        requested_bins = metadata.continuous_factor_bins or {}

    record: dict[str, Any] = {
        "auto_bin_method": getattr(metadata, "auto_bin_method", None),
        "factor_source": factor_source or _default_factor_source(),
        "requested_bins": dict(requested_bins),
        "excluded": list(excluded or ()),
        "factors": {},
        "dropped": {name: list(reasons) for name, reasons in metadata.dropped_factors.items()},
    }

    # One frame per level, fetched once — rows_at() materializes a frame per call.
    rows_by_level: dict[str, pl.DataFrame] = {}

    for name, info in factor_info.items():
        if info.level not in rows_by_level:
            rows_by_level[info.level] = metadata.rows_at(info.level)
        record["factors"][name] = _factor_entry(
            name, info, rows_by_level[info.level], requested_bins, record["auto_bin_method"]
        )

    # A request naming a factor the dataset does not carry is silently ignored
    # by DataEval (it warns and moves on), so it is called out here rather than
    # leaving the reader to diff two lists.
    unmatched = sorted(set(requested_bins) - set(factor_info))
    if unmatched:
        record["unmatched_bin_requests"] = unmatched

    return to_serializable(record)


def attach_binning(
    result_metadata: "ResultMetadata",
    metadata: "Metadata | Mapping[str, Metadata]",
    params: "MetadataConfigMixin",
) -> None:
    """Record binning decisions on a workflow's metadata envelope.

    Accepts either a single ``Metadata`` or a mapping of split name to one, so a
    multi-split workflow records each split separately — splits are binned
    independently, and two splits of the same dataset can land on different
    edges.

    Never raises.  A companion column renamed upstream costs the record, not the
    run, and the diagnostics captured alongside it still name the decision.
    """
    try:
        excluded = list(params.metadata_exclude) if params.metadata_exclude else None
        requested = params.metadata_continuous_factor_bins
        source = params.metadata_factor_source

        if isinstance(metadata, Mapping):
            result_metadata.metadata_binning = {
                "per_split": {
                    name: describe_binning(md, excluded=excluded, requested_bins=requested, factor_source=source)
                    for name, md in metadata.items()
                }
            }
        else:
            result_metadata.metadata_binning = describe_binning(
                metadata, excluded=excluded, requested_bins=requested, factor_source=source
            )
    except Exception:
        _logger.warning("Binning record unavailable", exc_info=True)
