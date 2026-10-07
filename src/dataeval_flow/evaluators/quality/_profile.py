"""The ``profile`` evaluator: how each statistic and metadata field of a Dataset is distributed, scope by scope.

A scope is what a row describes: ``image``, one row per item, and on detection data ``target``, one row per box.
Statistics come from DataEval's ``compute_stats`` under the stats policy, metadata fields from DataEval's ``Metadata``
at each field's own level. The summaries go in the result; every row's value goes beside it, so that a selection over a
bin or a category reads the rows themselves rather than re-measuring anything.
"""

__all__ = ["ProfileEvaluator", "summarize"]

import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import polars as pl
from dataeval import Metadata

from dataeval_flow._input_spec import InputKind
from dataeval_flow._metadata import IMAGE_STAT_GROUPS
from dataeval_flow._stats import columns_for
from dataeval_flow.evaluators._core import execution
from dataeval_flow.evaluators._evaluator import Evaluator
from dataeval_flow.evaluators._fields import require
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators.quality._config import ProfileConfig
from dataeval_flow.evaluators.quality._result import ProfileOutput

if TYPE_CHECKING:
    from dataeval.core import StatsResult
    from dataeval.types import FactorLevel

    from dataeval_flow._stats import ResolvedStatsPolicy

SCHEMA = 1
_FAMILIES = ("dimension", "pixel", "visual")
_KEYS = ("item", "target")


def summarize(series: pl.Series, *, bins: int, categories: int) -> dict[str, Any]:
    """One field's summary: its counts, then for a number its range, mean, median, population standard deviation and
    an equal-width histogram (``numpy.histogram``'s bins: each closed on the left, the last closed on both sides), and
    for anything else its `categories` most frequent values, by their JSON, with the rest counted as other.

    A null is missing; a NaN or an infinity is non-finite and sits in no bin. A constant number has one bin,
    ``[v, v]``, and a number with no finite value has no summary at all.
    """
    rows, missing = len(series), series.null_count()
    if series.dtype.is_numeric():
        values = series.drop_nulls().cast(pl.Float64).to_numpy()
        finite = values[np.isfinite(values)]
        counts: dict[str, Any] = {"rows": rows, "missing": missing, "non_finite": len(values) - len(finite)}
        field: dict[str, Any] = {"type": "numeric", **counts, "finite": len(finite)}
        if not finite.size:
            return field | dict.fromkeys(("min", "max", "mean", "median", "std", "histogram"))
        low, high = float(finite.min()), float(finite.max())
        if low == high:
            histogram = {"edges": [low, high], "counts": [int(finite.size)]}
        else:
            hist, edges = np.histogram(finite, bins=bins, range=(low, high))
            histogram = {"edges": edges.tolist(), "counts": hist.tolist()}
        summary = {"mean": float(finite.mean()), "median": float(np.median(finite)), "std": float(finite.std())}
        return field | {"min": low, "max": high, **summary, "histogram": histogram}
    present = [as_json(value) for value in series.drop_nulls().to_list()]
    ranked = sorted(Counter(present).items(), key=lambda pair: (-pair[1], pair[0]))
    named, rest = ranked[:categories], ranked[categories:]
    return {
        "type": "categorical",
        "rows": rows,
        "missing": missing,
        "non_finite": 0,
        "finite": len(present),
        "values": [{"value": json.loads(text), "count": count} for text, count in named],
        "other": {"values": len(rest), "count": sum(count for _, count in rest)} if rest else None,
    }


def as_json(value: Any) -> str:
    """A categorical value as the JSON a profile's rows hold it in, so ``1``, ``"1"`` and ``true`` stay apart."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _origins(policy: "ResolvedStatsPolicy") -> dict[str, tuple[str | None, str | None]]:
    """Each column the policy measures, with the view it was measured over (``None`` for the whole image, else a band
    group or a background) and its family."""
    views = [view for view, _ in policy.measure]
    if policy.background:
        views += ["background" if view is None else f"background_{view}" for view, _ in policy.measure]
    origins: dict[str, tuple[str | None, str | None]] = {}
    for view in views:
        for family in _FAMILIES:
            for column in columns_for([view], policy.families_of(view) & IMAGE_STAT_GROUPS[family]):
                origins.setdefault(column, (view, family))
    return origins


def _computed(
    stats: "StatsResult", policy: "ResolvedStatsPolicy"
) -> tuple[dict[str, pl.DataFrame], list[dict[str, Any]], list[tuple[str, str, Any]]]:
    """Each scope's rows from the statistics, the descriptor of each column, and each column's values per scope."""
    index = list(stats["source_index"])
    scopes = {
        "image": [row for row, address in enumerate(index) if address.key is None],
        "target": [row for row, address in enumerate(index) if address.key is not None],
    }
    keys = {
        scope: pl.DataFrame(
            {"item": [index[row].item for row in rows], "target": [index[row].key for row in rows]},
            schema={"item": pl.Int64, "target": pl.Int64},
        )
        for scope, rows in scopes.items()
        if rows or scope == "image"
    }
    origins = _origins(policy)
    fields: list[dict[str, Any]] = []
    values: list[tuple[str, str, Any]] = []
    for name in sorted(set(stats["stats"]) & set(origins)):
        array = np.asarray(stats["stats"][name])
        group, family = origins[name]
        for scope in keys:
            descriptor = {
                "name": name,
                "scope": scope,
                "origin": "computed",
                "column": f"computed:{name}",
                "group": group,
                "family": family,
            }
            rows = scopes[scope]
            if array.dtype.kind in "OUSV" or array.ndim > 1:
                reason = "vector-valued" if array.ndim > 1 else "a hash, not a distribution"
                fields.append({**descriptor, "type": "unsupported", "reason": reason, "rows": len(rows)})
                continue
            values.append((scope, descriptor["column"], array[rows]))
            fields.append(descriptor)
    return keys, fields, values


def _supplied(metadata: Metadata, scopes: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, pl.DataFrame]]:
    """The descriptor of each metadata factor, at its own level, and each scope's factor columns by item and target.

    On detection data a box-level factor is a ``target`` field; everything else, and every factor of classification
    data, whose one label per item is its only box, is an ``image`` field."""
    by_level: dict[FactorLevel, list[str]] = {}
    for name, info in sorted(metadata.factor_info.items()):
        by_level.setdefault(info.level, []).append(name)
    fields: list[dict[str, Any]] = []
    joined: dict[str, pl.DataFrame] = {}
    for level, names in by_level.items():
        scope = "target" if level == "instance" and metadata.multi_target else "image"
        frame = metadata.rows_at(level)
        present = [name for name in names if name in frame.columns]
        if scope not in scopes or not present:
            continue
        target = pl.col("target_index") if scope == "target" else pl.lit(None, pl.Int64)
        columns = [pl.col(name).alias(f"supplied:{name}") for name in present]
        rows = frame.select(pl.col("item_index").alias("item"), target.alias("target"), *columns)
        rows = rows.unique(subset=_KEYS, keep="first", maintain_order=True)
        joined[scope] = rows if scope not in joined else joined[scope].join(rows, on=_KEYS, nulls_equal=True)
        fields += [_supplied_field(name, scope) for name in present]
    for name, reasons in sorted(metadata.dropped_factors.items()):
        reason = ", ".join(str(reason) for reason in reasons)
        fields.append({**_supplied_field(name, "image"), "type": "unsupported", "reason": reason})
    return fields, joined


def _supplied_field(name: str, scope: str) -> dict[str, Any]:
    """A metadata factor's descriptor: supplied with the data, measured over no band group and in no family, and kept
    in a column named for its origin, so a factor named like a statistic, or ``item``, is a field of its own."""
    return {
        "name": name,
        "scope": scope,
        "origin": "supplied",
        "column": f"supplied:{name}",
        "group": None,
        "family": None,
    }


def _stored(series: pl.Series) -> pl.Series:
    """A field's rows as a profile keeps them: a number as ``Float64``, anything else as its value's JSON text."""
    if series.dtype.is_numeric():
        return series.cast(pl.Float64)
    return pl.Series(series.name, [None if value is None else as_json(value) for value in series.to_list()], pl.Utf8)


def profile(
    stats: "StatsResult", policy: "ResolvedStatsPolicy", metadata: Metadata, *, bins: int, categories: int
) -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
    """The fields' summaries, by scope, and each scope's rows, as ``profile`` reports and keeps them."""
    keys, fields, values = _computed(stats, policy)
    supplied, joined = _supplied(metadata, list(keys))
    frames = {}
    for scope, frame in keys.items():
        computed = [pl.Series(column, rows) for each_scope, column, rows in values if each_scope == scope]
        frame = frame.with_columns(computed)
        if scope in joined:
            frame = frame.join(joined[scope], on=_KEYS, how="left", nulls_equal=True)
        frames[scope] = frame
    for field in [*fields, *supplied]:
        if field.get("type") == "unsupported":
            continue
        field.update(summarize(frames[field["scope"]][field["column"]], bins=bins, categories=categories))
    for scope, frame in frames.items():
        frames[scope] = frame.with_columns(_stored(frame[name]) for name in frame.columns if name not in _KEYS)
    data = {
        "schema": SCHEMA,
        "binning": {"method": "equal_width", "bins": bins, "intervals": "left-closed, last bin closed"},
        "std": "population",
        "categories": categories,
        "scopes": {scope: {"rows": frame.height} for scope, frame in frames.items()},
        "fields": [*fields, *supplied],
    }
    return data, frames


class ProfileEvaluator(Evaluator[ProfileConfig, ProfileOutput]):
    """``profile``: each statistic and metadata field of a source summarized, binned and kept row by row."""

    name: ClassVar[str] = "profile"
    title: ClassVar[str] = "Profile"
    description: ClassVar[str] = "How each statistic and metadata field is distributed: counts, summaries, histograms."
    dataeval_class: ClassVar[Any] = profile
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.STATS: "__call__", InputKind.METADATA: "__call__"}

    def run(self, config: ProfileConfig, inputs: Sequence[EvaluatorInputs]) -> ProfileOutput:
        """Profile the source's statistics under its stats policy and its metadata under its metadata policy."""
        from dataeval_flow._stats import restrict_columns

        (source,) = inputs
        policy = require(source.stats_policy, "a stats policy", source.source)
        stats = restrict_columns(require(source.stats, "stats", source.source), policy.columns())
        metadata = require(source.metadata, "metadata", source.source)
        started, clock = datetime.now(UTC), time.monotonic()
        data, frames = profile(stats, policy, metadata, bins=config.bins, categories=config.categories)
        meta = execution("dataeval_flow.profile", started, time.monotonic() - clock, {})
        return ProfileOutput({"source": source.source, **data}, meta, frames)
