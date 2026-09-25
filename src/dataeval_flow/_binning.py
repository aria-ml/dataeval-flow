"""Record of how metadata factors were encoded, and how well that encoding fits.

Balance, diversity and parity read codes, so a reviewer comparing two runs must
be able to see the map between codes and the measured values.

Two members per factor, answering two different questions:

``encoding``
    The **policy**: the edges or the vocabulary, who chose them (``provenance``),
    and how they were placed (``method``).  Read from DataEval's own record and
    rendered by DataEval's own writer, so this is byte-for-byte what a committed
    descriptor holds and what the cache sidecar names.
``fit``
    The **observation**: how many rows reached each code in this run, the span
    they occupied, and which declared bins nothing reached at all.

Earlier versions reconstructed the policy from the observed contents of each bin,
which described the sample rather than the decision: the same cut over a
different sample printed a different record, and a declared cutoff never
survived into its own label — ``{"temp_c": [-inf, 0.0, inf]}`` was reported as
``[-40, -0.3]``.  The record now comes from DataEval's own record; occupancy is
measured alongside it.
"""

import json
import logging
import math
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import polars as pl

from dataeval_flow.workflows._common import to_serializable

if TYPE_CHECKING:
    from dataeval import Metadata

    from dataeval_flow._policy import ResolvedPolicy
    from dataeval_flow._result import ResultMetadata

__all__ = [
    "attach_binning",
    "describe_binning",
    "descriptor_from_record",
    "divergent_factors",
    "encodings_agree",
    "write_descriptor",
]

_logger: logging.Logger = logging.getLogger(__name__)

# DataEval writes one companion column per factor holding its codes: bin indices
# for a binned continuous factor, category ordinals for a digitized one.  The
# suffixes come from dataeval's private ``_metadata._columns``; they are mirrored
# here rather than imported so that a private rename upstream costs us the
# occupancy detail rather than raising ImportError on a working install.
#
# Only ``fit`` reads them.  ``encoding`` comes from the public record, so a rename
# now costs the observation and leaves the policy — which is the half that has to
# be right — untouched.
_BINNED_SUFFIX = "↕"
_DIGITIZED_SUFFIX = "#"


def _default_factor_source() -> str | None:
    """What ``factor_source`` a workflow leaving it unset actually gets.

    Read from a constructed evaluator, not imported: the constant lives in a
    private module.  Constructing a ``Balance`` runs no statistics.
    """
    try:
        from dataeval.bias import Balance

        return str(Balance().factor_source)
    except Exception:  # noqa: BLE001 - a release without the setting records nothing for it
        return None


class _Descriptor(NamedTuple):
    """The three sections of the ``export_encoding`` output, by name.

    The corrections array was added after the others and broke call sites that only
    read ``factors``.
    """

    factors: dict[str, dict[str, Any]]
    corrections: list[dict[str, Any]]
    version: int | None


def _unusable(metadata: "Metadata") -> dict[str, dict[str, Any]]:
    """What it would take to read each factor the walk could not, keyed by factor name.

    Complements ``dropped``, which states only that a factor was dropped and why:
    writing the repair needs the counts and the distinct values as the dataset spelled
    them.

    ``repairable`` separates a column a :class:`~dataeval.types.Remap` or
    :class:`~dataeval.types.ParseValue` can recover from one that is unrecoverable.

    Best effort, like :func:`_descriptor`: a release without the accessor costs this
    section only.
    """
    unusable = getattr(metadata, "unusable", None)
    if not unusable:
        return {}
    return {
        name: to_serializable(
            {
                "reasons": list(entry.reasons),
                "level": entry.level,
                "repairable": bool(entry.repairable),
                "counts": dict(entry.counts),
                # The spelling the dataset used, not a normalized one: repairs are written
                # against the literal column content.
                "distinct": {kind: list(values) for kind, values in entry.distinct.items()},
                "sampled": bool(entry.sampled),
            }
        )
        for name, entry in unusable.items()
    }


def _descriptor(metadata: "Metadata") -> _Descriptor:
    """Every factor's encoding and every correction, as the committed descriptor spells them.

    Round-tripped through ``Metadata.export_encoding``: that writer is the only public
    one, and it owns the format details — an infinity is the word ``"inf"`` because
    JSON has no literal for one, a missing level is ``null``, and a NumPy scalar
    unwraps to the Python value it stands for.  Reimplementing it here would give the
    envelope, the cache sidecar and a committed descriptor three chances to disagree.

    The format version travels back with the factors and is recorded, not assumed: it
    belongs to DataEval, so reading it off what DataEval just wrote keeps it true when
    it changes.

    Corrections travel the same way: they decide what the values *are*, before any code
    is assigned, so a descriptor that kept only the factors would read as a run that
    declared no repairs.

    Best effort: a release that cannot write one costs the policy half of the record;
    the caller still gets ``fit``.
    """
    export = getattr(metadata, "export_encoding", None)
    if export is None:
        return _Descriptor({}, [], None)
    try:
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "encoding.json"
            export(path)
            document = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # the record is worth less than the run it would otherwise take down
        _logger.debug("Encoding record unavailable", exc_info=True)
        return _Descriptor({}, [], None)
    factors = document.get("factors")
    corrections = document.get("corrections")
    version = document.get("version")
    return _Descriptor(
        factors if isinstance(factors, dict) else {},
        corrections if isinstance(corrections, list) else [],
        version if isinstance(version, int) else None,
    )


def _code_names(metadata: "Metadata") -> dict[str, dict[str, str]]:
    """What each factor's codes read as, per DataEval's own naming.

    Captured here, not at report time: reports render from an archived envelope, and
    the names come from a live ``Metadata``.  Carrying them means an archived result
    re-renders to the same strings, and anything reading the JSON gets them without
    recomputing.

    Taken from DataEval, not derived from the edges, because the two must agree: these
    are the strings :attr:`ParityOutput.insufficient_data` reports and the ``label=``
    axis groups carry.  Precision matters: six significant figures collapses seven
    epoch-millisecond bins onto three labels.

    Best effort: a release without the accessor costs the names; codes stand in.
    """
    names = getattr(metadata, "code_names", None)
    if names is None:
        return {}
    try:
        # JSON has no integer keys, so codes are stringed here; a round-tripped record
        # comes back the same.
        return {factor: {str(code): label for code, label in lookup.items()} for factor, lookup in names().items()}
    except Exception:  # names are worth less than the record they describe
        _logger.debug("Code names unavailable", exc_info=True)
        return {}


def _declared_bins(record: Mapping[str, Any]) -> int:
    """Intervals the edges describe.

    Not every code a value can land in: a finitely bounded list also yields below-first
    and above-last catchalls for out-of-range values.  Their being empty is the expected
    case: every value fell inside the range described.  Mirrors DataEval's own
    definition, so "empty" means the same thing here as in the warning it raises.
    """
    return max(len(record.get("edges") or ()) - 1, 0)


def _codes(df: pl.DataFrame, companion: str) -> pl.Series | None:
    """The code column for one factor, or None where the companion column is absent."""
    return df[companion] if companion in df.columns else None


def _bin_fit(df: pl.DataFrame, name: str, record: Mapping[str, Any]) -> dict[str, Any] | None:
    """How the rows of this run fell into a cut the record describes."""
    codes = _codes(df, f"{name}{_BINNED_SUFFIX}")
    if codes is None or name not in df.columns:
        return None

    # Renamed to fixed internal names first: a factor is free to be called "count", "min"
    # or "max", and aggregating into those aliases alongside it is a duplicate-column error.
    grouped = (
        df.select([codes.alias("_code"), pl.col(name).alias("_value")])
        .drop_nulls()
        .group_by("_code")
        .agg(pl.len().alias("_n"), pl.col("_value").min().alias("_min"), pl.col("_value").max().alias("_max"))
        .sort("_code")
    )
    populated = {
        int(row["_code"]): {"code": int(row["_code"]), "count": int(row["_n"]), "min": row["_min"], "max": row["_max"]}
        for row in grouped.to_dicts()
    }

    declared = _declared_bins(record)
    # Codes 1..declared are the declared intervals. Below-first is 0 and above-last is
    # `declared + 1`; both are catchalls, and the missing code sits above them.
    fit: dict[str, Any] = {
        "bins": [populated[code] for code in sorted(populated) if 1 <= code <= declared],
        "empty": [code for code in range(1, declared + 1) if code not in populated],
    }
    for label, code in (("below_range", 0), ("above_range", declared + 1), ("missing", declared + 2)):
        if code in populated:
            fit[label] = populated[code]["count"]
    return fit


def _level_fit(df: pl.DataFrame, name: str, record: Mapping[str, Any]) -> dict[str, Any] | None:
    """How the rows of this run fell across a vocabulary the record describes."""
    codes = _codes(df, f"{name}{_DIGITIZED_SUFFIX}")
    if codes is None:
        return None

    counts = df.select([codes.alias("_code")]).drop_nulls().group_by("_code").agg(pl.len().alias("_n")).sort("_code")
    populated = {int(row["_code"]): int(row["_n"]) for row in counts.to_dicts()}

    # Named from the record, not the column, so a level the vocabulary holds and this
    # sample does not still appears.
    levels = list(record.get("levels") or ())
    return {
        "levels": [
            {"code": code, "value": value, "count": populated.get(code, 0)} for code, value in enumerate(levels)
        ],
        "empty": [code for code in range(len(levels)) if code not in populated],
    }


# Cells a recorded histogram is drawn into.  A fixed display resolution, not taken from
# the factor's bin count: a chart drawn at the same cut cannot judge it, and factors cut
# into different numbers of bins cannot be compared.
_DISTRIBUTION_CELLS = 40

# Order statistics recorded per numeric factor.  Bin-free by construction: they describe where
# the values are, not where somebody cut them.
_QUANTILES: tuple[float, ...] = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)


def _as_float(value: Any) -> float | None:
    """One numeric value as a float, or None where it is not a number.

    Polars returns loosely typed literals for a column's extremes and elements; they are
    narrowed to numbers here.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _distribution(df: pl.DataFrame, name: str) -> dict[str, Any] | None:
    """A numeric column's shape, independent of any cut.

    The order statistics, and a histogram at a fixed display resolution.  Neither is
    derived from the encoding, so a reader can judge a proposed bin count against the
    values.

    ``None`` for a non-numeric column, whose distribution is its vocabulary, already
    recorded by the level fit.
    """
    if name not in df.columns or not df[name].dtype.is_numeric():
        return None
    values = df[name].drop_nulls()
    # NaN is not null in polars, and a `remap` writes NaN for a value coded as unrecorded.
    # Left in, it poisons the extremes and the arithmetic below; a corrected re-run would
    # fail where the first run succeeded.
    if values.dtype.is_float():
        values = values.drop_nans()
    if not len(values):
        return None
    # Polars types a column's extremes as any Python literal, so they are narrowed once
    # here instead of at each arithmetic site below.
    low, high = _as_float(values.min()), _as_float(values.max())
    if low is None or high is None:
        return None
    quantiles = {str(q): values.quantile(q) for q in _QUANTILES}
    counts = [0] * _DISTRIBUTION_CELLS
    span = high - low
    last = _DISTRIBUTION_CELLS - 1
    for value in values:
        scalar = _as_float(value)
        if scalar is None:
            continue
        offset = 0.0 if not span else (scalar - low) / span * _DISTRIBUTION_CELLS
        counts[min(int(offset), last)] += 1
    return {"quantiles": quantiles, "histogram": counts, "cells": _DISTRIBUTION_CELLS}


def _factor_entry(
    name: str,
    info: Any,
    df: pl.DataFrame,
    record: Mapping[str, Any] | None,
    names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """One factor's record: what it is, how it was encoded, and how that encoding fits."""
    entry: dict[str, Any] = {"type": info.factor_type, "level": info.level}
    if getattr(info, "aggregated_from", None) is not None:
        entry["aggregated_from"] = info.aggregated_from
    # Distinct value count beside row count.  The encoding and the fit cannot say this:
    # a cut reports the intervals values fell into, regardless of how many distinct values
    # produced them.  A non-numeric column with one value per row is dropped as
    # `cardinality_over_budget`; a numeric one is binned and kept, so this is the only
    # place the shape is visible.
    if name in df.columns:
        entry["n_distinct"] = int(df[name].n_unique())
        entry["rows"] = int(df.height)
    distribution = _distribution(df, name)
    if distribution is not None:
        # Beside `fit`, not inside it: fit describes the rows against the cut; the
        # distribution describes the values without any cut.
        entry["distribution"] = distribution

    if record is None:
        # Neither encoding path was reached, or the record could not be read.
        return entry

    entry["encoding"] = dict(record)
    # Beside the record, not inside it: the encoding member is byte-identical to the
    # committed descriptor, and names are not part of the descriptor.
    if names:
        entry["names"] = dict(names)
    fit = _bin_fit(df, name, record) if record.get("kind") == "bins" else _level_fit(df, name, record)
    if fit is not None:
        entry["fit"] = fit
    return entry


def describe_binning(
    metadata: "Metadata",
    *,
    excluded: Sequence[str] | None = None,
    requested_bins: Mapping[str, int | Sequence[float]] | None = None,
    factor_source: str | None = None,
    declared_bins: Mapping[str, int | Sequence[float]] | None = None,
    injected: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Describe how every factor was encoded, and how well that encoding fits this run.

    Parameters
    ----------
    metadata : Metadata
        A bound DataEval ``Metadata``.  Reading ``factor_info`` forces binning,
        so both the record and the companion columns are guaranteed to exist by
        the time they are read.
    excluded : Sequence[str] | None
        Factor names the configuration excluded.  Recorded because an excluded
        factor leaves no other trace.
    requested_bins : Mapping[str, int | Sequence[float]] | None
        The configured ``continuous_factor_bins``.  Defaults to what the
        ``Metadata`` was constructed with; pass explicitly to record a request
        that named a factor the dataset does not carry.
    factor_source : str | None
        The configured ``factor_source``, or None to record DataEval's default.
        Recorded because it decides whether each bias statistic read a factor's
        codes or its measured values, and it leaves no other trace.
    declared_bins : Mapping[str, int | Sequence[float]] | None
        The bins as the policy spelled them, before expansion onto level-split names.
        Recorded as ``bin_expansion`` so a reader can see that a bin written on
        ``brightness`` became ``unit_brightness`` and ``instance_brightness``.
    injected : Sequence[str] | None
        Factor names this run synthesised from intrinsic statistics.  In a run with
        injection on most factors are synthesised, and their cuts are derived from this
        draw; the envelope must be able to say which.

    Returns
    -------
    dict
        JSON-serializable record with ``auto_bin_method``, ``encoding_digest``,
        ``descriptor_version``, ``factor_source``, ``requested_bins``, ``excluded``,
        per-factor ``factors``, ``unreviewed``, ``dropped``, and ``unusable``.
        Each factor carries its ``encoding`` (the policy) and its ``fit`` (what
        this run's rows did against it) — see the module docstring.

    Notes
    -----
    ``requested_bins`` records what was requested; ``factors[name]["encoding"]``
    records what was applied.  A request of ``10`` is a count; the applied cut
    positions are recorded per factor.
    """
    # Read factor_info first: it forces _bin(), which is what writes the
    # companion columns the frames below are read for.
    factor_info = metadata.factor_info

    if requested_bins is None:
        requested_bins = metadata.continuous_factor_bins or {}

    record: dict[str, Any] = {
        "auto_bin_method": getattr(metadata, "auto_bin_method", None),
        "encoding_digest": getattr(metadata, "encoding_digest", None),
        "factor_source": factor_source or _default_factor_source(),
        "requested_bins": dict(requested_bins),
        "excluded": list(excluded or ()),
        "factors": {},
        "dropped": {name: list(reasons) for name, reasons in metadata.dropped_factors.items()},
        # Beside `dropped` rather than inside it: that mapping is the reason, and this is
        # what to do about it.
        "unusable": _unusable(metadata),
    }

    # One frame per level, fetched once — rows_at() materializes a frame per call.
    rows_by_level: dict[str, pl.DataFrame] = {}

    described = _descriptor(metadata)
    encodings = described.factors
    record["descriptor_version"] = described.version
    # Top-level, not per-factor: they apply in order and one factor may take several.
    # Carried through the envelope untouched.
    record["corrections"] = described.corrections
    names = _code_names(metadata)

    for name, info in factor_info.items():
        if info.level not in rows_by_level:
            rows_by_level[info.level] = metadata.rows_at(info.level)
        record["factors"][name] = _factor_entry(
            name, info, rows_by_level[info.level], encodings.get(name), names.get(name)
        )

    # Factors whose encoding reads "derived" were chosen by DataEval from this sample and
    # not reviewed: their cuts are not stable across draws.  Collected so the envelope can
    # be gated on them.
    record["unreviewed"] = sorted(
        name
        for name, entry in record["factors"].items()
        if (entry.get("encoding") or {}).get("provenance") == "derived"
    )

    # What each declared name became.  Matched against real level prefixes, as
    # expand_declared_bins does, not by bare suffix: a suffix match would claim an
    # unrelated factor like `camera_brightness` as an expansion of `brightness`.  Only
    # names that moved are recorded.
    expansion: dict[str, list[str]] = {}
    for name in declared_bins or {}:
        candidates = {name, *(f"{level}_{name}" for level in metadata.levels)}
        landed = sorted(candidates & set(requested_bins))
        if landed != [name]:
            expansion[name] = landed
    record["bin_expansion"] = expansion
    record["injected_factors"] = sorted(set(injected or ()) & set(factor_info))

    # DataEval silently ignores a request naming a factor the dataset does not carry
    # (it warns and moves on); recorded here.
    unmatched = sorted(set(requested_bins) - set(factor_info))
    if unmatched:
        record["unmatched_bin_requests"] = unmatched

    return to_serializable(record)


def attach_binning(
    result_metadata: "ResultMetadata",
    metadata: "Metadata | Mapping[str, Metadata]",
    policy: "ResolvedPolicy",
) -> None:
    """Record binning decisions on a workflow's metadata envelope.

    Reads the resolved policy, not the workflow's ``metadata_*`` fields: naming a
    policy leaves those fields empty, and a run under a named policy would
    otherwise record ``excluded: []`` and DataEval's default factor source.

    Accepts either a single ``Metadata`` or a mapping of split name to one, so a
    multi-split workflow records each split separately — splits are binned
    independently, and two splits of the same dataset can land on different
    edges.

    Also stamps :attr:`ResultMetadata.encoding_digest`, which makes two archived
    results comparable: without it a bias score that moved between runs cannot be
    attributed to the override or to the data.  For a multi-split workflow it is
    set only where every split agrees; the per-split digests say which differed.

    Never raises: an upstream column rename costs the record, not the run.
    """
    try:
        from dataeval.flags import ImageStats

        from dataeval_flow._metadata import expand_declared_bins, resolve_families, stat_names_for

        excluded = list(policy.exclude) or None
        declared = dict(policy.continuous_factor_bins) or None
        source = policy.factor_source

        # The statistics a policy's families produce, band-group and level prefixes
        # included.  Derived from the flags, not the Metadata, so a cache hit, which never
        # ran the injector, marks the same factors as a cache miss.
        #
        # A stats policy decides which views the injector reads, so the names it produces
        # carry those views' prefixes: `factors_from: [~, rgb]` injects `rgb_brightness`
        # beside `brightness`. Deriving from the statistic names alone would leave every
        # band factor out of this record.
        injected: set[str] = set()
        if policy.intrinsic_factors:
            families = resolve_families("image", policy.intrinsic_factors)
            # Band views are an image-statistics idea, so a non-image modality keeps the
            # bare names. This function never raises, so the narrowing degrades rather
            # than rejecting. `_inject` is where a mismatch is an error.
            if policy.stats is None or not isinstance(families, ImageStats):
                bare = set(stat_names_for(families))
            else:
                from dataeval_flow._stats import columns_for

                bare = columns_for(policy.stats.factors_from, families)
            injected = set(bare) | {f"{level}_{name}" for level in ("unit", "instance") for name in bare}

        def _describe(md: "Metadata") -> dict[str, Any]:
            # The bins as applied, not as spelled: `unmatched_bin_requests` is a set
            # difference against the factor names, and the declared bare name is not one.
            requested = expand_declared_bins(declared, md.factor_names, md.levels) if declared else None
            return describe_binning(
                md,
                excluded=excluded,
                requested_bins=requested,
                factor_source=source,
                declared_bins=declared,
                injected=sorted(injected),
            )

        if isinstance(metadata, Mapping):
            per_split = {name: _describe(md) for name, md in metadata.items()}
            result_metadata.metadata_binning = {"per_split": per_split}
            result_metadata.encoding_digest = _common_digest(per_split.values())
        else:
            record = _describe(metadata)
            result_metadata.metadata_binning = record
            result_metadata.encoding_digest = record.get("encoding_digest")
    except Exception:
        _logger.warning("Binning record unavailable", exc_info=True)


def _common_digest(records: "Iterable[Mapping[str, Any]]") -> str | None:
    """The one encoding every split ran under, or None where they did not share one.

    Splits are binned independently and can land on different edges; a single
    top-level digest must not hide that.  None says the splits are not comparable on
    factors; the per-split digests say which differed.
    """
    digests = {record.get("encoding_digest") for record in records}
    if len(digests) != 1:
        return None
    only = digests.pop()
    return str(only) if only is not None else None


def descriptor_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Render a binning record into the descriptor a person commits.

    The input is a record, not a live ``Metadata``, so the artifact is obtainable from
    an archived result without re-running the task.

    Byte-compatible with :meth:`dataeval.Metadata.export_encoding` by construction: the
    per-factor entries are exactly what that writer produced, carried through the envelope
    untouched.  What comes out here is what a policy's ``encoding`` takes back in.

    Parameters
    ----------
    record : Mapping
        One ``metadata_binning`` record — either a single run's or one split's.

    Returns
    -------
    dict
        ``{"version": ..., "factors": {...}}``.

    Raises
    ------
    ValueError
        When the record carries no encodings, or when it holds several splits that were
        not encoded alike: no single descriptor describes them.
    """
    if "per_split" in record:
        record = _one_split(record["per_split"])

    factors = {
        name: entry["encoding"] for name, entry in (record.get("factors") or {}).items() if entry.get("encoding")
    }
    if not factors:
        raise ValueError(
            "This result records no encodings, so there is no descriptor to write. Only a "
            "workflow that builds metadata produces one.",
        )
    version = record.get("descriptor_version")
    corrections = record.get("corrections")
    return {
        "version": version if isinstance(version, int) else 1,
        # Always written, empty included: DataEval writes the array unconditionally, and
        # an absent key would read as a different document.
        "corrections": corrections if isinstance(corrections, list) else [],
        "factors": factors,
    }


def encodings_agree(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Whether two encodings of one factor assign the same meaning to the same codes.

    Not equality.  A vocabulary grows **append-only**: a category the other split never saw
    takes the next free code and goes on the end, so every code they share still stands for
    the same value.

    Bin edges have no append: anything but identical edges means the same code names a
    different interval.

    Defined here, not in the renderer, because it is a property of the record: the
    descriptor writer and the report must reach the same verdict.
    """
    if a == b:
        return True
    if a.get("kind") != b.get("kind") or a.get("kind") != "levels":
        return False
    first, second = a.get("levels") or [], b.get("levels") or []
    shorter, longer = sorted((first, second), key=len)
    return list(longer[: len(shorter)]) == list(shorter)


def _widest(encodings: "Sequence[Mapping[str, Any]]") -> Mapping[str, Any]:
    """The encoding that subsumes the rest, for factors whose encodings agree.

    Append-only means every vocabulary is a prefix of the longest, so the longest is the
    one that names every code any split used.
    """
    return max(encodings, key=lambda encoding: len(encoding.get("levels") or ()))


def _encodings_by_factor(per_split: "Iterable[Mapping[str, Any]]") -> dict[str, list[Mapping[str, Any]]]:
    """Every split's encoding for each factor, keyed by factor name."""
    by_factor: dict[str, list[Mapping[str, Any]]] = {}
    for record in per_split:
        for name, info in (record.get("factors") or {}).items():
            if encoding := info.get("encoding"):
                by_factor.setdefault(name, []).append(encoding)
    return by_factor


def divergent_factors(per_split: "Iterable[Mapping[str, Any]]") -> list[str]:
    """Factors that do not mean the same thing in every split.

    Every encoding is compared against the widest, because the append-only rule is not
    transitive: two splits that each extend a third can still assign the same code to
    different values.
    """
    return sorted(
        name
        for name, seen in _encodings_by_factor(per_split).items()
        if any(not encodings_agree(_widest(seen), other) for other in seen)
    )


def _one_split(per_split: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    """The one encoding a multi-split result ran under, or a refusal naming the problem.

    Splits whose vocabularies merely grew describe one encoding: the widest names every
    code any split used and agrees with all of them on the codes they share.  Refused only
    where a factor means different things in different splits — the same test the report
    renders.
    """
    if not per_split:
        raise ValueError(
            "This result records no splits, so there is no descriptor to write. Only a "
            "workflow that builds metadata produces one.",
        )
    if divergent := divergent_factors(per_split.values()):
        digests = {name: split.get("encoding_digest") for name, split in per_split.items()}
        raise ValueError(
            f"These splits encode {divergent} differently ({digests}), so no one descriptor "
            "describes the run. Give every split the same encoding — set a policy's "
            "`reference_split`, or apply a committed `encoding` — and re-run.",
        )
    # Corrections are compared exactly, with no reconciliation: a repair decides what the
    # values *are*, so splits repaired differently were measured on different data.  Order
    # counts as difference: corrections apply in sequence, and one factor may take several.
    repairs = {name: split.get("corrections") or [] for name, split in per_split.items()}
    if len({json.dumps(entry, sort_keys=True) for entry in repairs.values()}) > 1:
        raise ValueError(
            f"These splits repair their factors differently ({repairs}), so no one descriptor "
            "describes the run. A correction changes what the values are, not how they are "
            "cut, so there is no widest reading to fall back on. Declare the same "
            "corrections for every split and re-run.",
        )

    widest = {name: _widest(seen) for name, seen in _encodings_by_factor(per_split.values()).items()}
    template = next(iter(per_split.values()))
    return {
        **template,
        "factors": {
            name: {**entry, "encoding": widest[name]} if name in widest else entry
            for name, entry in (template.get("factors") or {}).items()
        },
    }


def write_descriptor(record: Mapping[str, Any], path: "str | Path") -> None:
    """Write the descriptor for one binning record, for review and for committing.

    JSON with sorted keys and a fixed indent, matching what DataEval writes, so that the
    same encoding produces the same bytes and a change to one factor reads as a change to
    one factor in a pull request.
    """
    document = json.dumps(descriptor_from_record(record), indent=2, sort_keys=True, allow_nan=False) + "\n"
    Path(path).write_text(document, encoding="utf-8")
