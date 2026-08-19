"""Record of how metadata factors were encoded, and how well that encoding fits.

A bin edge is a claim about the world — *below 0 °C is freezing* — so where the
cuts fell, and who chose them, is part of what a result computed from the factor
means.  Balance, diversity and parity read codes rather than the values measured,
and the map from one to the other is the thing a reviewer comparing two runs has
to be able to see.

Two members per factor, answering two different questions:

``encoding``
    The **policy**: the edges or the vocabulary, who chose them (``provenance``),
    and how they were placed (``method``).  Read from DataEval's own record and
    rendered by DataEval's own writer, so this is byte-for-byte what a committed
    descriptor holds and what the cache sidecar names.
``fit``
    The **observation**: how many rows reached each code in this run, the span
    they occupied, and which declared bins nothing reached at all.

The split matters because it was previously collapsed.  This module used to
reconstruct the policy from the observed contents of each bin, which described
the draw rather than the decision: the same cut over a different sample printed
a different record, and a declared cutoff never survived into its own label —
``{"temp_c": [-inf, 0.0, inf]}`` was reported as ``[-40, -0.3]``, with nothing
saying that zero was where the meaning was.  The record now comes from the record.
Occupancy is still measured, and is still worth having; it is just no longer
mistaken for the policy.
"""

import json
import logging
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from dataeval_flow.workflows._common import to_serializable

if TYPE_CHECKING:
    from dataeval import Metadata

    from dataeval_flow.config.schemas._metadata import ResultMetadata
    from dataeval_flow.workflow.base import MetadataConfigMixin

__all__ = ["attach_binning", "descriptor_from_record", "describe_binning", "write_descriptor"]

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

    Read off a constructed evaluator rather than imported: the constant lives in a
    private module, and this record is worth less than the run it would take down on a
    release that spells it differently.  Constructing a ``Balance`` runs no statistics.
    """
    try:
        from dataeval.bias import Balance

        return str(Balance().factor_source)
    except Exception:  # noqa: BLE001 - a release without the setting records nothing for it
        return None


def _descriptor(metadata: "Metadata") -> tuple[dict[str, dict[str, Any]], int | None]:
    """Every factor's encoding, as the committed descriptor spells it.

    Round-tripped through ``Metadata.export_encoding`` rather than rendered here.  That
    writer is the only public one, and it owns decisions this module should not be making
    a second time: an infinity is the word ``"inf"`` because JSON has no literal for one,
    a missing level is ``null`` rather than a bare ``NaN`` token no other reader accepts,
    and a NumPy scalar unwraps to the Python value it stands for.  Reimplementing that
    here would give the envelope, the cache sidecar and a committed descriptor three
    chances to disagree about what the same record is.

    The format version travels back with the factors and is recorded rather than assumed.
    A descriptor written from this envelope has to say which format it is — that is the
    whole point of the field — and the number belongs to DataEval, so reading it off what
    DataEval just wrote is the only spelling that stays true when it changes.

    Best effort: a release that cannot write one costs the policy half of the record, and
    the caller still gets ``fit``.
    """
    export = getattr(metadata, "export_encoding", None)
    if export is None:
        return {}, None
    try:
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "encoding.json"
            export(path)
            document = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # the record is worth less than the run it would otherwise take down
        _logger.debug("Encoding record unavailable", exc_info=True)
        return {}, None
    factors = document.get("factors")
    version = document.get("version")
    return (factors if isinstance(factors, dict) else {}), (version if isinstance(version, int) else None)


def _code_names(metadata: "Metadata") -> dict[str, dict[str, str]]:
    """What each factor's codes read as, per DataEval's own naming.

    Captured here rather than rendered when a report is drawn, because a report is drawn
    from an archived envelope and the names come from a live ``Metadata``.  Carrying them
    means an archived result re-renders to the same strings, and anything reading the JSON
    gets them without recomputing.

    Asked of DataEval rather than derived from the edges, because the two must agree:
    these are the strings :attr:`ParityOutput.insufficient_data` reports and the ``label=``
    axis groups carry, and choosing a precision is not the local decision it looks like —
    six significant figures collapses seven epoch-millisecond bins onto three labels.

    Best effort: a release without the accessor costs the names, and codes stand in.
    """
    names = getattr(metadata, "code_names", None)
    if names is None:
        return {}
    try:
        # JSON has no integer keys, so a record that round-trips would come back stringed
        # anyway. Stringed here so it reads the same either way.
        return {factor: {str(code): label for code, label in lookup.items()} for factor, lookup in names().items()}
    except Exception:  # names are worth less than the record they describe
        _logger.debug("Code names unavailable", exc_info=True)
        return {}


def _declared_bins(record: Mapping[str, Any]) -> int:
    """Intervals the edges describe — the bins the cut is a claim *about*.

    Not every code a value can land in.  Digitizing has to put an out-of-range value
    somewhere, so a finitely bounded list also yields a below-first and an above-last
    catchall.  Nobody declared those and their being empty is the *good* case: it says
    every value fell inside the range described.  Mirrors DataEval's own definition, so
    "empty" means the same thing here as in the warning it raises.
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
    # Codes 1..declared are the intervals somebody described. Below-first is 0 and
    # above-last is `declared + 1`; both are catchalls nobody asked for, and the missing
    # code sits above them.
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

    # Named from the record rather than from the column, so a level the vocabulary holds
    # and this sample does not still appears — which is the whole point of asking.
    levels = list(record.get("levels") or ())
    return {
        "levels": [
            {"code": code, "value": value, "count": populated.get(code, 0)} for code, value in enumerate(levels)
        ],
        "empty": [code for code in range(len(levels)) if code not in populated],
    }


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

    if record is None:
        # Neither encoding path was reached, or the record could not be read.
        return entry

    entry["encoding"] = dict(record)
    # Beside the record rather than inside it: the encoding member is byte-for-byte what a
    # committed descriptor holds, and names are read off a record rather than part of one.
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
        JSON-serializable record with ``auto_bin_method``, ``encoding_digest``,
        ``descriptor_version``, ``factor_source``, ``requested_bins``, ``excluded``,
        per-factor ``factors``, ``unreviewed``, and ``dropped``.
        Each factor carries its ``encoding`` (the policy) and its ``fit`` (what
        this run's rows did against it) — see the module docstring.

    Notes
    -----
    ``requested_bins`` records what was *asked for*; ``factors[name]["encoding"]``
    records what was *applied*, which is not the same thing.  A request of ``10``
    is a count, and where its nine interior cuts landed used to be discarded.
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
    }

    # One frame per level, fetched once — rows_at() materializes a frame per call.
    rows_by_level: dict[str, pl.DataFrame] = {}

    encodings, descriptor_version = _descriptor(metadata)
    record["descriptor_version"] = descriptor_version
    names = _code_names(metadata)

    for name, info in factor_info.items():
        if info.level not in rows_by_level:
            rows_by_level[info.level] = metadata.rows_at(info.level)
        record["factors"][name] = _factor_entry(
            name, info, rows_by_level[info.level], encodings.get(name), names.get(name)
        )

    # The state a reviewer audits for. A factor whose encoding reads "derived" is one
    # DataEval chose from this sample and nobody has looked at: its cuts are not stable
    # across draws, and pinning them without reading them locks in an accident. Counted
    # here so the envelope can be gated on it rather than a reader having to scan.
    record["unreviewed"] = sorted(
        name
        for name, entry in record["factors"].items()
        if (entry.get("encoding") or {}).get("provenance") == "derived"
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

    Also stamps :attr:`ResultMetadata.encoding_digest`, which is what makes two
    archived results comparable: a bias score that moved between runs is otherwise
    unattributable between *my override worked* and *the data changed*.  For a
    multi-split workflow it is set only where every split agrees, because there is
    no single encoding to name when they do not — and the per-split digests say so.

    Never raises.  A companion column renamed upstream costs the record, not the
    run, and the diagnostics captured alongside it still name the decision.
    """
    try:
        excluded = list(params.metadata_exclude) if params.metadata_exclude else None
        requested = params.metadata_continuous_factor_bins
        source = params.metadata_factor_source

        if isinstance(metadata, Mapping):
            per_split = {
                name: describe_binning(md, excluded=excluded, requested_bins=requested, factor_source=source)
                for name, md in metadata.items()
            }
            result_metadata.metadata_binning = {"per_split": per_split}
            result_metadata.encoding_digest = _common_digest(per_split.values())
        else:
            record = describe_binning(metadata, excluded=excluded, requested_bins=requested, factor_source=source)
            result_metadata.metadata_binning = record
            result_metadata.encoding_digest = record.get("encoding_digest")
    except Exception:
        _logger.warning("Binning record unavailable", exc_info=True)


def _common_digest(records: "Iterable[Mapping[str, Any]]") -> str | None:
    """The one encoding every split ran under, or None where they did not share one.

    Splits are binned independently, so two splits of one dataset can land on different
    edges — which is exactly the case a single top-level digest must not paper over.
    Answering None says *these are not comparable on factors*, and the per-split digests
    say which differed.
    """
    digests = {record.get("encoding_digest") for record in records}
    if len(digests) != 1:
        return None
    only = digests.pop()
    return str(only) if only is not None else None


def descriptor_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Render a binning record back into the descriptor a person commits.

    Stage six of the lifecycle: lock the encoding in and put it under review.  The record
    is where this comes from rather than a live ``Metadata``, because the artifact has to
    be obtainable from a result somebody archived weeks ago — which is the case where
    pinning an encoding actually matters, and the one where re-running to get it is the
    thing you are trying to avoid.

    Byte-compatible with :meth:`dataeval.Metadata.export_encoding` by construction: the
    per-factor entries are exactly what that writer produced, carried through the envelope
    untouched.  So what comes out here is what goes back in through a policy's
    ``encoding``.

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
        When the record carries no encodings at all, or when it holds several splits that
        were not encoded alike — there is no single descriptor to write for those, and
        writing one of them would silently pick a policy nobody chose.
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
    return {"version": version if isinstance(version, int) else 1, "factors": factors}


def _one_split(per_split: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    """The single encoding a multi-split result ran under, or a refusal naming the problem."""
    digests = {name: split.get("encoding_digest") for name, split in per_split.items()}
    if len(set(digests.values())) > 1:
        raise ValueError(
            f"These splits were encoded differently ({digests}), so no one descriptor "
            "describes the run. Give every split the same encoding — set a policy's "
            "`reference_split`, or apply a committed `encoding` — and re-run.",
        )
    return next(iter(per_split.values()))


def write_descriptor(record: Mapping[str, Any], path: "str | Path") -> None:
    """Write the descriptor for one binning record, for review and for committing.

    JSON with sorted keys and a fixed indent, matching what DataEval writes, so that the
    same encoding produces the same bytes and a change to one factor reads as a change to
    one factor in a pull request.
    """
    document = json.dumps(descriptor_from_record(record), indent=2, sort_keys=True, allow_nan=False) + "\n"
    Path(path).write_text(document, encoding="utf-8")
