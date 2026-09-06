"""Find what a metadata run silently failed to read or encode well.

Pure analysis over the record :func:`~dataeval_flow.binning.describe_binning` produces.
Nothing here imports ``dataeval``: the record is a plain JSON dict, which is what lets these
run against an archived result envelope as readily as against a live run, and lets their
tests be hand-written dicts rather than datasets.

Best-effort throughout. The record degrades section by section upstream — a release without
an accessor costs `_descriptor` or `_unusable` and leaves the rest — so a missing key yields
no finding for that category rather than an exception. A triage that dies because one
section was unavailable is worse than one that reports the other four.
"""

__all__ = [
    "Category",
    "Finding",
    "Severity",
    "Suggestion",
    "find_issues",
    "incomplete_factors",
    "render_stanza",
    "suggest",
    "to_policy_stanza",
]

import difflib
import math
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Category = Literal["unreadable", "unbound_request", "unbinned", "unreviewed", "degenerate", "floor_mass"]
Severity = Literal["blocking", "warning", "note"]

#: Severity order for report grouping.  Blocking first: it is what the run did less of than
#: the configuration asked for.
_SEVERITY_RANK: Mapping[str, int] = {"blocking": 0, "warning": 1, "note": 2}

#: Fewest bins a suggestion will propose.  One bin is not a cut.
_MIN_BINS = 2

#: Rows a column needs before never repeating a value says anything about it.  A handful of
#: measurements are all distinct as a matter of course.
_IDENTIFIER_MIN_ROWS = 50

#: Other columns that must share an extreme before it reads as a convention rather than a
#: coincidence.  Corroboration only: the mass itself is what the finding is about.
_SHARED_EXTREME_FACTORS = 2

#: Values standing for "no reading", whatever the column otherwise holds.  A closed literal
#: list rather than a pattern, so what it claims is auditable.
_SENTINELS = frozenset(
    {
        "",
        "-",
        "--",
        "n/a",
        "na",
        "n\\a",
        "none",
        "null",
        "nil",
        "nan",
        "unknown",
        "unspecified",
        "missing",
        "-999",
        "-9999",
    }
)

#: Absolute periods, coarsest first.  The recurring half of DATETIME_GRANULARITIES
#: (`month_of_year`, `day_of_week`, `hour_of_day`) is deliberately absent: an absolute period
#: runs once, so a dataset split by time separates perfectly on any of them, while a
#: recurring position stays comparable across the split.  Which a user wants depends on the
#: question they are asking, so the suggestion emits the absolute reading and names the
#: alternative rather than choosing it.
_ABSOLUTE_PERIODS: tuple[str, ...] = ("year", "quarter", "month", "week", "day", "hour")

#: Formats tried for a timestamp, most specific first.  Where more than one reads every
#: value the column is genuinely ambiguous and nothing is suggested.
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
)

#: Characters the loose fallback in :func:`_numeric_drop` is allowed to strip: punctuation,
#: whitespace and currency symbols.  Letters are deliberately excluded — stripping them would
#: turn an identifier into a number (``"a7f"`` -> ``"7"``), which is corruption, not a repair.
#: A unit suffix like "kg" is still reachable: :func:`_common_suffix` strips each value down
#: to its trailing non-numeric run (letters included) and offers that whole run as its own
#: candidate, rather than this fallback stripping the letters character by character.
_LOOSE_DROPPABLE = frozenset(" \t,.;:_'\"()[]{}!?/\\|~^$€£¥¢")


class Suggestion(BaseModel):
    """A config change that would address one finding.

    Two halves because the findings are fixed two different ways: a correction changes what
    the values *are*, and a policy edit changes how they are cut.  Both land in the same
    ``metadata:`` stanza, which is why they travel together.
    """

    corrections: list[dict[str, Any]] = Field(default_factory=list)
    policy: dict[str, Any] = Field(default_factory=dict)
    complete: bool = False
    """Whether this runs as emitted.  False wherever a value is left for the user to code,
    which is what keeps verification from reporting recovery from a placeholder."""


class Finding(BaseModel):
    """One thing a run did that nobody asked for, and what to do about it."""

    factor: str
    category: Category
    severity: Severity
    reasons: tuple[str, ...] = ()
    level: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    repairable: bool = False
    remedy: str = ""
    suggestion: Suggestion | None = None


def find_issues(
    record: Mapping[str, Any],
    *,
    min_missing_fraction: float = 0.2,
    default_bins: int = 10,
) -> list[Finding]:
    """Every finding a binning record supports, worst first.

    Parameters
    ----------
    record : Mapping[str, Any]
        A :func:`~dataeval_flow.binning.describe_binning` record, live or off an archived
        envelope.
    min_missing_fraction : float, default 0.2
        Share of rows recording no value above which a factor is called degenerate.
    default_bins : int, default 10
        Bin count a suggestion falls back to where the run left no ``fit`` to read.  The
        populated bins of the derived cut are preferred wherever there are any: the auto-cut
        already found where the values are, and pinning that is not the same as replacing it.

    Returns
    -------
    list[Finding]
        Sorted by severity, then category, then factor name.
    """
    findings = [
        *_unreadable(record),
        *_unbound(record),
        *_encodings(record, default_bins),
        *_degenerate(record, min_missing_fraction),
        *_floor_mass(record),
    ]
    # A factor can carry more than one finding at once, and where it does the bin count is the
    # one to withdraw. A `degenerate` cut is one the same report calls useless; a `sentinel`
    # cut was derived from values that include a marker for "not recorded". Pinning either
    # would fix an accident in place. Both findings stay — they are true, and the reader should
    # see them — but neither leaves a cut behind in the stanza.
    withdrawn = {f.factor for f in findings if f.category in ("degenerate", "floor_mass")}
    for finding in findings:
        if finding.category == "unbinned" and finding.factor in withdrawn:
            finding.suggestion = None
        elif finding.suggestion is None:
            # Detectors that build their own suggestion keep it: `suggest` covers the two
            # categories it can derive one for and answers None everywhere else.
            finding.suggestion = suggest(finding, default_bins=default_bins)
        if finding.category == "unreadable" and finding.repairable and finding.suggestion is None:
            values = finding.detail.get("distinct", {}).get("text", [])
            if values and _datetime_format(values) is False and _looks_like_dates(values):
                finding.remedy = "ambiguous date format: multiple formats match"
        _name_recurring_alternative(finding)
    return sorted(findings, key=lambda f: (_SEVERITY_RANK[f.severity], f.category, f.factor))


#: The recurring counterpart to an absolute period this module chooses automatically, named
#: in the remedy so a user learns it exists without reading the source.  Only the three
#: periods DataEval gives one — ``year``, ``quarter`` and ``week`` have no recurring form.
_RECURRING_ALTERNATIVE: Mapping[str, str] = {
    "month": "month_of_year",
    "day": "day_of_week",
    "hour": "hour_of_day",
}


def _name_recurring_alternative(finding: Finding) -> None:
    """Append a sentence naming the recurring reading, where the suggestion chose an absolute one.

    The suggestion itself always emits the absolute period — a recurring one is never chosen
    automatically, because it answers a different question (see the module docstring on
    :data:`_ABSOLUTE_PERIODS`).  The remedy is where a reader learns the alternative exists at
    all; the YAML the suggestion renders is unchanged.
    """
    suggestion = finding.suggestion
    if suggestion is None or len(suggestion.corrections) != 1:
        return
    correction = suggestion.corrections[0]
    if correction.get("kind") != "parse_datetime":
        return
    every = correction.get("every")
    alternative = _RECURRING_ALTERNATIVE.get(every) if isinstance(every, str) else None
    if alternative:
        finding.remedy = f"{finding.remedy}. `{alternative}` is also available for cyclical analysis"


def _factors(record: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    """The record's factor entries, or an empty mapping where it carries none."""
    factors = record.get("factors")
    return factors if isinstance(factors, Mapping) else {}


def _unreadable(record: Mapping[str, Any]) -> Iterator[Finding]:
    """Factors the walk could not read.

    ``distinct`` is carried whole rather than sampled: it is what a repair has to be written
    against, and a truncated set would build a correction that looks complete and is not.
    """
    unusable = record.get("unusable")
    if not isinstance(unusable, Mapping):
        return
    for name, entry in sorted(unusable.items()):
        if not isinstance(entry, Mapping):
            continue
        repairable = bool(entry.get("repairable"))
        reasons = tuple(entry.get("reasons") or ())
        yield Finding(
            factor=name,
            category="unreadable",
            severity="blocking" if repairable else "note",
            reasons=reasons,
            level=entry.get("level"),
            repairable=repairable,
            detail={
                "counts": dict(entry.get("counts") or {}),
                "distinct": {k: list(v) for k, v in (entry.get("distinct") or {}).items()},
                "sampled": bool(entry.get("sampled")),
            },
            remedy=_unreadable_remedy(reasons, repairable),
        )


def _unreadable_remedy(reasons: Sequence[str], repairable: bool) -> str:
    """One line saying what this factor needs, in the reason's own terms."""
    if not repairable:
        return "non-scalar data; cannot be processed as a metadata factor"
    if "cardinality_over_budget" in reasons:
        return "high cardinality: unique value per row. Bin or group values before use"
    return "mixed types; remap or cast values to a single type"


def _unbound(record: Mapping[str, Any]) -> Iterator[Finding]:
    """Bin requests that named nothing.

    DataEval warns and moves on, so the declared cut is silently replaced by an auto-cut —
    which is why this is blocking rather than a note.
    """
    known = sorted(_factors(record))
    for name in record.get("unmatched_bin_requests") or ():
        near = difflib.get_close_matches(str(name), known, n=3, cutoff=0.6)
        yield Finding(
            factor=str(name),
            category="unbound_request",
            severity="blocking",
            detail={"near": near},
            remedy=(
                f"no factor named {name!r}. Did you mean {near[0]!r}?"
                if near
                else f"factor {name!r} not found; falling back to automatic binning"
            ),
        )


def _no_encoding_remedy(factor_type: Any) -> str:
    """What a factor needs once it is known to have no encoding at all."""
    if factor_type == "continuous":
        return "unencoded continuous values; declare a bin count"
    return "unencoded values; declare levels or provide a descriptor"


def _encodings(record: Mapping[str, Any], default_bins: int) -> Iterator[Finding]:
    """Factors whose cut or vocabulary nobody pinned.

    The record carries a precomputed ``unreviewed`` list holding both halves.  This
    partitions it instead of reading it, because the two halves take different remedies —
    reading the list directly would report every binned factor twice.

    The split is on the **encoding's kind**, not on the factor's type.  The remedy is about
    the encoding: a cut is pinned by declaring its bin count, and a vocabulary drawn from
    this sample can only be pinned by committing the descriptor that holds it.  Factor type
    does not decide which of those a factor got — DataEval reports SeaDrone's ``altitude``
    as ``discrete`` and bins it by ``uniform_width`` all the same — so keying on the type
    filed every real numeric column under ``unreviewed`` and told it to export a descriptor,
    and ``continuous_factor_bins`` was never suggested for anything.
    """
    for name, info in sorted(_factors(record).items()):
        encoding = info.get("encoding")
        if not encoding:
            yield Finding(
                factor=name,
                category="unbinned",
                severity="blocking",
                level=info.get("level"),
                detail={"type": info.get("type")},
                remedy=_no_encoding_remedy(info.get("type")),
            )
            continue
        if encoding.get("provenance") != "derived":
            continue
        # The whole factor entry, not its parts: the renderer reads `fit` and `encoding`
        # together, and carrying all three would triple the JSON for one chart.
        if encoding.get("kind") != "levels":
            count = _suggested_bins(info.get("fit"), default_bins)
            yield Finding(
                factor=name,
                category="unbinned",
                severity="warning",
                level=info.get("level"),
                detail={"info": dict(info)},
                remedy=(
                    f"derived from current sample. Declare `continuous_factor_bins: {{{name}: {count}}}` "
                    "to fix bin edges across runs"
                ),
            )
        else:
            yield Finding(
                factor=name,
                category="unreviewed",
                severity="warning",
                level=info.get("level"),
                detail={"info": dict(info)},
                remedy=(
                    "vocabulary derived from current sample. Export with `dataeval-flow encoding` "
                    "and reference in `encoding:`"
                ),
            )


def _suggested_bins(fit: Mapping[str, Any] | None, default_bins: int) -> int:
    """The populated bins the derived cut produced, or the fallback where there is no fit."""
    bins = (fit or {}).get("bins") or []
    return max(len(bins), _MIN_BINS) if bins else default_bins


def _buckets(fit: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """This fit's buckets, whichever encoding produced it."""
    return list(fit.get("bins") or fit.get("levels") or [])


def _fit_total(fit: Mapping[str, Any]) -> int:
    """Every row the fit describes, catchalls and unrecorded rows included."""
    total = sum(int(b.get("count") or 0) for b in _buckets(fit))
    for key in ("below_range", "above_range", "missing"):
        total += int(fit.get(key) or 0)
    return total


def _degenerate(record: Mapping[str, Any], min_missing_fraction: float) -> Iterator[Finding]:
    """Factors read successfully that carry no usable signal.

    A *wide* vocabulary is not one of these: thin levels are reported by
    ``ParityOutput.insufficient_data`` rather than removed.  What is reported here is a
    factor with one populated bucket — which groups nothing — one losing most of its rows to
    no recorded value, and one holding a different value on nearly every row.
    """
    for name, info in sorted(_factors(record).items()):
        identifier = _identifier_finding(name, info)
        if identifier is not None:
            yield identifier
            continue
        fit = info.get("fit")
        if not isinstance(fit, Mapping):
            continue
        total = _fit_total(fit)
        if not total:
            continue
        populated = [b for b in _buckets(fit) if int(b.get("count") or 0) > 0]
        if len(populated) <= 1:
            yield Finding(
                factor=name,
                category="degenerate",
                severity="note",
                level=info.get("level"),
                detail={"info": dict(info), "populated": len(populated)},
                remedy="only one populated bucket; exclude factor or adjust binning",
            )
            continue
        missing = int(fit.get("missing") or 0)
        if missing / total >= min_missing_fraction:
            share = round(100 * missing / total)
            yield Finding(
                factor=name,
                category="degenerate",
                severity="note",
                level=info.get("level"),
                detail={"info": dict(info), "missing_fraction": missing / total},
                remedy=f"{share}% of rows have no value; high missingness may distort downstream analysis",
            )


def _identifier_finding(name: str, info: Mapping[str, Any]) -> Finding | None:
    """A column holding its own value on nearly every row, or None where it groups them.

    Such a column names its rows rather than grouping them, so it carries nothing for any
    statistic that works by comparing groups — every group has one member.  Upstream refuses
    exactly this shape when the values are text, dropping it as ``cardinality_over_budget``.
    A *numeric* column in the same position is binned and kept instead, which is how
    SeaDrone's ``object_id`` — 1305 values over 1305 detections — arrives as a factor cut into
    twelve arbitrary intervals.

    Reported as ``degenerate`` because that is what it is, which also means the suppression in
    :func:`find_issues` withdraws the bin count that would otherwise be suggested for it.
    Pinning that cut is the one piece of advice here that would make things worse: it would
    fix an accident in place and lend it the authority of a declared decision.

    **Never repeating a value is not on its own the evidence**, which is the trap this rule has
    to avoid: a measurement taken at any real precision is all-distinct too, and binning one of
    those is exactly what binning is for.  What separates them is that the values are *whole
    numbers*.  A measured integer quantity repeats — SeaDrone's ``altitude`` holds 78 values
    over 200 frames, its ``frame`` 168 — so an integer column that reaches a thousand rows
    without ever landing on the same value twice is a label rather than a magnitude.  A float
    column is never called one, however distinct it is, because the same shape is what an
    ordinary reading looks like.
    """
    rows = info.get("rows")
    distinct = info.get("n_distinct")
    if not isinstance(rows, int) or not isinstance(distinct, int):
        return None
    if rows < _IDENTIFIER_MIN_ROWS or distinct != rows:
        return None
    bins = ((info.get("fit") or {}) if isinstance(info.get("fit"), Mapping) else {}).get("bins")
    if not bins:
        return None
    edges = (bins[0].get("min"), bins[-1].get("max"))
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in edges):
        return None
    return Finding(
        factor=name,
        category="degenerate",
        severity="warning",
        level=info.get("level"),
        detail={"info": dict(info), "n_distinct": distinct, "rows": rows},
        remedy=f"{distinct} unique integer values across {rows} rows; column appears to be an identifier, exclude it",
        suggestion=Suggestion(policy={"exclude": [name]}, complete=True),
    )


def _floor_mass(record: Mapping[str, Any]) -> Iterator[Finding]:
    """Columns where a quarter of the rows or more sit on a single extreme value.

    Judged from the order statistics rather than from any cut, which is the point: a chart
    drawn at a factor's own bin count shows the reader the cut they were asked to evaluate,
    and a value can be a quarter of a column while landing inside one bin of it, invisible.
    ``min == p25`` says a quarter of the rows share the lowest value however anyone cut it.

    Reported as what it is rather than as what it probably means.  SeaDrone writes ``-1``
    where the drone recorded nothing, and that is a marker; a boat at rest genuinely reads a
    speed of zero, and that is a reading.  Nothing here can tell those apart — but both change
    what every statistic over the factor says, and both make a cut derived from the column
    describe the mass rather than the spread.  So the finding states the shape and leaves the
    reading to somebody who knows the data.

    The same value flooring several columns is carried as corroboration, because a convention
    shared across unrelated columns is much more likely to be a marker than a reading.
    """
    factors = _factors(record)
    extremes: dict[Any, list[str]] = {}
    for name, info in factors.items():
        for value in _floor_values(info):
            extremes.setdefault(value, []).append(name)
    for name, info in sorted(factors.items()):
        for value in _floor_values(info):
            others = sorted(f for f in extremes.get(value, ()) if f != name)
            shared = len(others) >= _SHARED_EXTREME_FACTORS
            yield Finding(
                factor=name,
                category="floor_mass",
                severity="warning",
                level=info.get("level"),
                detail={"value": value, "shared_with": others, "info": dict(info)},
                remedy=(
                    f"value {value!r} appears in >=25% of rows. "
                    + (
                        f"{len(others)} other factors share this extreme, which may indicate "
                        "a missing reading marker. Remap to `.nan` if appropriate"
                        if shared
                        else "If this is a sentinel marker and not a valid reading, remap to `.nan`"
                    )
                ),
                suggestion=Suggestion(
                    corrections=[{"kind": "remap", "factor": name, "rules": [{"match": value, "to": None}]}],
                    complete=False,
                ),
            )


def _floor_values(info: Mapping[str, Any]) -> list[Any]:
    """The extreme values a quarter of this column sits on, at either end.

    A constant column is not one of these — it has no spread for a mass to be a mass *of*, and
    ``degenerate`` already reports it as grouping nothing.
    """
    quantiles = ((info.get("distribution") or {}) or {}).get("quantiles")
    if not isinstance(quantiles, Mapping):
        return []
    try:
        low, p25, p75, high = (float(quantiles[k]) for k in ("0.0", "0.25", "0.75", "1.0"))
    except (KeyError, TypeError, ValueError):
        return []
    if low == high:
        return []
    found = []
    if low == p25:
        found.append(quantiles["0.0"])
    if high == p75:
        found.append(quantiles["1.0"])
    return found


def _is_sentinel(value: Any) -> bool:
    """Whether this value stands for an unrecorded reading."""
    return str(value).strip().lower() in _SENTINELS


def _is_number(text: str) -> bool:
    """Whether this reads as a finite number somebody could have written.

    Non-finite spellings are sentinels rather than values, and a trailing decimal point is
    residue rather than a number.  ``float`` accepts ``"1001."``, which is what a drop leaves
    behind when it cuts a suffix off an identifier — ``"1001.jpg"`` less ``"jpg"`` — and
    accepting it let the recognizer read a filename as a number.  That is the domain guess
    this module refuses everywhere else: the reading is for a number wearing decoration,
    ``"6,000"`` or ``"12 kg"``, not for an identifier that happens to have a numeric stem.
    Requiring the reading to be well-formed is what tells the two apart, and it costs nothing
    real — nobody writes a measurement with a bare point at the end.
    """
    if text.endswith("."):
        return False
    try:
        return math.isfinite(float(text))
    except (TypeError, ValueError):
        return False


def _numeric_drop(values: Sequence[str]) -> list[str] | None:
    """Substrings whose removal makes every value a number, or None where none does.

    Candidates are tried narrowest first, because the narrowest reading that works is the one
    least likely to corrupt a value it was not aimed at: dropping ``","`` cannot change
    ``"1k2"`` the way dropping every non-numeric character can.
    """
    texts = [str(v).strip() for v in values]
    if not texts:
        return None
    suffix = _common_suffix(texts)
    candidates: list[list[str]] = [[","], [",", " "], [" "]]
    if suffix:
        candidates += [[suffix], [suffix, ","], [suffix, ",", " "]]
    # Punctuation, whitespace and currency symbols only — never letters, which would turn an
    # identifier into a number (see the module-level note on `_LOOSE_DROPPABLE`).
    loose = sorted({c for t in texts for c in t if not (c.isdigit() or c in ".-+") and c in _LOOSE_DROPPABLE})
    if loose:
        candidates.append(loose)
    for drop in candidates:
        if all(_is_number(_apply_drop(t, drop)) for t in texts):
            return drop
    return None


def _apply_drop(text: str, drop: Sequence[str]) -> str:
    """Remove each substring in order, exactly as ``ParseValue`` does."""
    for token in drop:
        text = text.replace(token, "")
    return text.strip()


def _common_suffix(texts: Sequence[str]) -> str:
    """The longest trailing non-numeric run every value shares, or an empty string."""
    tails = [_numeric_tail(t) for t in texts]
    shared = tails[0]
    for tail in tails[1:]:
        while shared and not tail.endswith(shared):
            shared = shared[1:]
    return shared


def _numeric_tail(text: str) -> str:
    """Whatever trails this value's last digit or ``.`` — a unit, a symbol, nothing at all."""
    stripped = text
    while stripped and not (stripped[-1].isdigit() or stripped[-1] == "."):
        stripped = stripped[:-1]
    return text[len(stripped) :]


def _datetime_format(values: Sequence[str]) -> str | None | Literal[False]:
    """The one format reading every value, None for ISO, or False where several do.

    False rather than a guess: ``03/04/2021`` is two different days depending on the reading,
    and picking one would move every downstream statistic without saying it had.
    """
    texts = [str(v).strip() for v in values]
    if not texts:
        return False
    # ISO first, and reported as "no format": `ParseDateTime` infers ISO-8601 itself, and
    # pinning a pattern would mean enumerating every ISO spelling — SeaDrone writes
    # microseconds, which no `%H:%M:%S` matches — to say what the reader already knows.
    if all(_parses_iso(t) for t in texts):
        return None
    matched = [fmt for fmt in _DATE_FORMATS if all(_parses(t, fmt) for t in texts)]
    if not matched:
        return False
    # Formats differing only in field order over the same separator are ambiguous; ones that
    # agree on every value's reading are not.
    readings = {tuple(datetime.strptime(t, fmt) for t in texts) for fmt in matched}
    return matched[0] if len(readings) == 1 else False


def _parses_iso(text: str) -> bool:
    """Whether one value reads as ISO-8601, which needs no declared format."""
    try:
        datetime.fromisoformat(text)
    except ValueError:
        return False
    return True


def _parses(text: str, fmt: str) -> bool:
    """Whether one value reads under one format."""
    try:
        datetime.strptime(text, fmt)
    except ValueError:
        return False
    return True


def _looks_like_dates(values: Sequence[str]) -> bool:
    """Whether any single reading takes every value, ambiguously or not."""
    texts = [str(v).strip() for v in values]
    return all(_parses_iso(t) for t in texts) or any(all(_parses(t, fmt) for t in texts) for fmt in _DATE_FORMATS)


def _granularity(values: Sequence[str], fmt: str | None) -> str:
    """The coarsest absolute period still telling these values apart.

    ``fmt`` is None for an ISO reading, which is parsed the way `ParseDateTime` will parse it.
    """
    stamps = [
        datetime.fromisoformat(str(v).strip()) if fmt is None else datetime.strptime(str(v).strip(), fmt)
        for v in values
    ]
    for period in _ABSOLUTE_PERIODS:
        if len({_period_key(s, period) for s in stamps}) > 1:
            return period
    return _ABSOLUTE_PERIODS[-1]


def _period_key(stamp: datetime, period: str) -> tuple[int, ...]:
    """The bucket a moment falls in, for one absolute period."""
    iso = stamp.isocalendar()
    return {
        "year": (stamp.year,),
        "quarter": (stamp.year, (stamp.month - 1) // 3),
        "month": (stamp.year, stamp.month),
        "week": (iso[0], iso[1]),
        "day": (stamp.year, stamp.month, stamp.day),
        "hour": (stamp.year, stamp.month, stamp.day, stamp.hour),
    }[period]


def suggest(finding: Finding, *, default_bins: int = 10) -> Suggestion | None:
    """The config change that would address one finding, or None where flow cannot say.

    Only two categories produce one.  ``unreviewed`` needs a descriptor path nobody can
    invent, and ``degenerate`` needs a judgment about whether the factor is wanted at all —
    emitting either would put a value in the stanza that the user did not choose.

    Parameters
    ----------
    finding : Finding
        Any finding, hand-built or from :func:`find_issues`.
    default_bins : int, default 10
        Bin count to propose for an ``unbinned`` factor that carries no ``fit`` to read a
        populated count from — see :data:`find_issues`'s parameter of the same name, which
        this should agree with when both are called on the same record.
    """
    if finding.category == "unbinned":
        return _bin_suggestion(finding, default_bins)
    if finding.category != "unreadable" or not finding.repairable:
        return None
    return _correction_for(finding)


def _bin_suggestion(finding: Finding, default_bins: int) -> Suggestion | None:
    """A bin-count suggestion for an ``unbinned`` factor, or None where a cut is not the fix.

    Two shapes of ``unbinned`` finding reach here. One carries the whole factor entry in
    ``detail["info"]`` — a derived cut DataEval already produced, whose populated bin count
    :func:`_suggested_bins` reads. The other, a factor with no ``encoding`` key at all,
    carries only ``detail["type"]``: there is no fit to read a count from, so this falls
    back to ``default_bins`` — the one live path that parameter exists for. Either way, only
    a ``continuous`` factor gets a bin count; anything else needs a vocabulary, which no
    predicate here can invent.
    """
    info = finding.detail.get("info")
    if info is not None:
        count = _suggested_bins(info.get("fit"), default_bins)
    elif finding.detail.get("type") == "continuous":
        count = default_bins
    else:
        return None
    return Suggestion(policy={"continuous_factor_bins": {finding.factor: count}}, complete=True)


def _correction_for(finding: Finding) -> Suggestion | None:
    """Read the held-back values and propose how to read them.

    The recognizers are asked about the values that claim to *be* something, which is every
    value that is not a sentinel.  A sentinel says a reading was not recorded, so it is not
    evidence against how the recorded ones read — and letting one veto the reading is how
    SeaDrone's ``date_time``, 199 timestamps and one empty string, came back with no
    suggestion at all.  The correction is still emitted against the whole column: an
    unparsable value survives ``ParseDateTime`` as a level of its own rather than raising,
    so the unrecorded rows stay visible as the group they are.
    """
    text = list(finding.detail.get("distinct", {}).get("text", []))
    if not text:
        return None
    # Sentinels are dropped only for the recognizers' verdict, never from the enumeration
    # below, where every value still needs a rule of its own.
    readable = [value for value in text if not _is_sentinel(value)]

    datetime_correction = _datetime_correction(finding.factor, readable) if readable else None
    if datetime_correction is not None:
        return datetime_correction
    # An ambiguous date must refuse outright rather than fall through to a numeric reading:
    # `/` is punctuation, so `_numeric_drop` could otherwise turn "03/04/2021" into "03042021".
    if readable and _looks_like_dates(readable):
        return None

    drop = _numeric_drop(readable) if readable else None
    if drop is not None:
        return Suggestion(
            corrections=[{"kind": "parse_value", "factor": finding.factor, "drop": drop}],
            complete=True,
        )

    # A sampled column's values are near-unique by definition, so no mapping could cover it.
    if finding.detail.get("sampled"):
        return None

    # A sentinel is answered here rather than left for the user: it means "no reading", and
    # NaN is the target that says so.  `None` does not — `Remap` uses it as the *key*
    # catch-all, and as a target it is just a non-numeric value, so a column of numbers and
    # nulls stays mixed and the correction recovers nothing while claiming to be complete.
    rules = [{"match": value, "to": float("nan") if _is_sentinel(value) else None} for value in text]
    return Suggestion(
        corrections=[{"kind": "remap", "factor": finding.factor, "rules": rules}],
        complete=all(_is_sentinel(value) for value in text),
    )


def _datetime_correction(factor: str, text: Sequence[str]) -> Suggestion | None:
    """A ``parse_datetime`` correction where one reading takes every value, else None.

    ``format`` is omitted for an ISO reading, which is what ``fmt is None`` means — not an
    absent granularity.  The period is chosen from the values either way: reading them is
    what says which period tells them apart, and a format's absence says nothing about that.
    """
    fmt = _datetime_format(text)
    if fmt is False:
        return None
    entry: dict[str, Any] = {
        "kind": "parse_datetime",
        "factor": factor,
        "every": _granularity(text, fmt),
    }
    if fmt:
        entry["format"] = fmt
    return Suggestion(corrections=[entry], complete=True)


def to_policy_stanza(findings: Sequence[Finding]) -> dict[str, Any]:
    """Every suggestion merged into one metadata policy body.

    The stanza is the deliverable: three of the five categories are fixed by a policy edit
    rather than by a correction, and a user pastes one block rather than reconciling two.

    Incomplete suggestions are included.  A skeleton is what the user fills in, and
    withholding it would leave them to write the value list by hand from the report.
    Verification is where an incomplete suggestion is refused, not here.

    Parameters
    ----------
    findings : Sequence[Finding]
        As returned by :func:`find_issues`.

    Returns
    -------
    dict[str, Any]
        A body valid against
        :class:`~dataeval_flow.config.schemas.MetadataPolicyConfig` once ``name`` is added.
        Empty where nothing was suggested.
    """
    corrections: list[dict[str, Any]] = []
    policy: dict[str, Any] = {}
    for finding in findings:
        if finding.suggestion is None:
            continue
        corrections.extend(finding.suggestion.corrections)
        for key, value in finding.suggestion.policy.items():
            if isinstance(value, Mapping):
                policy.setdefault(key, {}).update(value)
            elif isinstance(value, list):
                # `exclude` is a list, and a second identifier must extend it rather than
                # replace the first.
                policy.setdefault(key, []).extend(v for v in value if v not in policy[key])
            else:
                policy[key] = value
    stanza: dict[str, Any] = {}
    if corrections:
        stanza["corrections"] = corrections
    stanza.update(policy)
    return stanza


def incomplete_factors(findings: Sequence[Finding]) -> set[str]:
    """Factor names whose suggestions are incomplete and require user input.

    These are factors whose suggested corrections have placeholders (e.g., ``to: null``)
    that the user must fill in before the correction can be applied.

    Parameters
    ----------
    findings : Sequence[Finding]
        As returned by :func:`find_issues`.

    Returns
    -------
    set[str]
        Names of factors with incomplete suggestions. Empty where all suggestions are
        complete or where there are no suggestions.
    """
    result = set()
    for finding in findings:
        if finding.suggestion is not None and not finding.suggestion.complete:
            result.add(finding.factor)
    return result


def render_stanza(stanza: Mapping[str, Any], *, name: str = "standard", incomplete: set[str] | None = None) -> str:
    """The stanza as a YAML block, ready to paste under a config's ``metadata:`` key.

    Rendered rather than dumped so that ``to: null`` keeps its spelling — a value the user
    has to replace should read as a hole, and most YAML writers emit it as an empty string
    or the bare word, neither of which reads as one.

    Parameters
    ----------
    stanza : Mapping[str, Any]
        The policy body as returned by :func:`to_policy_stanza`.
    name : str, default "standard"
        The policy name.
    incomplete : set[str] | None, default None
        Factor names whose corrections are incomplete and should be marked with ``# TODO``.
        When None (the default), no lines are marked — over-marking placeholders is worse
        than under-marking them.
    """
    if not stanza:
        return ""
    import yaml

    if not incomplete:
        # No incomplete factors, so no marking needed
        return yaml.safe_dump({"metadata": [{"name": name, **dict(stanza)}]}, sort_keys=False)

    # Use a unique marker token that cannot appear in user data (null byte is invalid in strings)
    marker = "\x00INCOMPLETE_NULL_MARKER\x00"

    # Build a copy with markers instead of nulls for incomplete factors
    marked_stanza: dict[str, Any] = {}
    for key, value in stanza.items():
        if key == "corrections" and isinstance(value, list):
            marked_corrections = []
            for correction in value:
                marked_correction = dict(correction)
                factor = correction.get("factor")

                # If this correction belongs to an incomplete factor and has rules with nulls, mark them
                if factor in incomplete and "rules" in correction:
                    marked_rules = []
                    for rule in correction["rules"]:
                        if rule.get("to") is None:
                            marked_rule = dict(rule)
                            marked_rule["to"] = marker
                            marked_rules.append(marked_rule)
                        else:
                            marked_rules.append(rule)
                    marked_correction["rules"] = marked_rules

                marked_corrections.append(marked_correction)
            marked_stanza[key] = marked_corrections
        else:
            marked_stanza[key] = value

    body = yaml.safe_dump({"metadata": [{"name": name, **marked_stanza}]}, sort_keys=False)

    # Replace the marker with the marked null
    # YAML escapes the null byte as \0, so we replace the escaped form
    # The marker will be in double-quoted form with YAML escaping: "\0INCOMPLETE_NULL_MARKER\0"
    return body.replace('"\\0INCOMPLETE_NULL_MARKER\\0"', "null        # TODO")
