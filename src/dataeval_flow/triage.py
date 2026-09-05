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

Category = Literal["unreadable", "unbound_request", "unbinned", "unreviewed", "degenerate"]
Severity = Literal["blocking", "warning", "note"]

#: Severity order for report grouping.  Blocking first: it is what the run did less of than
#: the configuration asked for.
_SEVERITY_RANK: Mapping[str, int] = {"blocking": 0, "warning": 1, "note": 2}

#: Fewest bins a suggestion will propose.  One bin is not a cut.
_MIN_BINS = 2

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
    ]
    for finding in findings:
        finding.suggestion = suggest(finding)
        if finding.category == "unreadable" and finding.repairable and finding.suggestion is None:
            values = finding.detail.get("distinct", {}).get("text", [])
            if values and _datetime_format(values) is False and _looks_like_dates(values):
                finding.remedy = "reads as a date two different ways; the format is ambiguous"
    return sorted(findings, key=lambda f: (_SEVERITY_RANK[f.severity], f.category, f.factor))


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
        return "measured, but has no single-column form however it is read"
    if "cardinality_over_budget" in reasons:
        return "names its rows rather than grouping them; give it a vocabulary"
    return "declare how the disagreeing values are to be read"


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
                f"no factor named {name!r}; did you mean {near[0]!r}?"
                if near
                else f"no factor named {name!r}; its cut was replaced by an auto-cut"
            ),
        )


def _no_encoding_remedy(factor_type: Any) -> str:
    """What a factor needs once it is known to have no encoding at all.

    A continuous factor is fixed by declaring a bin count; anything else takes a vocabulary
    instead, so guiding it toward "declare a bin count" would send it to the wrong stanza.
    """
    if factor_type == "continuous":
        return "reached the evaluators as raw values; declare a bin count"
    return "reached the evaluators as raw values; commit a descriptor or declare its levels"


def _encodings(record: Mapping[str, Any], default_bins: int) -> Iterator[Finding]:
    """Factors whose cut or vocabulary nobody pinned.

    The record carries a precomputed ``unreviewed`` list holding both halves.  This
    partitions by factor type instead of reading it, because the two halves take different
    remedies — reading the list directly would report every continuous factor twice.
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
        if info.get("type") == "continuous":
            count = _suggested_bins(info.get("fit"), default_bins)
            yield Finding(
                factor=name,
                category="unbinned",
                severity="warning",
                level=info.get("level"),
                detail={"info": dict(info)},
                remedy=f"cut from this draw; declare `continuous_factor_bins: {{{name}: {count}}}`",
                suggestion=Suggestion(policy={"continuous_factor_bins": {name: count}}, complete=True),
            )
        else:
            yield Finding(
                factor=name,
                category="unreviewed",
                severity="warning",
                level=info.get("level"),
                detail={"info": dict(info)},
                remedy="vocabulary from this draw; export a descriptor and reference it from `encoding:`",
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
    factor with one populated bucket — which groups nothing — and one losing most of its
    rows to no recorded value.
    """
    for name, info in sorted(_factors(record).items()):
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
                remedy="one populated bucket, so it groups nothing; exclude it or cut it differently",
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
                remedy=(
                    f"{share}% of rows are missing a value; they score as a group of their own "
                    "in every contingency table"
                ),
            )


def _is_sentinel(value: Any) -> bool:
    """Whether this value stands for an unrecorded reading."""
    return str(value).strip().lower() in _SENTINELS


def _is_number(text: str) -> bool:
    """Whether this reads as a finite number.  Non-finite spellings are sentinels, not values."""
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
    matched = [fmt for fmt in _DATE_FORMATS if all(_parses(t, fmt) for t in texts)]
    if not matched:
        return False
    # Formats differing only in field order over the same separator are ambiguous; ones that
    # agree on every value's reading are not.
    readings = {tuple(datetime.strptime(t, fmt) for t in texts) for fmt in matched}
    return matched[0] if len(readings) == 1 else False


def _parses(text: str, fmt: str) -> bool:
    """Whether one value reads under one format."""
    try:
        datetime.strptime(text, fmt)
    except ValueError:
        return False
    return True


def _looks_like_dates(values: Sequence[str]) -> bool:
    """Whether any single format reads every value, ambiguously or not."""
    return any(all(_parses(str(v).strip(), fmt) for v in values) for fmt in _DATE_FORMATS)


def _granularity(values: Sequence[str], fmt: str) -> str:
    """The coarsest absolute period still telling these values apart."""
    stamps = [datetime.strptime(str(v).strip(), fmt) for v in values]
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


def suggest(finding: Finding) -> Suggestion | None:
    """The config change that would address one finding, or None where flow cannot say.

    Only two categories produce one.  ``unreviewed`` needs a descriptor path nobody can
    invent, and ``degenerate`` needs a judgment about whether the factor is wanted at all —
    emitting either would put a value in the stanza that the user did not choose.
    """
    if finding.category == "unbinned":
        return finding.suggestion
    if finding.category != "unreadable" or not finding.repairable:
        return None
    return _correction_for(finding)


def _correction_for(finding: Finding) -> Suggestion | None:
    """Read the held-back values and propose how to read them."""
    text = list(finding.detail.get("distinct", {}).get("text", []))
    if not text:
        return None

    datetime_correction = _datetime_correction(finding.factor, text)
    if datetime_correction is not None:
        return datetime_correction
    # An ambiguous date must refuse outright rather than fall through to a numeric reading:
    # `/` is punctuation, so `_numeric_drop` could otherwise turn "03/04/2021" into "03042021".
    if _looks_like_dates(text):
        return None

    drop = _numeric_drop(text)
    if drop is not None:
        return Suggestion(
            corrections=[{"kind": "parse_value", "factor": finding.factor, "drop": drop}],
            complete=True,
        )

    # A sampled column's values are near-unique by definition, so no mapping could cover it.
    if finding.detail.get("sampled"):
        return None

    rules = [{"match": value, "to": None} for value in text]
    return Suggestion(
        corrections=[{"kind": "remap", "factor": finding.factor, "rules": rules}],
        complete=all(_is_sentinel(value) for value in text),
    )


def _datetime_correction(factor: str, text: Sequence[str]) -> Suggestion | None:
    """A ``parse_datetime`` correction where one format reads every value, else None."""
    fmt = _datetime_format(text)
    if fmt is False:
        return None
    every = _granularity(text, fmt) if fmt else "day"
    entry: dict[str, Any] = {"kind": "parse_datetime", "factor": factor, "every": every}
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

    body = yaml.safe_dump({"metadata": [{"name": name, **dict(stanza)}]}, sort_keys=False)
    if incomplete:
        lines = body.split("\n")
        marked_lines = []
        current_factor = None
        for line in lines:
            # Track which factor this line belongs to
            if "factor:" in line:
                current_factor = line.split("factor:")[1].strip()
            # Mark null lines only if the current factor is incomplete
            if current_factor and current_factor in incomplete and ": null" in line:
                line = line.replace(": null", ": null        # TODO")
            marked_lines.append(line)
        body = "\n".join(marked_lines)
    return body
