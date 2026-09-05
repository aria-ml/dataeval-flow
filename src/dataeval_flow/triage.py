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

__all__ = ["Category", "Finding", "Severity", "Suggestion", "find_issues"]

import difflib
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

Category = Literal["unreadable", "unbound_request", "unbinned", "unreviewed", "degenerate"]
Severity = Literal["blocking", "warning", "note"]

#: Severity order for report grouping.  Blocking first: it is what the run did less of than
#: the configuration asked for.
_SEVERITY_RANK: Mapping[str, int] = {"blocking": 0, "warning": 1, "note": 2}

#: Fewest bins a suggestion will propose.  One bin is not a cut.
_MIN_BINS = 2


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
                remedy="reached the evaluators as raw values; declare a bin count",
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
