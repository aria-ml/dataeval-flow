"""How factors were typed and binned, and how their rows fell, as report blocks.

Binning decides what every evaluator reads: a continuous factor reaches balance and diversity as
interval codes, not as the values measured.  So the report states it rather than leaving it to the
envelope alone.
"""

__all__ = ["binning_blocks", "distribution_blocks", "proportion_block"]

from collections.abc import Mapping, Sequence
from typing import Any

from dataeval_flow._binning import divergent_factors
from dataeval_flow._blocks import (
    Block,
    BulletList,
    Column,
    Distribution,
    Fields,
    Paragraph,
    Proportion,
    Quantiles,
    Scalar,
    Section,
    Table,
)
from dataeval_flow._blocks._draw import fmt_num

# Above this many bins or levels, per-entry detail gives way to a summary.  A factor holding
# one category per sample would otherwise list one row per sample, burying every factor
# that carries real signal.
_MAX_ENUMERATED = 12

_PROVENANCE = {
    "edges": "edges declared",
    "count": "count declared",
    "declared": "declared",
    "accepted": "accepted",
    "derived": "derived",
}

_FACTOR_COLUMNS = [
    Column(key="factor", header="factor"),
    Column(key="lvl", header="lvl", align="left"),
    Column(key="bins", header="bins"),
    Column(key="shape", header="shape (low → high)", kind="sparkline"),
    Column(key="n/bin", header="n/bin"),
    Column(key="range", header="range", align="left"),
]


def _population_summary(counts: Sequence[int]) -> str:
    """How many samples landed in each bucket, as a spread or a single number."""
    low, high = min(counts), max(counts)
    return f"n={low}" if low == high else f"n={low}–{high}"


def _bin_names(info: Mapping[str, Any], declared: int) -> dict[int, str]:
    """Name each declared bin, preferring what DataEval called it.

    The names travel in the record because DataEval chooses them: one precision across a
    whole cut, picked by magnitude and by distinctness, so that no two bins print the same
    label.  Rendering them here would collapse distinct bins onto one label (seven
    epoch-millisecond bins onto three) and disagree with the labels
    ``ParityOutput.insufficient_data`` reports for the same bin.

    Falls back to the bare code where the record carries no name, as a release without the
    accessor leaves it.
    """
    names = info.get("names") or {}
    return {code: str(names.get(str(code), code)) for code in range(1, declared + 1)}


def _how_encoded(encoding: Mapping[str, Any]) -> str:
    """Who chose this encoding, and how it was placed.

    ``provenance`` is the field a reviewer audits: a descriptor still carrying ``derived``
    entries is one nobody has finished reviewing.
    """
    how = _PROVENANCE.get(encoding.get("provenance", ""), str(encoding.get("provenance")))
    method = encoding.get("method")
    return f"{how} ({method})" if method and how == "derived" else how


# -- The factor table ---------------------------------------------------------------------------


def _declared_counts(info: Mapping[str, Any]) -> list[int]:
    """How many rows fell in each declared bucket, empties reinstated as zero.

    ``fit["bins"]`` carries only the populated codes, so a cut with gaps would otherwise
    report an occupancy range that skipped its own empty bins, the very thing a reader
    checks a bin count against.
    """
    fit = info.get("fit") or {}
    if fit.get("levels") is not None:
        return [int(entry.get("count") or 0) for entry in fit["levels"]]
    declared = max(len((info.get("encoding") or {}).get("edges") or ()) - 1, 0)
    populated = {b["code"]: int(b.get("count") or 0) for b in fit.get("bins") or []}
    return [populated.get(code, 0) for code in range(1, declared + 1)]


def _observed_span(info: Mapping[str, Any]) -> str:
    """The extremes the values actually reached, as ``low – high``.

    Read from the recorded order statistics where there are any, because those describe the
    column rather than the cut laid over it, and fall back to the occupied bins otherwise.
    """
    quantiles = (info.get("distribution") or {}).get("quantiles") or {}
    try:
        low, high = float(quantiles["0.0"]), float(quantiles["1.0"])
    except (KeyError, TypeError, ValueError):
        bins = (info.get("fit") or {}).get("bins") or []
        if not bins:
            return ""
        low, high = min(b["min"] for b in bins), max(b["max"] for b in bins)
    return f"{fmt_num(low)} – {fmt_num(high)}"


def _declared_edges(info: Mapping[str, Any]) -> str:
    """The cut points somebody wrote down, where the row cannot imply them.

    A declared bin *count* places edges uniformly across the observed span, so ``bins`` and
    ``range`` together recover every one of them and printing them again says nothing.  A
    verbatim edge list is arbitrary by construction, which is what declaring one is for,
    and nothing else in the row says where it fell.
    """
    encoding = info.get("encoding") or {}
    if encoding.get("provenance") != "edges":
        return ""
    interior = [
        fmt_num(edge)
        for edge in encoding.get("edges") or ()
        if isinstance(edge, (int, float)) and edge not in (float("inf"), float("-inf"))
    ]
    return ", ".join(interior)


def _factor_table(record: Mapping[str, Any]) -> list[Block]:
    """Every encoded factor as one row, then the cut points any of them declared.

    A row apiece gives every factor the same room, and the shape column is drawn from the
    *recorded* histogram rather than from the factor's own cut: the chart exists to judge
    that cut, so drawing it through the cut would be circular.  ``bins`` counts levels for a
    digitized factor; the detailed breakdown names them.
    """
    rows: list[dict[str, Any]] = []
    edges: list[tuple[str, Scalar]] = []
    for name, info in (record.get("factors") or {}).items():
        if not info.get("encoding"):
            continue
        counts = _declared_counts(info)
        level = str(info.get("level", "?"))
        rows.append(
            {
                "factor": name,
                "lvl": "inst" if level == "instance" else level,
                "bins": len(counts),
                "shape": [float(c) for c in (info.get("distribution") or {}).get("histogram") or counts],
                "n/bin": f"{min(counts)}–{max(counts)}" if counts else "",
                "range": _observed_span(info),
            }
        )
        if declared := _declared_edges(info):
            edges.append((f"{name} edges", declared))
    if not rows:
        return []
    blocks: list[Block] = [Table(columns=_FACTOR_COLUMNS, rows=rows)]
    if edges:
        blocks.append(Fields(items=edges))
    return blocks


# -- Per-factor detail --------------------------------------------------------------------------


def _binned_detail(head: str, info: Mapping[str, Any], encoding: Mapping[str, Any], fit: Mapping[str, Any]) -> Section:
    """A cut, its provenance, and how this run's rows fell into it."""
    names = _bin_names(info, max(len(encoding.get("edges") or ()) - 1, 0))
    populated = {b["code"]: b for b in fit.get("bins") or []}
    empty = set(fit.get("empty") or ())
    brief = f"{len(names)} bins, {_how_encoded(encoding)}"
    if empty:
        brief += f", {len(empty)} empty"
    if len(names) > _MAX_ENUMERATED:
        spans = list(populated.values())
        if spans:
            low, high = min(b["min"] for b in spans), max(b["max"] for b in spans)
            brief += f", [{fmt_num(low)}, {fmt_num(high)}] occupied"
        return Section(title=head, brief=brief)
    rows: list[dict[str, Any]] = []
    for code in sorted(names):
        bucket = populated.get(code)
        if bucket is None:
            rows.append({"bin": names[code], "n": 0, "occupied": "empty"})
        else:
            span = f"[{fmt_num(bucket['min'])}, {fmt_num(bucket['max'])}]"
            rows.append({"bin": names[code], "n": bucket["count"], "occupied": span})
    for label, key in (("below range", "below_range"), ("above range", "above_range"), ("missing", "missing")):
        if fit.get(key):
            rows.append({"bin": label, "n": fit[key], "occupied": None})
    columns = [
        Column(key="bin", header="bin"),
        Column(key="n", header="n"),
        Column(key="occupied", header="occupied", align="left"),
    ]
    return Section(title=head, brief=brief, blocks=[Table(columns=columns, rows=rows)])


def _digitized_detail(head: str, encoding: Mapping[str, Any], fit: Mapping[str, Any]) -> Section:
    """A vocabulary, its provenance, and how this run's rows fell across it."""
    levels = fit.get("levels") or []
    brief = f"{len(levels)} levels, {_how_encoded(encoding)}"
    if len(levels) > _MAX_ENUMERATED:
        counts = [entry["count"] for entry in levels]
        # One category per sample means an identifier column, not a grouping: it carries
        # nothing for balance or diversity, and saying so is all a reader needs from it.
        if counts and max(counts) == 1:
            return Section(title=head, brief=f"{brief} (one per sample)")
        return Section(title=head, brief=f"{brief}, {_population_summary(counts)} per level" if counts else brief)
    # Sorted by value rather than by code: a vocabulary grows append-only, so a level added
    # after the first structuring carries a code out of sort order, and listing by code
    # would read as scrambled.
    rows = [
        {"value": str(entry["value"]), "code": entry["code"], "n": entry["count"]}
        for entry in sorted(levels, key=lambda e: (e["value"] is None, str(e["value"])))
    ]
    columns = [Column(key="value", header="value"), Column(key="code", header="code"), Column(key="n", header="n")]
    return Section(title=head, brief=brief, blocks=[Table(columns=columns, rows=rows)])


def _factor_detail(name: str, info: Mapping[str, Any]) -> Section:
    """One factor: what it is, how it was encoded, and how that encoding fits.

    A factor with more buckets than ``_MAX_ENUMERATED`` reports what the reader can act on,
    how many and how they were populated, instead of listing them.  The full map stays in the
    envelope either way.
    """
    head = f"{name} [{info.get('type', '?')} @ {info.get('level', '?')}]"
    encoding, fit = info.get("encoding"), info.get("fit")
    if not encoding:
        return Section(title=head, brief="not encoded")
    if fit is None:
        return Section(title=head, brief=_how_encoded(encoding))
    if encoding.get("kind") == "bins":
        return _binned_detail(head, info, encoding, fit)
    return _digitized_detail(head, encoding, fit)


# -- One record, and the section ----------------------------------------------------------------


def _review_state(record: Mapping[str, Any]) -> list[tuple[str, Scalar]]:
    """Which parts of this encoding are reviewed and which are derived.

    Requiring declared cuts is meant to make engineers encode domain knowledge they would
    otherwise skip: *below 10 lux is night*, *over 500 px is a large object*.  That only
    works if the un-reviewed state is visible: a factor reading `derived` was cut by
    DataEval from this sample, its bin count moves with the draw, and nothing about it is
    a claim anyone made.
    """
    factors = record.get("factors") or {}
    encoded = [name for name, info in factors.items() if info.get("encoding")]
    unreviewed = record.get("unreviewed")
    if not encoded or unreviewed is None:  # None: a record written before this was tracked
        return []
    if not unreviewed:
        return [("Policy", f"all {len(encoded)} factors declared or reviewed")]
    return [
        (
            "Policy",
            (
                f"{len(unreviewed)} of {len(encoded)} factors still derived — nobody has reviewed them\n"
                f"({', '.join(unreviewed)})"
            ),
        )
    ]


def _dropped(dropped: Mapping[str, Sequence[str]]) -> list[Block]:
    """Columns that never became factors, grouped so each reason is stated once."""
    if not dropped:
        return []
    by_reason: dict[str, list[str]] = {}
    for name, reasons in dropped.items():
        by_reason.setdefault(", ".join(reasons), []).append(name)
    lines = [f"{reason} — {', '.join(names)}" for reason, names in sorted(by_reason.items())]
    return [Fields(items=[("Dropped", "\n".join(lines))])]


def _record_blocks(record: Mapping[str, Any], *, detailed: bool) -> list[Block]:
    """One dataset's (or split's) encoding, its factor table, and what was dropped."""
    items: list[tuple[str, Scalar]] = []
    if record.get("encoding_digest"):
        items.append(("Encoding", record["encoding_digest"]))
    if record.get("auto_bin_method"):
        items.append(("Auto-bin method", record["auto_bin_method"]))
    if record.get("excluded"):
        items.append(("Excluded", ", ".join(record["excluded"])))
    if record.get("unmatched_bin_requests"):
        items.append(("Unmatched bins", ", ".join(record["unmatched_bin_requests"])))
    items.extend(_review_state(record))
    blocks: list[Block] = [Fields(items=items)] if items else []
    blocks.extend(_factor_table(record))
    blocks.extend(_dropped(record.get("dropped") or {}))
    factors = record.get("factors") or {}
    if detailed and factors:
        # The table is what the section is read for; the breakdown is what it is audited
        # from.  Keeping both leaves the bin edges on the page for anyone who asked for
        # them, without every reader paying a dozen lines a factor to learn a cut is fine.
        details = [_factor_detail(name, info) for name, info in factors.items()]
        blocks.append(Section(title="Per-factor detail", blocks=list(details)))
    return blocks


def _split_comparability(per_split: Mapping[str, Any]) -> list[Block]:
    """Whether the splits were read under one encoding, where there is more than one.

    Splits encoded independently land on different cuts for the same factor, and their
    per-factor statistics sit side by side in one report under different alphabets that a
    reader cannot tell not to compare.  The verdict comes from
    :func:`dataeval_flow._binning.divergent_factors` rather than a rule of its own, so this
    line, the envelope's ``encoding_digest`` and the descriptor writer give the same answer.
    """
    if len(per_split) < 2 or not any((record.get("factors") or {}) for record in per_split.values()):
        return []
    divergent = divergent_factors(per_split.values())
    if not divergent:
        # Comparable is not the same as identical: an append-only vocabulary that grew
        # leaves every shared code meaning what it meant, but the digests differ and
        # `encoding_digest` is then None.  Saying "share one encoding" over that would
        # contradict the envelope.
        if len({record.get("encoding_digest") for record in per_split.values()}) == 1:
            return [Paragraph(text="Splits share one encoding — factor statistics are comparable across them.")]
        return [
            Paragraph(
                text="Splits encode every shared code the same way — factor statistics are comparable across "
                "them. The digests differ only because a vocabulary grew, which appends."
            )
        ]
    which = "that factor is" if len(divergent) == 1 else "those factors are"
    return [
        Paragraph(
            text=f"Splits encode {', '.join(divergent)} differently — statistics on {which} NOT comparable "
            "across them. Set a metadata policy's `reference_split`, or apply a committed `encoding`, so every "
            "split is cut the same way."
        )
    ]


def binning_blocks(
    binning: Mapping[str, Any] | None, diagnostics: Sequence[str] = (), *, detailed: bool = False
) -> list[Block]:
    """The METADATA FACTORS section: how factors were typed and binned, and any library diagnostics.

    Empty when the run recorded neither.  A multi-split run shows one subsection per split;
    their factor tables share columns because they sit in one top-level section.
    """
    if not binning and not diagnostics:
        return []
    blocks: list[Block] = []
    if binning and "per_split" in binning:
        per_split = binning["per_split"]
        blocks.extend(
            Section(title=f"[{name}]", blocks=_record_blocks(record, detailed=detailed))
            for name, record in per_split.items()
        )
        blocks.extend(_split_comparability(per_split))
    elif binning:
        blocks.extend(_record_blocks(binning, detailed=detailed))
    if diagnostics:
        blocks.append(Section(title="Diagnostics", blocks=[BulletList(items=list(diagnostics))]))
    return [Section(title="METADATA FACTORS", blocks=blocks)]


# -- Distribution charts: what makes a bin count arguable rather than arbitrary ------------------


def _bucket_counts(info: Mapping[str, Any]) -> list[tuple[str, int]]:
    """Every bucket this factor's fit describes, as (label, count), in code order.

    Empty bins are reinstated from ``fit["empty"]``: ``fit["bins"]`` carries only the
    populated ones, and a chart that omitted the gaps would hide the very thing a bin count
    is chosen from.
    """
    fit = info.get("fit") or {}
    if fit.get("levels") is not None:
        return [(str(e["value"]), int(e.get("count") or 0)) for e in fit["levels"]]
    encoding = info.get("encoding") or {}
    names = _bin_names(info, max(len(encoding.get("edges") or ()) - 1, 0))
    populated = {b["code"]: int(b.get("count") or 0) for b in fit.get("bins") or []}
    rows = [(names.get(code, str(code)), populated.get(code, 0)) for code in sorted(names)]
    for label, key in (("below range", "below_range"), ("above range", "above_range"), ("missing", "missing")):
        if fit.get(key):
            rows.append((label, int(fit[key])))
    return rows


def _shape(info: Mapping[str, Any]) -> Distribution | None:
    """A cut numeric factor's recorded distribution, drawn without reference to any cut.

    Only for a factor that was *cut*.  A digitized one's vocabulary is the values themselves
    rather than an imposed partition, so showing it directly is not circular and reads far
    better than order statistics over category codes.
    """
    if (info.get("encoding") or {}).get("kind") != "bins":
        return None
    dist = info.get("distribution") or {}
    hist, quantiles = dist.get("histogram"), dist.get("quantiles")
    if not hist or not isinstance(quantiles, Mapping) or not max(hist):
        return None
    try:
        low, q1, med, q3, high = (float(quantiles[k]) for k in ("0.0", "0.25", "0.5", "0.75", "1.0"))
    except (KeyError, TypeError, ValueError):
        return None
    return Distribution(histogram=list(hist), quantiles=Quantiles(low=low, q1=q1, median=med, q3=q3, high=high))


def distribution_blocks(info: Mapping[str, Any]) -> list[Block]:
    """How this factor's rows fell, as a chart.

    A cut numeric factor draws its recorded shape as a box plot.  Anything else draws a
    sparkline, since the shape is the argument for a bin count and it costs one line, then
    one bar per bucket up to ``_MAX_ENUMERATED``, above which the sparkline carries the
    shape on its own.
    """
    shape = _shape(info)
    if shape is not None:
        return [shape]
    rows = _bucket_counts(info)
    counts = [count for _, count in rows]
    if not max(counts, default=0):
        return []
    blocks: list[Block] = [Distribution(histogram=counts)]
    if len(rows) > _MAX_ENUMERATED:
        return blocks
    columns = [
        Column(key="label", align="right"),
        Column(key="count", kind="bar"),
        Column(key="count", format="{:>5}"),
        Column(key="note", align="left"),
    ]
    table_rows: list[dict[str, Any]] = [
        {"label": label, "count": count, "note": None if count else "empty"} for label, count in rows
    ]
    return [*blocks, Table(columns=columns, rows=table_rows)]


def proportion_block(counts: Mapping[str, int]) -> Proportion:
    """How a held-back column's rows split between kinds, the numeric share first."""
    return Proportion(parts=sorted(counts.items(), key=lambda part: (part[0] != "numeric", part[0])))
