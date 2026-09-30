"""Finding builders for the metadata triage workflow."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from dataeval_flow._binning_report import distribution_blocks, proportion_block
from dataeval_flow._blocks import (
    Block,
    BulletList,
    Cell,
    Code,
    Column,
    Fields,
    ItemRef,
    Paragraph,
    Scalar,
    Section,
    Table,
)
from dataeval_flow._tables import group_cells, table_limits
from dataeval_flow._triage import TriageFinding
from dataeval_flow.workflows._base import Finding
from dataeval_flow.workflows.metadata_triage._outputs import MetadataTriageRawOutput

__all__ = ["Places", "build_findings", "minority_kind", "summarize"]

# Each problem value of a factor, most rows first: the value, how many rows hold it, and the first few of their items.
Places = Mapping[str, Sequence[tuple[str, int, Sequence[ItemRef]]]]

_TITLES: dict[str, str] = {
    "unreadable": "Unreadable factors",
    "unbound_request": "Unmatched bin requests",
    "floor_mass": "Columns dominated by one value",
    "degenerate": "Degenerate factors",
    "unbinned": "Unpinned continuous bins",
    "unreviewed": "Unpinned categorical vocabularies",
}

_ORDER = ("unreadable", "unbound_request", "floor_mass", "degenerate", "unbinned", "unreviewed")

# Categories whose remedy is the same sentence for every factor that has them, stated once
# here rather than repeated under each.
_COLLAPSED: dict[str, str] = {
    "unbinned": (
        "These bin counts were derived from this sample. Declaring them in configuration "
        "ensures consistent binning across runs."
    ),
    "unreviewed": (
        "These categorical vocabularies were derived from this sample. Export them with "
        "`dataeval-flow encoding` and reference the file in `encoding:`."
    ),
}


def _withdrawn_reason(factor: str, findings: list[TriageFinding]) -> str:
    """Why no cut is offered for this factor, named rather than pointed at."""
    other = {f.category for f in findings if f.factor == factor} - {"unbinned"}
    if "floor_mass" in other:
        return "no bin count suggested: single value appears in >=25% of rows"
    if "degenerate" in other:
        return "no bin count suggested: factor is an identifier"
    return "no bin count suggested"


def minority_kind(counts: Mapping[str, int]) -> str | None:
    """The kind fewer of a mixed column's values read as, whose values are its problem values; a tie takes text.

    ``None`` for a column whose values all read one way.
    """
    if len(counts) < 2:
        return None
    return min(counts, key=lambda kind: (counts[kind], kind != "text"))


def build_findings(raw: MetadataTriageRawOutput, max_examples: int, places: Places | None = None) -> list[Finding]:
    """One Finding per category that produced a finding.

    ``severity`` is ``"warning"`` only where the category holds a blocking finding, which is
    what makes ``WorkflowResult.health`` flag on exactly those: a blocking finding means the
    run did less than the configuration asked for without saying so. *places*, by factor, are
    where each of a mixed column's problem values sits, for its items to be pictured.
    """
    findings: list[Finding] = []
    for category in _ORDER:
        group = [f for f in raw.findings if f.category == category]
        if not group:
            continue
        blocking = any(f.severity == "blocking" for f in group)
        severity: Literal["ok", "info", "warning"] = "warning" if blocking else "info"
        blocks: list[Block]
        if category == "floor_mass":
            blocks = _floor_mass_blocks(group)
        elif shared := _COLLAPSED.get(category):
            blocks = [Paragraph(text=shared), *_collapsed_sections(group, list(raw.findings))]
        else:
            blocks = [_finding_section(finding, max_examples, (places or {}).get(finding.factor)) for finding in group]
        findings.append(
            Finding(severity=severity, title=_TITLES[category], brief=f"{len(group)} factors", blocks=blocks)
        )
    if raw.suggested_policy_yaml:
        findings.append(
            Finding(
                severity="info",
                title="Suggested policy",
                brief="add to configuration under `metadata:`",
                blocks=[Code(text=raw.suggested_policy_yaml.rstrip("\n"), language="yaml")],
            )
        )
    if raw.verification:
        findings.append(
            Finding(
                severity="info",
                title="Verified",
                brief=f"{sum(1 for v in raw.verification if v.recovered)} recovered",
                blocks=[Fields(items=[(v.factor, v.detail) for v in raw.verification])],
            )
        )
    elif raw.verification_error:
        # Distinct from the section above being absent for `verify: false` or nothing to
        # verify: a reader must be able to tell "verification blew up" from those two, and
        # an omitted section says nothing at all.
        findings.append(
            Finding(
                severity="warning",
                title="Verification failed",
                brief="not verified",
                blocks=[Paragraph(text=raw.verification_error)],
            )
        )
    return findings


def _floor_mass_blocks(group: list[TriageFinding]) -> list[Block]:
    """One section per shared value, not one per factor."""
    by_value: dict[str, list[TriageFinding]] = {}
    for finding in group:
        by_value.setdefault(repr(finding.detail.get("value")), []).append(finding)
    blocks: list[Block] = []
    for value, findings in sorted(by_value.items()):
        names = sorted(f.factor for f in findings)
        plural = "factors" if len(names) > 1 else "factor"
        if len(names) > 1:
            advice = (
                "A common extreme value across multiple factors may indicate a missing reading sentinel. "
                "Verify and remap to `.nan` if appropriate."
            )
        else:
            advice = "This may indicate a missing reading sentinel. Remap to `.nan` if appropriate."
        blocks.append(
            Section(
                title=value,
                brief=f"appears in >=25% of rows across {len(names)} {plural}",
                blocks=[
                    BulletList(items=names),
                    Paragraph(text=advice),
                    Paragraph(
                        text=(
                            "If this is a valid measurement, note the high concentration at this value. "
                            "No automatic bin count is suggested for skewed distributions."
                        )
                    ),
                ],
            )
        )
    return blocks


def _collapsed_sections(group: list[TriageFinding], everything: list[TriageFinding]) -> list[Block]:
    """Each factor with its own shape, under a remedy stated once for all of them.

    What repeats across these findings is the *sentence*, and printing it ten times reads as ten
    problems. What does not repeat is the distribution: it is the evidence for the bin count
    being proposed, and the reader is meant to disagree with a suggestion by looking at it. So
    the prose collapses and the charts do not.
    """
    sections: list[Block] = []
    for finding in sorted(group, key=lambda f: f.factor):
        policy = (finding.suggestion.policy if finding.suggestion else {}) or {}
        bins = (policy.get("continuous_factor_bins") or {}).get(finding.factor)
        if bins is not None:
            detail = f"declare {bins} bins"
        elif finding.suggestion is None and finding.category == "unbinned":
            detail = _withdrawn_reason(finding.factor, everything)
        else:
            detail = _bucket_count(finding)
        charts = distribution_blocks(finding.detail.get("info") or {})
        sections.append(Section(title=finding.factor, brief=detail, blocks=charts))
    return sections


def _bucket_count(finding: TriageFinding) -> str:
    """How many buckets this factor's encoding holds, as a plain phrase."""
    fit = (finding.detail.get("info") or {}).get("fit") or {}
    levels = fit.get("levels")
    if levels is not None:
        return f"{len(levels)} levels"
    return f"{len(fit.get('bins') or ())} bins"


def _finding_section(
    finding: TriageFinding, max_examples: int, places: Sequence[tuple[str, int, Sequence[ItemRef]]] | None = None
) -> Section:
    """One finding as a section: what it is, its shape, the values it read, where they are, and what to do."""
    brief = f"[{finding.severity}] {', '.join(finding.reasons) or finding.category}"
    if finding.level:
        brief += f" @ {finding.level}"
    blocks: list[Block] = []
    counts = finding.detail.get("counts")
    if counts:
        blocks.append(proportion_block(counts))
    # Not for an identifier: the chart would be the arbitrary cut this finding exists to
    # reject, drawn at full size and lending it the authority of a measurement.
    if "n_distinct" not in finding.detail:
        blocks.extend(distribution_blocks(finding.detail.get("info") or {}))
    if examples := _examples(finding, max_examples):
        blocks.append(Fields(items=examples))
    if places and (kind := minority_kind(finding.detail.get("counts") or {})):
        blocks.extend(_places_blocks(places, kind))
    blocks.append(Paragraph(text=f"-> {finding.remedy}"))
    return Section(
        title=finding.factor,
        brief=brief,
        severity="warning" if finding.severity == "blocking" else "info",
        blocks=blocks,
    )


def _places_blocks(places: Sequence[tuple[str, int, Sequence[ItemRef]]], kind: str) -> list[Block]:
    """Each problem value, most rows first: how many rows hold it, and up to eight of their items, named and pictured.

    At most ``result: max_rows``, 500 by default values, with a paragraph counting the rest.
    """
    limits = table_limits()
    rows: list[dict[str, Cell]] = []
    for value, count, refs in places[: limits.rows]:
        items, shown = group_cells(refs, total=count)
        rows.append({"value": value, "count": count, "items": items, "image": shown})
    columns = [
        Column(key="value", header="Value"),
        Column(key="count", header="Count"),
        Column(key="items", header="Items", align="left"),
        Column(key="image", kind="image"),
    ]
    blocks: list[Block] = [
        Paragraph(text=f"Where the values that read as {kind} are:"),
        Table(columns=columns, rows=rows, preview=limits.preview),
    ]
    if limits.rows is not None and len(places) > limits.rows:
        blocks.append(
            Paragraph(text=f"{len(places):,} values read as {kind}; the {limits.rows:,} on the most rows are listed.")
        )
    return blocks


def _examples(finding: TriageFinding, max_examples: int) -> list[tuple[str, Scalar]]:
    """The values a repair has to be written against, truncated for reading only."""
    items: list[tuple[str, Scalar]] = []
    for kind, values in (finding.detail.get("distinct") or {}).items():
        if not values:
            continue
        shown = [repr(v) for v in values[:max_examples]]
        more = f" (+{len(values) - max_examples} more)" if len(values) > max_examples else ""
        items.append((f"{kind} reads", f"{', '.join(shown)}{more}"))
    return items


def summarize(raw: MetadataTriageRawOutput) -> dict[str, Any]:
    """Counts by category and by severity, for the raw outputs."""
    counts: dict[str, Any] = {}
    for finding in raw.findings:
        counts[finding.category] = counts.get(finding.category, 0) + 1
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    return counts
