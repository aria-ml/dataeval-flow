"""Reportable builders for the metadata triage workflow."""

from typing import Any, Literal

from dataeval_flow.triage import Finding
from dataeval_flow.workflow._text_report import _render_distribution, _render_ratio
from dataeval_flow.workflow.base import Reportable
from dataeval_flow.workflows.metadata_triage.outputs import MetadataTriageRawOutputs

__all__ = ["build_findings", "summarize"]

_TITLES: dict[str, str] = {
    "unreadable": "Unreadable factors",
    "unbound_request": "Requests that bound nothing",
    "floor_mass": "Columns dominated by one extreme value",
    "degenerate": "Factors carrying no signal",
    "unbinned": "Cuts nobody pinned",
    "unreviewed": "Vocabularies nobody pinned",
}

_ORDER = ("unreadable", "unbound_request", "floor_mass", "degenerate", "unbinned", "unreviewed")

#: Categories whose remedy is the same sentence for every factor that has them, stated once
#: here rather than repeated under each.  Ten copies of one paragraph read as ten problems and
#: bury the findings that genuinely differ.  Only the prose is shared: each factor still shows
#: its own distribution, because that is the evidence for the count being proposed and the
#: reader is meant to be able to disagree with it.
_COLLAPSED: dict[str, str] = {
    "unbinned": (
        "Each cut below came from this sample, so it is not stable across draws. Declaring the "
        "counts changes nothing about the numbers above — they are the cuts that already ran — "
        "it holds the same cuts for the next sample."
    ),
    "unreviewed": (
        "Each vocabulary below was drawn from this sample. Export one with "
        "`dataeval-flow encoding` and reference the file from `encoding:` to hold it."
    ),
}


def _withdrawn_reason(factor: str, findings: list[Finding]) -> str:
    """Why no cut is offered for this factor, named rather than pointed at."""
    other = {f.category for f in findings if f.factor == factor} - {"unbinned"}
    if "floor_mass" in other:
        return "no cut — a quarter of its rows sit on one extreme (see above)"
    if "degenerate" in other:
        return "no cut — it names its rows rather than grouping them (see above)"
    return "no cut suggested"


def build_findings(raw: MetadataTriageRawOutputs, max_examples: int) -> list[Reportable]:
    """One Reportable per category that produced a finding.

    ``severity`` is ``"warning"`` only where the category holds a blocking finding, which is
    what makes ``WorkflowResult.health`` flag on exactly those: a blocking finding means the
    run did less than the configuration asked for without saying so.
    """
    findings: list[Reportable] = []
    for category in _ORDER:
        group = [f for f in raw.findings if f.category == category]
        if not group:
            continue
        blocking = any(f.severity == "blocking" for f in group)
        severity: Literal["ok", "info", "warning"] = "warning" if blocking else "info"
        lines: list[str] = []
        if category == "floor_mass":
            lines.extend(_floor_mass_lines(group))
        elif shared := _COLLAPSED.get(category):
            lines.extend([shared, ""])
            lines.extend(_collapsed_lines(group, list(raw.findings)))
        else:
            for finding in group:
                lines.extend(_finding_lines(finding, max_examples))
        findings.append(
            Reportable(
                report_type="key_value",
                severity=severity,
                title=_TITLES[category],
                data={"brief": f"{len(group)} factors", "detail_lines": lines},
            )
        )
    if raw.suggested_policy_yaml:
        findings.append(
            Reportable(
                report_type="key_value",
                severity="info",
                title="Suggested policy",
                data={
                    "brief": "paste under your config's `metadata:` key",
                    "detail_lines": raw.suggested_policy_yaml.splitlines(),
                },
            )
        )
    if raw.verification:
        findings.append(
            Reportable(
                report_type="key_value",
                severity="info",
                title="Verified",
                data={
                    "brief": f"{sum(1 for v in raw.verification if v.recovered)} recovered",
                    "detail_lines": [f"{v.factor}: {v.detail}" for v in raw.verification],
                },
            )
        )
    elif raw.verification_error:
        # Distinct from the section above being absent for `verify: false` or nothing to
        # verify: a reader must be able to tell "verification blew up" from those two, and
        # an omitted section says nothing at all.
        findings.append(
            Reportable(
                report_type="key_value",
                severity="warning",
                title="Verification failed",
                data={"brief": "not verified", "detail_lines": [raw.verification_error]},
            )
        )
    return findings


def _floor_mass_lines(group: list[Finding]) -> list[str]:
    """One block per shared value, not one per factor.

    A shared extreme is a single observation about several columns at once. Rendered per factor
    it becomes the same sentence once per column with the other names permuted through it —
    which reads as many problems, and buries the one thing worth knowing: that one value sits
    at the end of all of them.
    """
    by_value: dict[str, list[Finding]] = {}
    for finding in group:
        by_value.setdefault(repr(finding.detail.get("value")), []).append(finding)
    lines: list[str] = []
    for value, findings in sorted(by_value.items()):
        names = sorted(f.factor for f in findings)
        plural = "factors" if len(names) > 1 else "factor"
        lines.append(f"{value} is an extreme held by a quarter or more of {len(names)} {plural}:")
        lines.append(f"  {', '.join(names)}")
        lines.append("")
        if len(names) > 1:
            lines.append("One value at the end of several unrelated columns is usually a convention,")
            lines.append("and the convention is almost always 'not recorded'.")
        else:
            lines.append("It may be a marker or a genuine reading — nothing here can tell which.")
        lines.append("Either way a cut derived from this column describes the mass rather than the")
        lines.append("spread, which is why no bin count is suggested for it.")
        lines.append("")
        lines.append("If it is a marker, code it missing (`.nan`); if it is a reading, know that")
        lines.append("this factor is mostly that one value.")
        lines.append("")
    return lines


def _collapsed_lines(group: list[Finding], everything: list[Finding]) -> list[str]:
    """Each factor with its own shape, under a remedy stated once for all of them.

    What repeats across these findings is the *sentence*, and printing it ten times reads as ten
    problems. What does not repeat is the distribution: it is the evidence for the bin count
    being proposed, and the reader is meant to disagree with a suggestion by looking at it. So
    the prose collapses and the charts do not.
    """
    lines: list[str] = []
    for finding in sorted(group, key=lambda f: f.factor):
        policy = (finding.suggestion.policy if finding.suggestion else {}) or {}
        bins = (policy.get("continuous_factor_bins") or {}).get(finding.factor)
        if bins is not None:
            detail = f"declare {bins} bins"
        elif finding.suggestion is None and finding.category == "unbinned":
            detail = _withdrawn_reason(finding.factor, everything)
        else:
            detail = _bucket_count(finding)
        lines.append(f"{finding.factor} — {detail}")
        lines.extend(f"  {line}" for line in _render_distribution(finding.detail.get("info") or {}))
        lines.append("")
    return lines


def _bucket_count(finding: Finding) -> str:
    """How many buckets this factor's encoding holds, as a plain phrase."""
    fit = (finding.detail.get("info") or {}).get("fit") or {}
    levels = fit.get("levels")
    if levels is not None:
        return f"{len(levels)} levels"
    return f"{len(fit.get('bins') or ())} bins"


def _finding_lines(finding: Finding, max_examples: int) -> list[str]:
    """One finding as report lines: what it is, its shape, and what to do."""
    head = f"[{finding.severity}] {finding.factor} [{', '.join(finding.reasons) or finding.category}"
    head += f" @ {finding.level}]" if finding.level else "]"
    lines = [head]
    counts = finding.detail.get("counts")
    if counts:
        lines.append(f"  {_render_ratio(counts)}")
    # Not for an identifier: the chart would be the arbitrary cut this finding exists to
    # reject, drawn at full size and lending it the authority of a measurement.
    if "n_distinct" not in finding.detail:
        lines.extend(f"  {line}" for line in _render_distribution(finding.detail.get("info") or {}))
    lines.extend(f"  {line}" for line in _example_lines(finding, max_examples))
    lines.append(f"  -> {finding.remedy}")
    lines.append("")
    return lines


def _example_lines(finding: Finding, max_examples: int) -> list[str]:
    """The values a repair has to be written against, truncated for reading only."""
    lines: list[str] = []
    for kind, values in (finding.detail.get("distinct") or {}).items():
        if not values:
            continue
        shown = [repr(v) for v in values[:max_examples]]
        more = f" (+{len(values) - max_examples} more)" if len(values) > max_examples else ""
        lines.append(f"{kind} reads: {', '.join(shown)}{more}")
    return lines


def summarize(raw: MetadataTriageRawOutputs) -> dict[str, Any]:
    """Counts by category and by severity, for the raw outputs."""
    counts: dict[str, Any] = {}
    for finding in raw.findings:
        counts[finding.category] = counts.get(finding.category, 0) + 1
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    return counts
