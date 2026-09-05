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
    "unbinned": "Factors needing a cut or a vocabulary",
    "unreviewed": "Encodings nobody pinned",
    "degenerate": "Factors carrying no signal",
}

_ORDER = ("unreadable", "unbound_request", "unbinned", "unreviewed", "degenerate")


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
    return findings


def _finding_lines(finding: Finding, max_examples: int) -> list[str]:
    """One finding as report lines: what it is, its shape, and what to do."""
    head = f"[{finding.severity}] {finding.factor} [{', '.join(finding.reasons) or finding.category}"
    head += f" @ {finding.level}]" if finding.level else "]"
    lines = [head]
    counts = finding.detail.get("counts")
    if counts:
        lines.append(f"  {_render_ratio(counts)}")
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
