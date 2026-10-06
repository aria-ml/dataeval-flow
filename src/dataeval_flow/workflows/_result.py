"""How a finding reads in a report: its detail section, and its label where its group is not shown."""

__all__ = ["element_key", "finding_section", "summary_label"]

from dataeval_flow._blocks import Block, Paragraph, Section
from dataeval_flow.workflows._base import Finding


def finding_section(finding: Finding) -> Section:
    """The finding's detail section: its title, brief and severity, the description as the lede, then its blocks."""
    lede: list[Block] = [Paragraph(text=finding.description)] if finding.description else []
    return Section(
        title=finding.title,
        brief=finding.brief or None,
        severity=finding.severity,
        blocks=[*lede, *finding.blocks],
    )


def element_key(finding: Finding) -> str | None:
    """The key of the list element a check judged where it ran once per element; ``None`` for any other finding."""
    step = finding.step or ""
    return step[step.index("[") + 1 : -1] if step.endswith("]") and "[" in step else None


def summary_label(finding: Finding) -> str:
    """A finding named where its group is not: its title, then the element it judged, "Duplicates [train]"."""
    key = element_key(finding)
    return finding.title if key is None else f"{finding.title} [{key}]"
