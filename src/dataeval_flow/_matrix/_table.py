"""The comparison a matrix's report opens with: a row per run, a column per finding (task-matrix spec §7.1)."""

__all__ = ["comparison_blocks", "comparison_table"]

from collections import Counter
from typing import TYPE_CHECKING, Any

from dataeval_flow._blocks import Block, Column, Paragraph, Section, Table
from dataeval_flow._blocks._text import _MARKERS
from dataeval_flow.config._schemas._matrix import show_value

if TYPE_CHECKING:
    from dataeval_flow._matrix._result import MatrixResult
    from dataeval_flow._result import Result
    from dataeval_flow.workflows._base import Finding

# A finding's column: the step that made it (none for a lone workflow's), its title, and which of that step's
# findings with that title it is.
_ColumnKey = tuple[str | None, str, int]


def _findings(result: "Result[Any, Any]") -> "dict[_ColumnKey, Finding]":
    from dataeval_flow.steps._result import ChainResult
    from dataeval_flow.workflows._result import WorkflowResult

    if isinstance(result, ChainResult):
        pairs: list[tuple[str | None, Finding]] = list(result.findings_by_step())
    elif isinstance(result, WorkflowResult):
        pairs = [(None, finding) for finding in result.findings]
    else:  # an evaluator makes determinations, not findings
        pairs = []
    seen: Counter[tuple[str | None, str]] = Counter()
    cells: dict[_ColumnKey, Finding] = {}
    for step, finding in pairs:
        seen[(step, finding.title)] += 1
        cells[(step, finding.title, seen[(step, finding.title)])] = finding
    return cells


def _headers(keys: "list[_ColumnKey]") -> "dict[_ColumnKey, str]":
    """The title; its step where columns of two steps share it; its number where one step repeats it."""
    steps: dict[str, set[str | None]] = {}
    for step, title, _ in keys:
        steps.setdefault(title, set()).add(step)
    headers: dict[_ColumnKey, str] = {}
    for step, title, occurrence in keys:
        header = f"{title} · {step}" if len(steps[title]) > 1 and step is not None else title
        headers[(step, title, occurrence)] = f"{header} ({occurrence})" if occurrence > 1 else header
    return headers


def _cell(finding: "Finding") -> str:
    marker = _MARKERS[finding.severity].strip()
    return f"{marker} {finding.brief}" if finding.brief else marker


def comparison_table(result: "MatrixResult") -> Table:
    """The table: each run's number, its value for each key (blank where its grid sets none), its health, and each
    finding's marker and brief (``—`` where it made none)."""
    per_run = [_findings(run.result) for run in result.runs]
    finding_keys = list(dict.fromkeys(key for cells in per_run for key in cells))
    headers = _headers(finding_keys)
    key_ids = {key: f"key{index}" for index, key in enumerate(result.keys)}
    finding_ids = {key: f"finding{index}" for index, key in enumerate(finding_keys)}
    columns = [
        Column(key="number", header="#", align="left"),
        *(Column(key=key_ids[key], header=key, align="left") for key in result.keys),
        Column(key="health", header="Health", align="left"),
        *(Column(key=finding_ids[key], header=headers[key], align="left") for key in finding_keys),
    ]
    rows: list[dict[str, Any]] = []
    for run, cells in zip(result.runs, per_run, strict=True):
        row: dict[str, Any] = {"number": run.number, "health": run.status}
        row.update({key_ids[key]: show_value(run.values[key]) if key in run.values else "" for key in result.keys})
        row.update({finding_ids[key]: _cell(cells[key]) if key in cells else "—" for key in finding_keys})
        rows.append(row)
    return Table(columns=columns, rows=rows)


def _health_line(result: "MatrixResult") -> str:
    health, count = result.health, len(result.runs)
    if health["status"] == "failed":
        failed = ", ".join(str(number) for number in health["failed_runs"])
        return f"Health: failed [!!] — run{'s' if len(health['failed_runs']) > 1 else ''} {failed} of {count} failed"
    if health["status"] == "warning":
        return f"Health: {health['warnings']} warning(s) [!!] across {count} runs — review flagged findings"
    return f"Health: ok — {count} runs, no warnings"


def comparison_blocks(result: "MatrixResult", *, detailed: bool) -> list[Block]:
    """The health line, the table, each failed run's error, and, when *detailed*, each run's report."""
    blocks: list[Block] = [Paragraph(text=_health_line(result)), comparison_table(result)]
    blocks.extend(
        Paragraph(text=f"Run {run.number} failed: {run.result.errors[0] if run.result.errors else 'failed'}")
        for run in result.runs
        if not run.result.success
    )
    if detailed:
        blocks.extend(
            Section(title=f"Run {run.number} · {run.label}", blocks=run.result._document(detailed=True).blocks)  # noqa: SLF001
            for run in result.runs
        )
    return blocks
