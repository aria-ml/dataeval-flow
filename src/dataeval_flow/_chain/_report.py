"""A chain's report: a summary, then one section per step, headed by where its Datasets came from."""

__all__ = ["chain_blocks", "lineage_line", "step_heading"]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from dataeval_flow._blocks import Block, Fields, Paragraph, Section
from dataeval_flow._result import LineageRecord, failure_section
from dataeval_flow._step_title import step_title

if TYPE_CHECKING:
    from dataeval_flow.steps._result import ChainResult, StepResult


def lineage_line(address: str, lineage: Sequence[LineageRecord]) -> str:
    """``address`` walked back through the Datasets it was made from, to the source: "`few` ← `k` ← `a` (src)".

    A list is walked back through its elements, which lineage records one by one, with each element's key dropped:
    "`kept` ← `cams` (cam1, cam2)" names the source of every element that walks back the way the first one does.
    """
    records = {record.name: record for record in lineage}
    elements = [name for name in records if name.startswith(f"{address}[") and name.endswith("]")]
    if address in records or not elements:
        parts, source = _walk(address, records)
        return _line(parts, [source] if source else [])
    walks = [_walk(name, records) for name in elements]
    keyed = [
        ([part.removesuffix(f"[{name[len(address) + 1 : -1]}]") for part in parts], source)
        for name, (parts, source) in zip(elements, walks, strict=True)
    ]
    head = keyed[0][0]
    sources = [source for parts, source in keyed if parts == head and source]
    return _line(head, list(dict.fromkeys(sources)))


def _walk(address: str, records: Mapping[str, LineageRecord]) -> tuple[list[str], str | None]:
    """The addresses from `address` back through each first input to a chain input, and that input's source."""
    parts: list[str] = []
    current: str | None = address
    while current is not None and current not in parts:
        record = records.get(current)
        parts.append(current)
        current = record.inputs[0] if record is not None and record.inputs else None
    last = records.get(parts[-1])
    return parts, last.source if last is not None else None


def _line(parts: Sequence[str], sources: Sequence[str]) -> str:
    text = " ← ".join(f"`{part}`" for part in parts)
    return f"{text} ({', '.join(sources)})" if sources else text


def step_heading(record: "StepResult") -> str:
    """A step's heading: its type's friendly title, with its name set beside it where that differs from the type id:
    "Outliers" for step `outliers`, "Duplicates · dupes" for step `dupes`."""
    title = step_title(record.kind, record.type)
    return title if record.name == record.type else f"{title} · {record.name}"


def chain_blocks(result: "ChainResult", *, detailed: bool) -> list[Block]:
    """The summary, then one section per step, in run order, as top-level sections alongside Configuration."""
    counts = {status: sum(r.status == status for r in result.steps.values()) for status in ("ok", "failed", "skipped")}
    steps: list[Block] = [
        Section(
            title=step_heading(record),
            brief=None if record.status == "ok" else record.status,
            blocks=_step(record, result, detailed=detailed),
        )
        for record in result.steps.values()
    ]
    return [
        Fields(
            items=[
                ("Steps", len(result.steps)),
                ("Ran", counts["ok"]),
                ("Failed", counts["failed"]),
                ("Skipped", counts["skipped"]),
            ]
        ),
        *result._summary_blocks(),  # noqa: SLF001 - a chain's report reuses a workflow's summary
        *steps,
    ]


def _step(record: "StepResult", result: "ChainResult", *, detailed: bool) -> list[Block]:
    blocks: list[Block] = []
    if record.inputs:
        blocks.append(
            Paragraph(
                text="On " + ", ".join(lineage_line(address, result.metadata.lineage) for address in record.inputs)
            )
        )
    if record.reason is not None:
        blocks.append(Paragraph(text=f"Skipped: {record.reason}" if record.status == "skipped" else record.reason))
    if record.status == "failed" and record.errors:
        blocks.append(failure_section(record.errors))
    if record.elements is not None:
        for key, element in record.elements.items():
            blocks.append(
                Section(
                    title=f"[{key}]",
                    brief=None if element.status == "ok" else element.status,
                    blocks=_step(element, result, detailed=detailed),
                )
            )
        return blocks
    if record.status != "ok":
        return blocks
    blocks.extend(_output_blocks(record, detailed=detailed))
    return blocks


def _output_blocks(record: "StepResult", *, detailed: bool) -> list[Block]:
    """What a completed step made: a check's findings, an evaluator's or workflow's report, a transform's section."""
    from dataeval_flow.evaluators._result import EvaluatorResult
    from dataeval_flow.steps._registry import TRANSFORMS
    from dataeval_flow.workflows._result import WorkflowResult, finding_section

    if record.kind == "check":
        return [finding_section(finding) for finding in record.output or []]
    step_result = record.result
    if isinstance(step_result, EvaluatorResult):
        return list(step_result._report_output(detailed=detailed))  # noqa: SLF001 - a step's own report has no public accessor
    if isinstance(step_result, WorkflowResult):
        return list(step_result._report_output(detailed=detailed))  # noqa: SLF001 - a step's own report has no public accessor
    if record.kind == "transform":
        section = TRANSFORMS.get(record.type)().section(record) if record.type in TRANSFORMS.names() else []
        return list(section or [Fields(items=_dataset_fields(record.summary))])
    return []


def _dataset_fields(summary: object) -> list[tuple[str, str | int | float | bool | None]]:
    if isinstance(summary, dict) and "items" in summary:
        return [("Items", summary["items"]), ("Digest", summary.get("digest"))]
    if isinstance(summary, dict):
        return [
            (str(key), value["items"] if isinstance(value, dict) and "items" in value else str(value))
            for key, value in summary.items()
        ]
    return [("Output", str(summary))]
