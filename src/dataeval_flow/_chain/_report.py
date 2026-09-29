"""A chain's report: a summary, then one section per step, headed by where its Datasets came from."""

__all__ = ["chain_blocks", "lineage_line"]

from collections.abc import Sequence
from typing import TYPE_CHECKING

from dataeval_flow._blocks import Block, Fields, Paragraph, Section
from dataeval_flow._result import LineageRecord, failure_section

if TYPE_CHECKING:
    from dataeval_flow.steps._result import ChainResult, StepResult


def lineage_line(address: str, lineage: Sequence[LineageRecord]) -> str:
    """``address`` walked back through the Datasets it was made from, to the source: "`few` ← `k` ← `a` (src)"."""
    records = {record.name: record for record in lineage}
    parts: list[str] = []
    current: str | None = address
    while current is not None and current not in parts:
        record = records.get(current)
        parts.append(current)
        current = record.inputs[0] if record is not None and record.inputs else None
    text = " ← ".join(f"`{part}`" for part in parts)
    last = records.get(parts[-1])
    return f"{text} ({last.source})" if last is not None and last.source else text


def chain_blocks(result: "ChainResult", *, detailed: bool) -> list[Block]:
    """The summary, then a "Steps" section holding one section per step, in run order.

    Nested one level under "Steps" so a step's own heading, such as ``few (toy-first)``, keeps its case: the
    report's top-level sections, like the workflow health summary, render theirs in capitals.
    """
    counts = {status: sum(r.status == status for r in result.steps.values()) for status in ("ok", "failed", "skipped")}
    steps: list[Block] = [
        Section(
            title=f"{name} ({record.type})",
            brief=None if record.status == "ok" else record.status,
            blocks=_step(record, result, detailed=detailed),
        )
        for name, record in result.steps.items()
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
        *result._summary_blocks(result.findings),  # noqa: SLF001 - a chain's report reuses a workflow's summary
        Section(title="Steps", blocks=steps),
    ]


def _step(record: "StepResult", result: "ChainResult", *, detailed: bool) -> list[Block]:
    from dataeval_flow.evaluators._report import output_blocks, serialized_of
    from dataeval_flow.evaluators._result import EvaluatorResult
    from dataeval_flow.steps._registry import TRANSFORMS
    from dataeval_flow.workflows._result import WorkflowResult

    blocks: list[Block] = []
    if record.inputs:
        blocks.append(
            Paragraph(
                text="On "
                + ", ".join(lineage_line(address.split("[")[0], result.metadata.lineage) for address in record.inputs)
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
    step_result = record.result
    if isinstance(step_result, EvaluatorResult):
        blocks.extend(output_blocks(serialized_of(step_result), detailed=detailed))
    elif isinstance(step_result, WorkflowResult):
        blocks.extend(step_result._report_output(detailed=detailed))  # noqa: SLF001 - a step's own report has no public accessor
    elif record.kind == "transform":
        section = TRANSFORMS.get(record.type)().section(record) if record.type in TRANSFORMS.names() else []
        blocks.extend(section or [Fields(items=_dataset_fields(record.summary))])
    return blocks


def _dataset_fields(summary: object) -> list[tuple[str, str | int | float | bool | None]]:
    if isinstance(summary, dict) and "items" in summary:
        return [("Items", summary["items"]), ("Digest", summary.get("digest"))]
    if isinstance(summary, dict):
        return [
            (str(key), value["items"] if isinstance(value, dict) and "items" in value else str(value))
            for key, value in summary.items()
        ]
    return [("Output", str(summary))]
