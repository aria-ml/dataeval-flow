"""A chain's report: a summary, each finding beside the evidence it judged, the other steps, then every step."""

__all__ = ["Evidence", "chain_blocks", "finding_sections", "lineage_line", "step_heading"]

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dataeval_flow._blocks import Block, Cell, Column, Fields, Paragraph, Section, Table
from dataeval_flow._result import LineageRecord, failure_section
from dataeval_flow._step_title import step_title

if TYPE_CHECKING:
    from dataeval_flow.steps._result import ChainResult, StepResult, StepStatus


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
    """The step count, the summary, each check's findings beside their evidence, the other steps, then the Steps
    table: top-level sections alongside Configuration. The other steps are those not shown as evidence that have
    something to show, a check among them only where it did not complete."""
    evidence = Evidence(result, detailed=detailed)
    # A check that ran once per element shows each element's findings beside that element's evidence, key by key.
    findings = [
        section
        for record in result.steps.values()
        if record.kind == "check"
        for run in ([record] if record.elements is None else record.elements.values())
        for section in finding_sections(run, evidence)
    ]
    others = [section for record in result.steps.values() if (section := _other(record, evidence)) is not None]
    return [
        Fields(items=[("Steps", _count(result.steps.values()))]),
        *result._summary_blocks(),  # noqa: SLF001 - a chain's report reuses a workflow's summary
        *findings,
        *others,
        *([_steps_table(result)] if result.steps else []),
    ]


@dataclass
class Evidence:
    """The evidence a chain's report has shown so far: each step, or element of one, by the finding it is under.

    Evidence is shown once, under the first finding that reads it; a later finding that reads it points there.
    """

    result: "ChainResult"
    detailed: bool
    shown: dict[tuple[str, str | None], str] = field(default_factory=dict)
    """By step name and element key, ``None`` for the whole step, the title of the finding it is shown under."""

    def under(self, name: str, key: str | None) -> str | None:
        """The title of the finding step `name`'s element `key`, or the whole step, is shown under; ``None`` when
        neither is shown yet."""
        return self.shown.get((name, key)) or self.shown.get((name, None))

    def elements_shown(self, name: str) -> dict[str, str]:
        """Each element of step `name` shown on its own, by the finding it is under."""
        return {key: title for (step, key), title in self.shown.items() if step == name and key is not None}

    def blocks(self, addresses: Sequence[str], title: str) -> list[Block]:
        """What finding `title` read at `addresses`: each step that made it, walking back through combines, as a
        "From" section the first time it is read and as a line naming the finding it is under after that."""
        return [
            block for record, key in _reads(addresses, self.result.steps) for block in self._show(record, key, title)
        ]

    def _show(self, record: "StepResult", key: str | None, title: str) -> list[Block]:
        heading = step_heading(record) + ("" if key is None else f" [{key}]")
        earlier = self.under(record.name, key)
        if earlier is not None:
            return [Paragraph(text=f"Evidence: {heading}, under {earlier}.")]
        at = _at(record, key)
        skip = self.elements_shown(record.name) if key is None and record.elements is not None else {}
        pointers: list[Block] = [
            Paragraph(text=f"Evidence: {heading} [{element}], under {finding}.") for element, finding in skip.items()
        ]
        blocks = _step(at, detailed=self.detailed, skip=skip.keys())
        if not blocks:  # a combine that shows nothing adds no heading
            return pointers
        self.shown[(record.name, key)] = title
        return [*pointers, Section(title=f"From {heading}", brief=_brief(_status(at, skip.keys())), blocks=blocks)]


def finding_sections(record: "StepResult", evidence: Evidence) -> list[Section]:
    """One section per finding `record` made, a check that ran once or one element of one: the finding's own blocks,
    then the evidence it judged, from the steps `record` read."""
    from dataeval_flow.workflows._result import finding_section, summary_label

    findings = record.output if record.status == "ok" and record.elements is None else None
    sections: list[Section] = []
    for finding in findings or []:
        own, title = finding_section(finding), summary_label(finding)
        sections.append(
            Section(
                title=title,
                brief=own.brief,
                severity=own.severity,
                blocks=[*own.blocks, *evidence.blocks(record.inputs, title)],
            )
        )
    return sections


def _producer(address: str, steps: Mapping[str, "StepResult"]) -> tuple["StepResult", str | None] | None:
    """The step that made `address`, and the key of the element of it read where it ran once per element; ``None``
    for a chain input.

    `address` names a step, one output of it (`split.train`), or a preset step's declared output (`cleaning.clean`),
    which its spliced step `cleaning/clean` made; `[key]` names one element.
    """
    base, _, rest = address.partition("[")
    key = rest[:-1] if rest.endswith("]") else None
    name, _, output = base.partition(".")
    record = steps.get(base) or steps.get(name) or (steps.get(f"{name}/{output}") if output else None)
    if record is None:
        return None
    return record, key if key is not None and record.elements is not None and key in record.elements else None


def _reads(addresses: Sequence[str], steps: Mapping[str, "StepResult"]) -> list[tuple["StepResult", str | None]]:
    """Each step, or element of one, that made what `addresses` hold, in order, each combine followed by what it
    read in turn."""
    found: list[tuple[StepResult, str | None]] = []

    def visit(address: str) -> None:
        made = _producer(address, steps)
        if made is None or any(record is made[0] and key == made[1] for record, key in found):
            return
        found.append(made)
        record, key = made
        if record.kind == "combine":
            for read in _at(record, key).inputs:
                visit(read)

    for address in addresses:
        visit(address)
    return found


def _at(record: "StepResult", key: str | None) -> "StepResult":
    """`record`, or its element `key`."""
    return record if key is None or record.elements is None else record.elements[key]


def _other(record: "StepResult", evidence: Evidence) -> Section | None:
    """A step not shown as evidence, where it has something to show: what it made, or why it made nothing. A check's
    findings have sections of their own, so a check shows here only where it, or an element of it, did not complete."""
    if (record.name, None) in evidence.shown:
        return None
    skip = evidence.elements_shown(record.name).keys()
    blocks = _step(record, detailed=evidence.detailed, skip=skip)
    if not blocks:
        return None
    return Section(title=step_heading(record), brief=_brief(_status(record, skip)), blocks=blocks)


def _status(record: "StepResult", skip: Collection[str]) -> "StepStatus":
    """`record`'s status, or, less the elements in `skip`, the status the rest of its elements add up to."""
    if not skip or record.elements is None:
        return record.status
    statuses = {element.status for key, element in record.elements.items() if key not in skip}
    return "failed" if "failed" in statuses else "skipped" if statuses == {"skipped"} else "ok"


def _brief(status: "StepStatus") -> str | None:
    return None if status == "ok" else status


def _count(records: Iterable["StepResult"]) -> str:
    """How many steps there are: "10 ran", or "10 (8 ran, 1 failed, 1 skipped)" where some did not complete."""
    statuses = [record.status for record in records]
    if all(status == "ok" for status in statuses):
        return f"{len(statuses)} ran"
    words: dict[StepStatus, str] = {"ok": "ran", "failed": "failed", "skipped": "skipped"}
    parts = [f"{count} {word}" for status, word in words.items() if (count := statuses.count(status))]
    return f"{len(statuses)} ({', '.join(parts)})"


_COLUMNS = (
    ("step", "Step"),
    ("title", "Title"),
    ("type", "Type"),
    ("status", "Status"),
    ("reads", "Reads"),
    ("note", "Note"),
)


def _steps_table(result: "ChainResult") -> Section:
    """Every step, in run order: its name, title, type and status, where each Dataset it read came from, and why it
    made nothing, if it did not."""
    rows: list[dict[str, Cell]] = [
        {
            "step": record.name,
            "title": step_title(record.kind, record.type),
            "type": record.type,
            "status": record.status,
            "reads": "\n".join(lineage_line(address, result.metadata.lineage) for address in record.inputs),
            "note": _note(record),
        }
        for record in result.steps.values()
    ]
    columns = [Column(key=key, header=header, align="left") for key, header in _COLUMNS]
    return Section(title="Steps", reference=True, blocks=[Table(columns=columns, rows=rows)])


def _note(record: "StepResult") -> str:
    """Why a step made nothing: its failure or skip reason, or "no findings" for a check; each element's, by key."""
    if record.elements is not None:
        notes = ((key, _note(element)) for key, element in record.elements.items())
        return "\n".join(f"[{key}] {note}" for key, note in notes if note)
    if record.status == "failed":
        return "; ".join(record.errors)
    if record.status == "skipped":
        return record.reason or ""
    return "no findings" if record.kind == "check" and not record.output else ""


def _step(record: "StepResult", *, detailed: bool, skip: Collection[str] = ()) -> list[Block]:
    """What a step shows: why it was skipped or how it failed, what it made, or each element's, less those in `skip`."""
    blocks: list[Block] = []
    if record.reason is not None:
        blocks.append(Paragraph(text=f"Skipped: {record.reason}" if record.status == "skipped" else record.reason))
    if record.status == "failed" and record.errors:
        blocks.append(failure_section(record.errors))
    if record.elements is not None:
        for key, element in record.elements.items():
            inner = [] if key in skip else _step(element, detailed=detailed)
            if inner:
                blocks.append(Section(title=f"[{key}]", brief=_brief(element.status), blocks=inner))
        return blocks
    if record.status != "ok":
        return blocks
    blocks.extend(_output_blocks(record, detailed=detailed))
    return blocks


def _output_blocks(record: "StepResult", *, detailed: bool) -> list[Block]:
    """What a completed step made: an evaluator's or workflow's report, or a transform's section. A check's findings
    have sections of their own, and a combine shows nothing."""
    from dataeval_flow.evaluators._result import EvaluatorResult
    from dataeval_flow.steps._registry import TRANSFORMS
    from dataeval_flow.workflows._result import WorkflowResult

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
