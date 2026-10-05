"""A chain's report: a summary, each finding beside the evidence it judged, the other steps, then every step.

A preset that declares a verdict, a record or questions has its own: the verdict, the record, a section per question,
next steps, the other steps, then every step.
"""

__all__ = ["Evidence", "chain_blocks", "finding_sections", "lineage_line", "question_status", "step_heading"]

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dataeval_flow._blocks import Block, BulletList, Cell, Column, Fields, Paragraph, Scalar, Section, Table, Verdict
from dataeval_flow._chain import _verdict
from dataeval_flow._result import LineageRecord, failure_section
from dataeval_flow._step_title import step_title

if TYPE_CHECKING:
    from dataeval_flow.steps._result import ChainResult, StepResult, StepStatus
    from dataeval_flow.workflows._preset import NextSteps, PresetChain, ReportGroup


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
    something to show, a check among them only where it did not complete. Short (not *detailed*), only the count, the
    summary and a compact Steps table: Step, Status and Note, with no finding and no evidence.

    A preset chain that declares questions, a record or a verdict reports as it declares (:func:`_declared_blocks`).
    """
    plan = result.preset_chain
    if plan is not None and (plan.groups or plan.record is not None or plan.blocking is not None):
        return _declared_blocks(result, plan, detailed=detailed)
    if not detailed:
        return [
            Fields(items=[("Steps", _count(result.steps.values()))]),
            *result._summary_blocks(),  # noqa: SLF001 - a chain's report reuses a workflow's summary
            *([_steps_table(result, compact=True)] if result.steps else []),
        ]
    evidence = Evidence(result, detailed=detailed)
    findings = _findings(result.steps.values(), evidence)
    others = _others(result.steps.values(), evidence)
    return [
        Fields(items=[("Steps", _count(result.steps.values()))]),
        *result._summary_blocks(),  # noqa: SLF001 - a chain's report reuses a workflow's summary
        *findings,
        *others,
        *([_steps_table(result)] if result.steps else []),
    ]


def _findings(records: Iterable["StepResult"], evidence: "Evidence") -> list[Block]:
    """The findings of the checks among `records`, each beside its evidence. A check that ran once per element shows
    each element's findings beside that element's evidence, in a section of that element's key; the findings of checks
    that did not run per element come first, ungrouped."""
    ungrouped: list[Block] = []
    groups: dict[str, list[Block]] = {}
    for record in records:
        if record.kind != "check":
            continue
        if record.elements is None:
            ungrouped.extend(finding_sections(record, evidence))
            continue
        for key, element in record.elements.items():
            if sections := finding_sections(element, evidence):
                groups.setdefault(key, []).extend(sections)
    return [*ungrouped, *(Section(title=key, blocks=blocks) for key, blocks in groups.items())]


def _declared_blocks(result: "ChainResult", plan: "PresetChain", *, detailed: bool) -> list[Block]:
    """A preset chain's report, as `plan` declares it.

    The verdict, and under it what it rests on, or the summary where the run gave no verdict, since it failed or
    `plan` declares none; the record; a section per question, its brief the question's status, holding its checks'
    findings beside their evidence and then its evidence steps; next steps; the findings of checks no question names;
    the other steps, less those the record or a question shows; then the Steps table. Short (not *detailed*), the
    verdict, the record, a Questions section with a line per question and its status, and the compact Steps table.
    A question none of whose checks is in the chain is left out, as it judged nothing.
    """
    verdict = result.verdict
    records = list(result.steps.values())
    head = [_verdict_block(verdict)] if verdict is not None else result._summary_blocks()  # noqa: SLF001 - as above
    record = [_record_section(result, plan)] if plan.record is not None else []
    groups = [g for g in plan.groups if any(r.kind == "check" and r.type in g.checks for r in records)]
    statuses = [(group, question_status(result.steps, group, plan.next_steps)) for group in groups]
    steps = [_steps_table(result, compact=not detailed)] if result.steps else []
    if not detailed:
        lines: list[tuple[str, Scalar]] = [(group.heading, status) for group, status in statuses]
        questions: list[Block] = [Section(title="Questions", blocks=[Fields(items=lines)])] if lines else []
        return [*head, *record, *questions, *steps]
    evidence = Evidence(result, detailed=detailed)
    sections = [
        Section(
            title=group.heading,
            brief=status,
            blocks=[
                *_findings((record for record in records if record.type in group.checks), evidence),
                *_shown_under(group.heading, [record for record in records if record.type in group.evidence], evidence),
            ],
        )
        for group, status in statuses
    ]
    named = {check for group in groups for check in group.checks}
    loose = _findings((record for record in records if record.type not in named), evidence)
    shown = {*(plan.record.steps if plan.record is not None else ()), *(t for g in groups for t in g.evidence)}
    others = _others((record for record in records if record.type not in shown), evidence)
    advice = _verdict.next_step_lines(verdict, plan.next_steps) if verdict is not None else []
    return [
        *head,
        *(_verdict_section(verdict) if verdict is not None else []),
        *record,
        *sections,
        *([Section(title="Next steps", blocks=[BulletList(items=advice)])] if advice else []),
        *loose,
        *others,
        *steps,
    ]


def _shown_under(heading: str, records: Sequence["StepResult"], evidence: "Evidence") -> list[Section]:
    """`records`' sections, each shown under the question `heading`, so that a later finding reading one points here."""
    sections: list[Section] = []
    for record in records:
        if (section := _other(record, evidence)) is not None:
            evidence.shown[(record.name, None)] = heading
            sections.append(section)
    return sections


def question_status(steps: Mapping[str, "StepResult"], group: "ReportGroup", plan: "NextSteps") -> str:
    """A question's status, from the runs of its checks among `steps`, each check that ran once or each element of
    one: "ok" where every run was assessed and none warned; "not assessed: <reason>" where every run went unassessed
    for one class of reason, as `plan` classes reasons, the class's first letter lower case; else how many warnings
    and how many runs not assessed, as "1 warning, 1 not assessed". An accepted warning counts here: it still warns,
    and the verdict records the acceptance.
    """
    checks = {name: record for name, record in steps.items() if record.kind == "check" and record.type in group.checks}
    tally = _verdict.judge(checks, blocking=(), accepted={})
    warnings, missed = len(tally.warnings), tally.not_assessed
    runs = sum(1 if record.elements is None else len(record.elements) for record in checks.values())
    reasons = {_verdict.reason_class(unassessed.reason, plan) for unassessed in missed}
    if missed and len(missed) == runs and len(reasons) == 1:
        reason = reasons.pop()
        return f"not assessed: {reason[:1].lower()}{reason[1:]}"
    counts = [f"{warnings} warning{'s' if warnings != 1 else ''}"] if warnings else []
    counts += [f"{len(missed)} not assessed"] if missed else []
    return ", ".join(counts) or "ok"


def _verdict_block(verdict: "_verdict.Verdict") -> Verdict:
    return Verdict(level=verdict.level, label=verdict.label, line=verdict.line())


def _verdict_section(verdict: "_verdict.Verdict") -> list[Block]:
    """What `verdict` rests on, where it rests on anything: the blocking warnings, the other warnings, each acceptance
    with whether its check warned and why it is accepted, and each check not assessed with why."""

    def warned(item: _verdict.VerdictItem) -> str:
        return f"{item.title} ({item.step})" + (f": {item.brief}" if item.brief else "")

    parts = [
        ("Blocking", [warned(item) for item in verdict.blocking]),
        ("Warnings", [warned(item) for item in verdict.warnings]),
        (
            "Accepted risks",
            [f"{step_title('check', a.check)} ({a.state.replace('-', ' ')}): {a.reason}" for a in verdict.accepted],
        ),
        ("Not assessed", [f"{step_title('check', u.check)} ({u.step}): {u.reason}" for u in verdict.not_assessed]),
    ]
    sections: list[Block] = [Section(title=title, blocks=[BulletList(items=items)]) for title, items in parts if items]
    return [Section(title="Verdict", blocks=sections)] if sections else []


def _record_section(result: "ChainResult", plan: "PresetChain") -> Section:
    """`plan`'s record of what the chain read: a table with a column per source, then the run, and the criteria the
    verdict applied."""
    record = plan.record
    assert record is not None  # noqa: S101 - the caller draws a record only where the plan declares one
    meta = result.metadata
    libraries = ", ".join(f"{name} {version}" for name, version in meta.library_versions.items())
    facts: list[tuple[str, Scalar]] = [
        ("Flow", meta.tool_version),
        ("Libraries", libraries),
        ("Extractor", meta.model_id),
        ("Device", meta.device),
        ("Seed", meta.resolved_config.get("seed")),
        ("Timestamp", meta.timestamp.isoformat() if meta.timestamp else None),
    ]
    checks: dict[str, Any] = (meta.resolved_config.get("workflow") or {}).get("checks") or {}
    criteria: list[tuple[str, Scalar]] = [(check, _settings(settings)) for check, settings in checks.items()]
    if plan.blocking is not None:
        criteria.append(("Blocking", ", ".join(plan.blocking) or "none"))
    if plan.accepted:
        criteria.append(("Accepted", "\n".join(f"{check}: {reason}" for check, reason in plan.accepted.items())))
    return Section(
        title=record.title,
        blocks=[
            *_record_table(result, record.steps),
            Section(title="Run", blocks=[Fields(items=[(label, v) for label, v in facts if v not in (None, "")])]),
            *([Section(title="Criteria", blocks=[Fields(items=criteria)])] if criteria else []),
        ],
    )


def _record_table(result: "ChainResult", kinds: Sequence[str]) -> list[Block]:
    """A column per source, in the task's order, and a row per fact any source has: its description and provenance;
    what each result of the step types `kinds` records, the types in that order and each type's rows in its own; the
    label space its labels were conformed to; its metadata factors and their encoding; and how the Dataset the
    record's steps read was made. A digest too long for a cell shows its first characters there, and in full in the
    fields under the table, a line each."""
    meta = result.metadata
    lineage = {entry.name: entry for entry in meta.lineage}
    inputs = {entry.source: entry.name for entry in meta.lineage if entry.step is None and entry.source is not None}
    runs = [
        (kind, *run)
        for kind in kinds
        for step in result.steps.values()
        if step.type == kind
        for run in _runs(step, lineage)
    ]
    reads: dict[str, str] = {}
    for _, split, run in runs:
        if split is not None and run.inputs:
            reads.setdefault(split, run.inputs[0])
    made = {split: reads.get(split, address) for split, address in inputs.items()}
    owners = {address: split for split, address in made.items()}
    sources: list[dict[str, Any]] = meta.resolved_config.get("sources") or []
    facts = [
        *(
            ("Source", source.get("name"), text)
            for source, text in zip(sources, meta.source_descriptions, strict=False)
        ),
        *(("Provenance", source.get("name"), _provenance(source)) for source in sources),
        *((label, split, value) for _, split, run in runs for label, value in _record_rows(run)),
        *(
            ("Label space", space.source if space.source in inputs else owners.get(space.source), space.digest)
            for space in meta.label_space
        ),
        *_binning_facts(meta.metadata_binning, list(inputs), lineage),
        *(("How it was made", split, lineage_line(address, meta.lineage)) for split, address in made.items()),
    ]
    cells: dict[str, dict[str, str]] = {}
    for label, split, value in facts:
        if split in inputs and value:
            cells.setdefault(label, {}).setdefault(split, value)
    order = [
        "Source",
        "Provenance",
        *_row_order(runs),
        "Label space",
        "Metadata factors",
        "Encoding",
        "How it was made",
    ]
    rows = [label for label in dict.fromkeys(order) if label in cells]
    if not rows:
        return []
    columns = [Column(key="", header="", align="left"), *(Column(key=s, header=s, align="left") for s in inputs)]
    table: list[dict[str, Cell]] = [
        {"": label, **{s: _cell(cells[label].get(s, "")) for s in inputs}} for label in rows
    ]
    full: list[tuple[str, Scalar]] = [
        (f"{label} ({s})", value)
        for s in inputs
        for label in rows
        if _DIGEST.fullmatch(value := cells[label].get(s, ""))
    ]
    return [Table(columns=columns, rows=table), *([Fields(items=full)] if full else [])]


# A digest a record cell has no room for: a hash of 32 hex characters or more, such as a content digest's 64.
_DIGEST = re.compile(r"[0-9a-f]{32,}")


def _cell(value: str) -> str:
    """`value` as a record cell shows it: a long digest by its first 12 characters."""
    return f"{value[:12]}…" if _DIGEST.fullmatch(value) else value


def _row_order(runs: Sequence[tuple[str, str | None, "StepResult"]]) -> list[str]:
    """The labels of the rows `runs`' results record: each step type's in the order its results give them, merged over
    its runs so a row one run lacks keeps its place, and the types in the order `runs` takes them."""
    orders: dict[str, list[str]] = {}
    for kind, _, run in runs:
        order, at = orders.setdefault(kind, []), 0
        for label, _ in _record_rows(run):
            if label not in order:
                order.insert(at, label)
            at = order.index(label) + 1
    return [label for order in orders.values() for label in order]


def _runs(step: "StepResult", lineage: Mapping[str, LineageRecord]) -> list[tuple[str | None, "StepResult"]]:
    """Each run of `step`, with the source it read: each element by its key, which names the source a list slot bound
    it from; a step that ran once, by the source its input walks back to through `lineage`."""
    if step.elements is not None:
        return list(step.elements.items())
    return [(_walk(step.inputs[0], lineage)[1] if step.inputs else None, step)]


def _record_rows(run: "StepResult") -> list[tuple[str, str]]:
    from dataeval_flow.evaluators._result import EvaluatorResult

    return run.result.record_rows() if isinstance(run.result, EvaluatorResult) else []


def _binning_facts(
    binning: Mapping[str, Any] | None, splits: list[str], lineage: Mapping[str, LineageRecord]
) -> list[tuple[str, str | None, str | None]]:
    """Each source's metadata factors, counted and named, and the encoding they were read under: each Dataset's binning
    record, by the source its address walks back to; one record alone, every source's."""
    records = (binning or {}).get("per_split") or ({None: binning} if binning else {})
    facts: list[tuple[str, str | None, str | None]] = []
    for key, encoded in records.items():
        names = list(encoded.get("factors") or {})
        for split in splits if key is None else [_walk(key.partition(" (")[0], lineage)[1]]:
            facts.append(("Metadata factors", split, f"{len(names)}: {', '.join(names)}" if names else None))
            facts.append(("Encoding", split, encoded.get("encoding_digest")))
    return facts


def _provenance(source: Mapping[str, Any]) -> str:
    """A source's dataset's `provenance:`, a `name: value` line each; a merged source's, each of its operands'."""
    facts = [
        (operand.get("dataset_config") or {}).get("provenance") or {} for operand in source.get("merge") or [source]
    ]
    return "\n".join(f"{name}: {value}" for fact in facts for name, value in fact.items())


def _settings(settings: object) -> str:
    """A check's settings on one line: "warning 3.0, info none"; a nested mapping in brackets, a list in square ones."""
    if isinstance(settings, Mapping):
        return ", ".join(
            f"{name} ({_settings(value)})" if isinstance(value, Mapping) else f"{name} {_settings(value)}"
            for name, value in settings.items()
        )
    if isinstance(settings, list | tuple):
        return f"[{', '.join(_settings(value) for value in settings)}]"
    return "none" if settings is None else str(settings).lower() if isinstance(settings, bool) else str(settings)


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
            return [Paragraph(text=f"Evidence: {heading}, under {_ended(earlier)}")]
        at = _at(record, key)
        skip = self.elements_shown(record.name) if key is None and record.elements is not None else {}
        pointers: list[Block] = [
            Paragraph(text=f"Evidence: {heading} [{element}], under {_ended(finding)}")
            for element, finding in skip.items()
        ]
        blocks = _step(at, detailed=self.detailed, skip=skip.keys())
        if not blocks:  # a combine that shows nothing adds no heading
            return pointers
        self.shown[(record.name, key)] = title
        return [*pointers, Section(title=f"From {heading}", brief=_brief(_status(at, skip.keys())), blocks=blocks)]


def _ended(title: str) -> str:
    """`title` ending a sentence: with a full stop, unless it is a question, which ends one already."""
    return title if title.endswith("?") else f"{title}."


def finding_sections(record: "StepResult", evidence: Evidence) -> list[Section]:
    """One section per finding `record` made, a check that ran once or one element of one: the finding's own blocks,
    then the evidence it judged, from the steps `record` read. A section is titled by the finding, without the element
    it judged: the chain's report groups an element's findings under its key. Evidence points back to a finding by the
    title that names its element."""
    from dataeval_flow.workflows._result import finding_section, summary_label

    findings = record.output if record.status == "ok" and record.elements is None else None
    sections: list[Section] = []
    for finding in findings or []:
        own, title = finding_section(finding), summary_label(finding)
        sections.append(
            Section(
                title=own.title,
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


def _others(records: Iterable["StepResult"], evidence: Evidence) -> list[Section]:
    return [section for record in records if (section := _other(record, evidence)) is not None]


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


def _steps_table(result: "ChainResult", *, compact: bool = False) -> Section:
    """Every step, in run order: its name, title, type and status, where each Dataset it read came from, and why it
    made nothing, if it did not. *compact* keeps the step, its status and that note, and is open, not folded: it is the
    short form's only account of the steps."""
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
    # Text leaves the title out: each step's section heading gives it, and text has little room.
    kept = ("step", "status", "note") if compact else tuple(key for key, _ in _COLUMNS)
    columns = [
        Column(key=key, header=header, align="left", in_text=key != "title") for key, header in _COLUMNS if key in kept
    ]
    return Section(title="Steps", reference=not compact, blocks=[Table(columns=columns, rows=rows)])


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
    have sections of their own, and a combine shows its section where it draws one."""
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
    if record.kind == "combine":
        from dataeval_flow.steps._registry import COMBINES

        return list(COMBINES.get(record.type)().section(record) or []) if record.type in COMBINES.names() else []
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
