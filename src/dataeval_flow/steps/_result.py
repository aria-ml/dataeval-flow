"""What a chain's steps produced: one result per step, and the chain's own result over every step (spec §7)."""

__all__ = ["ChainMetadata", "ChainOutput", "ChainResult", "StepResult"]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from pydantic import Field

from dataeval_flow._binning_report import binning_blocks
from dataeval_flow._blocks import Block, Paragraph, Scalar, Section, Summary, SummaryItem
from dataeval_flow._result import LineageRecord, Result, ResultKind, ResultMetadata, failure_section, finite_json
from dataeval_flow.steps._step import StepKind
from dataeval_flow.workflows._base import Finding
from dataeval_flow.workflows._result import element_key

if TYPE_CHECKING:
    from dataeval_flow._chain._presets import SpliceRun
    from dataeval_flow._chain._run import ChainRun
    from dataeval_flow._chain._verdict import Verdict
    from dataeval_flow.workflows._preset import PresetChain, ReportGroup

StepStatus = Literal["ok", "failed", "skipped"]


@dataclass
class StepResult:
    """One step's outcome in a chain: its status, what it read, and what it made.

    ``output`` is the live object: a Dataset (a DataEval ``View`` you can go on to use), DataEval's output, an
    export record, or a mapping of them for a step with several outputs. A step that ran once
    per element of a list has ``elements`` instead, one :class:`StepResult` per key.
    """

    name: str
    kind: StepKind
    type: str
    inputs: list[str]
    status: StepStatus
    output: Any = None
    reason: str | None = None
    errors: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    elements: "dict[str, StepResult] | None" = None
    result: "Result[Any, Any] | None" = None
    summary: Any = None
    optional: bool = False
    details: dict[str, Any] | None = None
    not_assessed: str | None = None
    """Why this check, or element of one, judged nothing; or, for a step run once per element of a list holding none,
    the reason that list gives, such as an empty list input's `empty:` (audit spec §9.1). ``None`` otherwise. On a
    step run once per element, each element records its own."""

    def to_dict(self) -> dict[str, Any]:
        """This step as JSON: its kind, status and inputs, then what it made, or why it made nothing."""
        body: dict[str, Any] = {
            "kind": self.kind,
            "type": self.type,
            "status": self.status,
            "inputs": list(self.inputs),
        }
        if self.optional:
            body["optional"] = True
        if self.reason is not None:
            body["reason"] = self.reason
        if self.not_assessed is not None:
            body["not_assessed"] = self.not_assessed
        if self.errors:
            body["errors"] = list(self.errors)
        if self.elements is not None:
            body["elements"] = {key: element.to_dict() for key, element in self.elements.items()}
        elif self.status == "ok":
            body.update(_made(self))
        if self.details is not None:
            body["details"] = self.details
        body["elapsed_s"] = round(self.elapsed, 3)
        return body


def _made(record: "StepResult") -> dict[str, Any]:
    from dataeval_flow.evaluators._report import serialized_of
    from dataeval_flow.evaluators._result import EvaluatorResult

    result = record.result
    if isinstance(result, EvaluatorResult):
        return {"dataeval": result.metadata.dataeval.model_dump(mode="json"), "output": serialized_of(result)}
    return {"output": record.summary}


class ChainMetadata(ResultMetadata):
    """The JATIC envelope of a custom workflow's result, with where each of its Datasets came from."""

    workflow: str = Field(default="", description="The custom workflow's name.")
    lineage: list[LineageRecord] = Field(
        default_factory=list, description="Each Dataset in the chain, in the order it was made, with what made it."
    )


@dataclass
class ChainOutput:
    """A successful chain's steps, by name, in run order."""

    steps: dict[str, StepResult]


class ChainResult(Result[ChainMetadata, ChainOutput]):
    """A workflow's result, a custom workflow's or a workflow type's: every step's outcome, in order, whether or not a
    step failed.

    A required step's failure fails the result (``success`` is false, and ``output`` raises, as for any result), and
    ``health["status"]`` is ``"failed"``, as it is for a task refused before any step ran, whose reason is in
    ``errors``. The steps that ran stay readable in :attr:`steps`. Its findings are its check steps', counted by
    :attr:`health` and listed at the JSON's top level. ``kind`` is ``"workflow"``.

    Fields
    ------
    steps
        Every step's :class:`~dataeval_flow.steps.StepResult`, by name, in run order — readable even where a
        later step failed or was skipped.
    metadata.workflow
        The custom workflow's name.
    metadata.lineage
        Each Dataset in the chain, in the order it was made, with what made it.

    Examples
    --------
    >>> from dataeval_flow import load_config, run_tasks
    >>> result = run_tasks(load_config("pipeline.yaml"))["clean"]  # doctest: +SKIP
    >>> result.steps["clean"].output  # the cleaned Dataset, a DataEval View  # doctest: +SKIP
    """

    kind: ClassVar[ResultKind] = "workflow"
    metadata_type: ClassVar[type[ResultMetadata]] = ChainMetadata

    def __init__(
        self,
        *,
        type: str,  # noqa: A002
        success: bool,
        metadata: ChainMetadata,
        output: ChainOutput | None = None,
        errors: Sequence[str] = (),
        dataset: Any = None,
        sources: Any = None,
        steps: Mapping[str, StepResult] | None = None,
    ) -> None:
        super().__init__(
            type=type,
            success=success,
            metadata=metadata,
            output=output,
            errors=errors,
            dataset=dataset,
            sources=sources,
        )
        self.steps: dict[str, StepResult] = dict(steps or {})
        self._preset = False  # whether a preset entry's chain ran, which the banner names; a custom workflow has none
        self.preset_chain: PresetChain | None = None
        """What the preset entry expanded to, with the report's groups, record and next steps; ``None`` otherwise."""
        self.verdict: Verdict | None = None
        """Whether the data is ready, for a preset chain that declares `blocking` and succeeded; ``None`` otherwise."""
        self.splice: str | None = None
        """The spliced step whose chain `preset_chain` is, as `audit`; ``None`` where it is the task's own."""
        self.custom_groups: tuple[ReportGroup, ...] = ()
        """The custom workflow's own groups, reported after a splice's questions."""
        self.splice_runs: dict[str, SpliceRun] = {}
        """How each splice started, by step name; the Datasets its slots bound, or why it never started or failed."""
        self.no_verdict: str | None = None
        """Why a chain that declares a verdict gave none though the task succeeded; its splice never started, or
        failed as it started."""
        self.splice_binning: dict[str, dict[str, Any]] = {}
        """By step name, the binning record of what each splice's own steps read; the encoding its record shows."""

    @classmethod
    def from_run(cls, name: str, run: "ChainRun", *, type_id: str | None = None, preset: bool = False) -> "ChainResult":
        """The result of running custom workflow `name`: failed when any required step failed. Its ``type`` is
        `type_id`, a preset's type id, or `name` when unset; `preset` says a preset entry's chain ran, which the banner
        names, where a custom workflow has no type. Its envelope records the encodings its steps read (spec §10.10)."""
        failed = [step for step, record in run.steps.items() if record.status == "failed"]
        errors = [f"{step}: {'; '.join(_errors(run.steps[step]))}" for step in failed]
        metadata = ChainMetadata(workflow=name, lineage=list(run.lineage), label_space=list(run.label_space))
        from dataeval_flow._chain._reads import attach_reads, binning_record

        attach_reads(metadata, run.reads)
        result = cls(
            type=type_id or name,
            success=not failed,
            metadata=metadata,
            output=ChainOutput(dict(run.steps)) if not failed else None,
            errors=errors,
            steps=run.steps,
        )
        result._preset = preset
        result.splice_runs = dict(run.splices)
        result.splice_binning = {
            name: record for name, reads in run.splice_reads.items() if (record := binning_record(reads)) is not None
        }
        return result

    @property
    def declared_steps(self) -> dict[str, StepResult]:
        """The steps preset_chain's verdict, record and questions read; its splice's alone, or every step."""
        if self.splice is None:
            return self.steps
        prefix = f"{self.splice}/"
        return {name: record for name, record in self.steps.items() if name.startswith(prefix)}

    def attach_preset(
        self, chain: "PresetChain", *, splice: str | None = None, groups: "Sequence[ReportGroup]" = ()
    ) -> None:
        """Keep `chain`, the preset entry's expansion or a spliced step's, and judge the verdict it declares over its
        steps, if the task succeeded and the splice started."""
        from dataeval_flow._chain._verdict import judge

        self.preset_chain, self.splice, self.custom_groups = chain, splice, tuple(groups)
        if chain.blocking is None or not self.success:
            return
        started = self.splice_runs.get(splice) if splice is not None else None
        if started is not None and started.skipped is not None:
            self.no_verdict = f"step `{splice}` did not run: {started.skipped}"
            return
        if started is not None and started.failed is not None:
            self.no_verdict = f"step `{splice}` failed as it started: {started.failed}"
            return
        prefix = f"{splice}/" if splice is not None else ""
        self.verdict = judge(self.declared_steps, blocking=chain.blocking, accepted=chain.accepted, prefix=prefix)

    @property
    def failed_steps(self) -> list[str]:
        """The required steps that failed, in run order."""
        return [name for name, record in self.steps.items() if record.status == "failed"]

    @property
    def findings(self) -> list[Finding]:
        """Every finding the health counts, in run order: each check's."""
        return [finding for record in self.steps.values() for finding in _step_findings(record)]

    def findings_by_step(self) -> list[tuple[str, Finding]]:
        """Every finding :attr:`findings` holds, beside the name of the step that made it, in run order."""
        return [(name, finding) for name, record in self.steps.items() for finding in _step_findings(record)]

    @property
    def warning_count(self) -> int:
        """How many findings are warnings: the one count every renderer reads."""
        return sum(finding.severity == "warning" for finding in self.findings)

    @property
    def health(self) -> dict[str, Any]:
        """``failed`` when a required step failed, or the chain was refused before any step ran; else ``warning`` or
        ``ok``, with the counts behind it."""
        findings, warnings = self.findings, self.warning_count
        status = "failed" if self.failed_steps or not self.success else "warning" if warnings else "ok"
        return {"status": status, "warnings": warnings, "findings": len(findings), "failed_steps": self.failed_steps}

    def to_dict(self) -> dict[str, object]:
        """Kind, envelope, health and every step, then any errors and thumbnails, whether or not a step failed.
        Non-finite floats are written as ``null``.
        """
        payload: dict[str, object] = {
            "kind": self.kind,
            "metadata": self.metadata.model_dump(mode="json"),
            "health": self.health,
            "steps": {name: record.to_dict() for name, record in self.steps.items()},
            "findings": [finding.model_dump(mode="json") for finding in self.findings],
        }
        if self.verdict is not None:
            payload["verdict"] = self.verdict.model_dump(mode="json")
        if self.no_verdict:
            payload["no_verdict"] = self.no_verdict
        if self.errors:
            payload["errors"] = list(self.errors)
        if self.assets:
            payload["assets"] = [asset.model_dump(mode="json") for asset in self.assets]
        return cast("dict[str, object]", finite_json(payload))

    def _report_title(self) -> str:
        return super()._report_title() if self._preset else self.metadata.workflow or self.type

    def _report_ran(self) -> tuple[str, Scalar]:
        return super()._report_ran() if self._preset else ("Workflow", f"{self._report_title()} (custom workflow)")

    def _report_body(self, *, detailed: bool) -> list[Block]:
        """Every step's report, whether or not a step failed; or, for a chain refused before any step ran, why."""
        from dataeval_flow._chain._report import chain_blocks

        if not self.success and not self.steps:
            return [failure_section(self.errors)]
        blocks = chain_blocks(self, detailed=detailed)
        blocks.extend(binning_blocks(self.metadata.metadata_binning, self.metadata.diagnostics, detailed=detailed))
        return blocks

    def _report_output(self, *, detailed: bool) -> list[Block]:
        return self._report_body(detailed=detailed)

    def _dict_body(self) -> dict[str, object]:
        return {key: value for key, value in self.to_dict().items() if key not in ("kind", "metadata")}

    def _summary_blocks(self) -> list[Block]:
        """One summary line per finding, then the health verdict: failed, naming the required steps that failed, where
        :attr:`health` says the run failed; else the warnings :attr:`warning_count` counted."""
        findings = self.findings
        health = self.health
        failed = list(health.get("failed_steps") or []) if health["status"] == "failed" else []
        if not findings and not failed:
            return [Paragraph(text="No findings to report.")]
        # A finding of a check that ran once per element sits under its element's key, after those that did not.
        keys = list(dict.fromkeys(key for f in findings if (key := element_key(f)) is not None))
        ordered = [f for group in [None, *keys] for f in findings if element_key(f) == group]
        items = [
            SummaryItem(label=f.title, value=f.brief or "", severity=f.severity, group=element_key(f) or "")
            for f in ordered
        ]
        summary = Summary(items=items, warnings=self.warning_count, failed=failed)
        lede: list[Block] = [] if findings else [Paragraph(text="No findings to report.")]
        return [Section(title="Summary", blocks=[*lede, summary])]


def _step_findings(record: StepResult) -> list[Finding]:
    """The findings one step made: a check's own; each element's, in key order."""
    if record.elements is not None:
        return [finding for element in record.elements.values() for finding in _step_findings(element)]
    return list(record.output or []) if record.kind == "check" and record.status == "ok" else []


def _errors(record: StepResult) -> list[str]:
    if record.elements is not None:
        return [f"[{key}] {error}" for key, element in record.elements.items() for error in _errors(element)]
    return list(record.errors)
