"""What a chain's steps produced: one result per step, and the chain's own result over every step (spec §7)."""

__all__ = ["ChainMetadata", "ChainOutput", "ChainResult", "StepResult"]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from dataeval_flow._blocks import Block
from dataeval_flow._result import LineageRecord, ResultMetadata
from dataeval_flow.steps._step import StepKind
from dataeval_flow.workflows._base import Finding
from dataeval_flow.workflows._result import WorkflowResult

if TYPE_CHECKING:
    from dataeval_flow._chain._run import ChainRun
    from dataeval_flow._result import Result

StepStatus = Literal["ok", "failed", "skipped"]


@dataclass
class StepResult:
    """One step's outcome in a chain: its status, what it read, and what it made.

    ``output`` is the live object: a Dataset (a DataEval ``View`` you can go on to use), DataEval's output, a
    workflow's result, an export record, or a mapping of them for a step with several outputs. A step that ran once
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
    if isinstance(result, WorkflowResult):
        return {"output": result._dict_body()}  # noqa: SLF001 - a step's own report has no public accessor
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


class ChainResult(WorkflowResult[ChainMetadata, ChainOutput]):  # type: ignore[reportInvalidTypeArguments]
    """A custom workflow's result: every step's outcome, in order, whether or not a step failed.

    A required step's failure fails the result (``success`` is false, and ``output`` raises, as for any result), and
    ``health["status"]`` is ``"failed"``, as it is for a task refused before any step ran, whose reason is in
    ``errors``. The steps that ran stay readable in :attr:`steps`. Its findings are its check steps' and those of the
    workflow-type steps it ran, all counted by :attr:`health`. The JSON lists the check findings at its top level,
    while a workflow-type step's findings stay inside that step.

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
    >>> result = run_tasks(load_config("pipeline.yaml"))["audit"]  # doctest: +SKIP
    >>> result.steps["clean"].output  # the cleaned Dataset, a DataEval View  # doctest: +SKIP
    """

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

    @classmethod
    def from_run(cls, name: str, run: "ChainRun", *, type_id: str | None = None, preset: bool = False) -> "ChainResult":
        """The result of running custom workflow `name`: failed when any required step failed. Its ``type`` is
        `type_id`, a preset's type id, or `name` when unset; `preset` says a preset entry's chain ran, which the banner
        names, where a custom workflow has no type."""
        failed = [step for step, record in run.steps.items() if record.status == "failed"]
        errors = [f"{step}: {'; '.join(_errors(run.steps[step]))}" for step in failed]
        metadata = ChainMetadata(workflow=name, lineage=list(run.lineage), label_space=list(run.label_space))
        result = cls(
            type=type_id or name,
            success=not failed,
            metadata=metadata,
            output=ChainOutput(dict(run.steps)) if not failed else None,
            errors=errors,
            steps=run.steps,
        )
        result._preset = preset
        return result

    @property
    def failed_steps(self) -> list[str]:
        """The required steps that failed, in run order."""
        return [name for name, record in self.steps.items() if record.status == "failed"]

    @property
    def findings(self) -> list[Finding]:
        """Every finding the health counts, in run order: each check's, and each completed workflow-type step's."""
        return [finding for record in self.steps.values() for finding in _step_findings(record)]

    @property
    def check_findings(self) -> list[Finding]:
        """The findings the check steps made, in run order: those the JSON lists at the top level (spec §7.3)."""
        return [
            finding for record in self.steps.values() if record.kind == "check" for finding in _step_findings(record)
        ]

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
        """Kind, envelope, health and every step, then any errors and thumbnails, whether or not a step failed."""
        payload: dict[str, object] = {
            "kind": self.kind,
            "metadata": self.metadata.model_dump(mode="json"),
            "health": self.health,
            "steps": {name: record.to_dict() for name, record in self.steps.items()},
            "findings": [finding.model_dump(mode="json") for finding in self.check_findings],
        }
        if self.errors:
            payload["errors"] = list(self.errors)
        if self.assets:
            payload["assets"] = [asset.model_dump(mode="json") for asset in self.assets]
        return payload

    def _report_title(self) -> str:
        if self._preset:
            return super()._report_title()
        return f"{self.metadata.workflow or self.type}\n{self._report_subtitle()}"

    def _report_subtitle(self) -> str:
        return super()._report_subtitle() if self._preset else "custom workflow"

    def _report_body(self, *, detailed: bool) -> list[Block]:
        from dataeval_flow._chain._report import chain_blocks

        return chain_blocks(self, detailed=detailed)

    def _report_output(self, *, detailed: bool) -> list[Block]:
        return self._report_body(detailed=detailed)

    def _dict_body(self) -> dict[str, object]:
        return {key: value for key, value in self.to_dict().items() if key not in ("kind", "metadata")}


def _step_findings(record: StepResult) -> list[Finding]:
    """The findings one step made: a check's own, or a completed workflow type's; each element's, in key order."""
    if record.elements is not None:
        return [finding for element in record.elements.values() for finding in _step_findings(element)]
    if record.kind == "check":
        return list(record.output or []) if record.status == "ok" else []
    result = record.result
    return list(result.findings) if isinstance(result, WorkflowResult) and result.success else []


def _errors(record: StepResult) -> list[str]:
    if record.elements is not None:
        return [f"[{key}] {error}" for key, element in record.elements.items() for error in _errors(element)]
    return list(record.errors)
