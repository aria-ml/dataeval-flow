"""Preparing a preset's decomposition for an editor: a whole candidate pipeline, or why there can't be one (preset
decomposition spec §5). Reads no data, stores nothing and queues nothing; it imports no web framework."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dataeval_flow import __version__
from dataeval_flow._service import _contract
from dataeval_flow.steps._address import STEP_NAME_PATTERN

if TYPE_CHECKING:
    from collections.abc import Collection

    from dataeval_flow._service._resolve import Resolution
    from dataeval_flow.config import PipelineConfig
    from dataeval_flow.config._schemas import TaskConfig
    from dataeval_flow.workflows._preset import Preset, StructuralChoice

__all__ = [
    "SUPPORTED_OPERATIONS",
    "DecomposeAnswer",
    "DecomposeError",
    "DecomposeReady",
    "DecomposeReason",
    "DecomposeRequest",
    "DecomposeUnsupported",
    "prepare",
]

SUPPORTED_OPERATIONS: tuple[str, ...] = ("copy_definition", "inline_step")
_INDEX = r"(0|[1-9]\d*)"  # an array index as RFC 6901 writes it: no leading zeros
_DEFINITION = re.compile(rf"/workflows/{_INDEX}")
_STEP = re.compile(rf"/workflows/{_INDEX}/steps/{_INDEX}")


class DecomposeRequest(BaseModel):
    """A draft pipeline, the preset in it to decompose, and how."""

    model_config = ConfigDict(extra="forbid")

    pipeline: dict[str, Any] = Field(description="The whole draft pipeline, as `/v1/resolve` takes it.")
    expected_schema_fingerprint: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
        description="The `schema_fingerprint` the client read the schema under; a different one is "
        "`409 schema_changed`.",
    )
    operation: Literal["copy_definition", "inline_step"] = Field(
        description="`copy_definition` copies a preset entry into a new custom workflow; `inline_step` replaces one "
        "step running a preset with that preset's steps."
    )
    target_pointer: str = Field(
        description="`/workflows/<i>`, a preset entry, for `copy_definition`; `/workflows/<i>/steps/<j>`, a step "
        "running one, for `inline_step`."
    )
    new_name: str | None = Field(
        default=None, pattern=STEP_NAME_PATTERN, description="The copied workflow's name: one no workflow has."
    )
    reassign_tasks: list[str] = Field(
        default_factory=list,
        description="Tasks running the target entry to run the copy instead. None by default.",
    )


class DecomposeError(ValueError):
    """A request preparation refuses, located where a 422 locates it."""

    def __init__(self, loc: list[str | int], message: str) -> None:
        super().__init__(message)
        self.loc = loc


# A response field with a default is still always sent, so the response schema marks it required.
_ALWAYS_SENT = ConfigDict(json_schema_serialization_defaults_required=True)


class DecomposePresetRef(BaseModel):
    """The preset entry a reason concerns."""

    name: str
    type: str


class DecomposeReason(BaseModel):
    """Why a decomposition can't be prepared, where, and what would change that."""

    model_config = _ALWAYS_SENT

    code: Literal["unsupported_matrix_rewrite", "undeclared_structure", "unrepresentable_preflight"] = Field(
        description="What the reason is; branch on it."
    )
    category: Literal["implementation_gap", "plugin_contract_gap", "scope_limit"]
    message: str
    document_pointer: str = Field(description="Where in the draft the reason is, as a JSON Pointer.")
    preset: DecomposePresetRef
    missing_capability: str | None = None
    remediation: str
    details: dict[str, Any] = Field(default_factory=dict)


class DecomposeChange(BaseModel):
    """One edit the candidate makes to the draft."""

    op: Literal["add", "replace"]
    pointer: str = Field(description="Where in the candidate, as a JSON Pointer.")
    summary: str


class DecomposeRepresentationalChange(BaseModel):
    """Something the candidate represents differently from the preset, though its results mean the same."""

    code: Literal["result_type", "outputs_not_carried", "step_names", "record_criteria"]
    pointer: str
    message: str


class DecomposeNames(BaseModel):
    """What the candidate calls what the preset expanded to."""

    model_config = _ALWAYS_SENT

    workflow: str
    steps: dict[str, str] = Field(description="Each preset-local step name to its name in the candidate.")
    evaluators: dict[str, str] = Field(description="Each of the preset's evaluator names to its copy's.")
    addresses: dict[str, str] = Field(
        default_factory=dict,
        description="Each address that read the inlined step's output, such as `c.clean`, to the address that reads "
        "it now; empty for a copy.",
    )


class DecomposeStructuralDecision(BaseModel):
    """What one structural setting chose: the steps the copy keeps, and the ones it leaves out for good."""

    model_config = _ALWAYS_SENT

    setting: str
    pointer: str = Field(description="The setting in the draft's preset entry, as a JSON Pointer.")
    value: Any = Field(description="The setting's value, its default where the entry omits it.")
    written: bool = Field(description="Whether the entry writes the setting.")
    effective_value: Any = Field(default=None, description="What a `null` value stands for, where it stands for one.")
    included_steps: list[str]
    excluded_steps: list[str] = Field(description="Steps another value would add; never in the candidate.")
    rule: str
    explanation: str


class _Answer(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True, serialize_by_alias=True, json_schema_serialization_defaults_required=True
    )

    contract_schema: int = Field(default=_contract.CONTRACT_SCHEMA, alias="schema")
    schema_fingerprint: str = Field(default_factory=_contract.schema_fingerprint)
    operation: str
    target_pointer: str
    supported_operations: list[str] = Field(default_factory=lambda: list(SUPPORTED_OPERATIONS))


class DecomposeReady(_Answer):
    """A candidate pipeline to show and, once the user accepts it, apply as one edit."""

    status: Literal["ready"] = "ready"
    pipeline: dict[str, Any] = Field(description="The whole candidate, normalized as `/v1/resolve` returns a pipeline.")
    valid: bool = Field(description="Whether the whole candidate resolves; unrelated draft problems count.")
    diagnostics: list[dict[str, Any]] = Field(
        description="Every problem resolving the candidate found, as `/v1/resolve`."
    )
    names: DecomposeNames
    changes: list[DecomposeChange]
    structure: list[DecomposeStructuralDecision]
    representational_changes: list[DecomposeRepresentationalChange]


class DecomposeUnsupported(_Answer):
    """Why no candidate can be prepared: every reason found, with nothing to apply."""

    status: Literal["unsupported"] = "unsupported"
    reasons: list[DecomposeReason]


DecomposeAnswer = Annotated[DecomposeReady | DecomposeUnsupported, Field(discriminator="status")]
"""Either answer, told apart by `status`."""


def prepare(pipeline: PipelineConfig, request: DecomposeRequest) -> DecomposeReady | DecomposeUnsupported:
    """Prepare `request`'s decomposition of a preset in `pipeline`, the request's draft as loaded.

    Raises
    ------
    DecomposeError
        Where the request names nothing it can decompose, or asks for a name or task it can't have.
    ContractError
        ``too_many_nodes`` where the candidate resolves to more nodes than the service resolves.
    """
    if request.operation == "inline_step":
        from dataeval_flow._service._inline import inline

        return inline(pipeline, request)
    return _copy(pipeline, request)


def _copy(pipeline: PipelineConfig, request: DecomposeRequest) -> DecomposeReady | DecomposeUnsupported:
    from dataeval_flow.workflows._preset import expand_preset, structural_choices

    pointer = request.target_pointer
    entry, preset = _preset_entry(pipeline, pointer, _DEFINITION, "`/workflows/<index>`")
    new_name = request.new_name
    if new_name is None:
        raise DecomposeError(["body", "new_name"], "`copy_definition` needs `new_name`: the copied workflow's name.")
    if any(workflow.name == new_name for workflow in pipeline.workflows or ()):
        raise DecomposeError(["body", "new_name"], f"A workflow named '{new_name}' exists: choose another `new_name`.")
    tasks = _reassigned(pipeline, entry.name, request.reassign_tasks)
    ref = DecomposePresetRef(name=entry.name, type=entry.type)
    reasons = [*_plugin_gaps(preset, pointer, ref), *_matrix_gaps(pipeline, entry.name, tasks, ref)]
    if reasons:
        return DecomposeUnsupported(operation=request.operation, target_pointer=pointer, reasons=reasons)

    workflow, evaluators = expand_preset(entry, preset)
    taken = {existing.name for existing in pipeline.evaluators or ()}
    renamed: dict[str, str] = {}
    for evaluator in evaluators:
        renamed[evaluator.name] = _unique(f"{new_name}-{evaluator.name}", taken)
        taken.add(renamed[evaluator.name])
    document = pipeline.model_dump(mode="json", by_alias=True)
    copied = workflow.model_dump(mode="json")
    copied["name"] = new_name
    for step in copied["steps"]:
        if "evaluator" in step:
            step["evaluator"] = renamed[step["evaluator"]]
    copied["derived_from"] = [
        {
            "type": entry.type,
            "entry": entry.name,
            "settings": entry.model_dump(mode="json", by_alias=True),
            "flow_version": __version__,
        }
    ]
    workflows = document.setdefault("workflows", []) or []
    document["workflows"] = workflows
    workflows.append(copied)
    pool = document.get("evaluators") or []
    document["evaluators"] = pool
    first = len(pool)
    for evaluator in evaluators:
        pool.append({**evaluator.model_dump(mode="json", by_alias=True), "name": renamed[evaluator.name]})
    changes = [
        DecomposeChange(
            op="add",
            pointer=f"/workflows/{len(workflows) - 1}",
            summary=f"Workflow '{new_name}', copied from preset '{entry.name}' ({entry.type}).",
        ),
        *(
            DecomposeChange(
                op="add",
                pointer=f"/evaluators/{first + offset}",
                summary=f"Evaluator '{renamed[evaluator.name]}', the copy's own '{evaluator.name}'.",
            )
            for offset, evaluator in enumerate(evaluators)
        ),
    ]
    for index, task in tasks:
        document["tasks"][index]["workflow"] = new_name
        changes.append(
            DecomposeChange(
                op="replace", pointer=f"/tasks/{index}/workflow", summary=f"Task '{task.name}' runs '{new_name}'."
            )
        )

    candidate, resolution = _resolve_candidate(document, {index for index, _ in tasks})
    copy_pointer = f"/workflows/{len(workflows) - 1}"
    return DecomposeReady(
        operation=request.operation,
        target_pointer=pointer,
        pipeline=candidate.model_dump(mode="json", by_alias=True),
        valid=not resolution.diagnostics,
        diagnostics=resolution.diagnostics,
        names=DecomposeNames(
            workflow=new_name, steps={step["name"]: step["name"] for step in copied["steps"]}, evaluators=renamed
        ),
        changes=changes,
        structure=[_decision(choice, pointer) for choice in structural_choices(entry, preset) or ()],
        representational_changes=_representational(preset, entry, new_name, copy_pointer),
    )


def _resolve_candidate(document: dict[str, Any], also: set[int]) -> tuple[PipelineConfig, Resolution]:
    """`document` loaded and resolved as `/v1/resolve` resolves a draft, running its enabled tasks and those at the
    indices `also` names; refused with ``too_many_nodes`` past the service's bound."""
    from dataeval_flow._service._resolve import resolve
    from dataeval_flow.config import PipelineConfig

    candidate = PipelineConfig.model_validate(document, context={"graphs": False})
    # The enabled tasks, as `/v1/resolve` selects them, and `also`: any may be none, and `resolve` still diagnoses every
    # workflow.
    selected = [task for index, task in enumerate(candidate.tasks or ()) if task.enabled or index in also]
    resolution = resolve(candidate, selected)
    limit = _contract.MAX_RESOLVED_NODES
    if resolution.node_count > limit:
        raise _contract.ContractError(
            413,
            "too_many_nodes",
            f"The candidate resolves to {resolution.node_count} nodes; the service resolves at most {limit}.",
            limit=limit,
            count=resolution.node_count,
        )
    return candidate, resolution


def _preset_entry(
    pipeline: PipelineConfig, pointer: str, shape: re.Pattern[str], written: str
) -> tuple[Any, type[Preset]]:
    """The preset entry `pointer` names; refused where it names no workflow, or one that isn't a preset."""
    from dataeval_flow.workflows._preset import preset_of

    match = shape.fullmatch(pointer)
    if match is None:
        raise DecomposeError(["body", "target_pointer"], f"`{pointer}` isn't {written}.")
    workflows = list(pipeline.workflows or ())
    index = int(match.group(1))
    if index >= len(workflows):
        raise DecomposeError(
            ["body", "target_pointer"], f"`{pointer}` names no workflow: the draft has {len(workflows)}."
        )
    entry = workflows[index]
    preset = preset_of(entry)
    if preset is None:
        raise DecomposeError(
            ["body", "target_pointer"],
            f"`{pointer}` is workflow '{entry.name}', which is no preset: only a preset entry decomposes.",
        )
    return entry, preset


def _reassigned(pipeline: PipelineConfig, workflow: str, names: list[str]) -> list[tuple[int, TaskConfig]]:
    """Each task `names` lists, with its index; refused where one is unknown, repeated, or runs another workflow."""
    tasks = list(pipeline.tasks or ())
    found: list[tuple[int, TaskConfig]] = []
    for position, name in enumerate(names):
        loc: list[str | int] = ["body", "reassign_tasks", position]
        if name in names[:position]:
            raise DecomposeError(loc, f"`reassign_tasks` names '{name}' more than once.")
        index = next((index for index, task in enumerate(tasks) if task.name == name), None)
        if index is None:
            raise DecomposeError(loc, f"No task is named '{name}'.")
        task = tasks[index]
        if task.kind != "workflow" or task.workflow != workflow:
            raise DecomposeError(loc, f"Task '{name}' doesn't run workflow '{workflow}', so it can't run its copy.")
        found.append((index, task))
    return found


def _plugin_gaps(preset: type[Preset], pointer: str, ref: DecomposePresetRef) -> list[DecomposeReason]:
    """What a preset's author has to declare before it decomposes."""
    from dataeval_flow.workflows._preset import Preset

    reasons: list[DecomposeReason] = []
    if preset.structure is None:
        reasons.append(
            DecomposeReason(
                code="undeclared_structure",
                category="plugin_contract_gap",
                message=f"Preset type '{ref.type}' declares no `structure`, so Flow can't say which of its settings "
                "decide its steps.",
                document_pointer=pointer,
                preset=ref,
                missing_capability="preset_structure_declaration",
                remediation="The preset's author declares `structure` on its class: a "
                "`dataeval_flow.workflows.Structural` per setting that adds or removes steps, or `()` where none does.",
            )
        )
    if getattr(preset.preflight, "__func__", None) is not Preset.preflight.__func__:  # type: ignore[attr-defined]
        reasons.append(
            DecomposeReason(
                code="unrepresentable_preflight",
                category="plugin_contract_gap",
                message=f"Preset type '{ref.type}' refuses runs in Python (`preflight`), which no custom workflow "
                "can carry.",
                document_pointer=pointer,
                preset=ref,
                missing_capability="portable_workflow_preflight",
                remediation="The preset's author declares its refusals in its chain's `preflight` rather than "
                "overriding `Preset.preflight`.",
            )
        )
    return reasons


def _matrix_gaps(
    pipeline: PipelineConfig,
    workflow: str,
    tasks: list[tuple[int, TaskConfig]],
    ref: DecomposePresetRef,
    *,
    inlined: str | None = None,
    moved: Collection[str] = (),
) -> list[DecomposeReason]:
    """A reason per matrix key of `tasks` that varies preset entry `workflow`'s settings, or is among `moved`, at the
    key in its grid (spec §5.3, §6.2). Inlining step `inlined`, the reasons speak of inlining it rather than of
    reassigning a task, and `moved` names the keys that sweep an address the inlining moves."""
    from dataeval_flow._matrix._keys import canonical

    reasons: list[DecomposeReason] = []
    for index, task in tasks:
        if task.matrix is None:
            continue
        grids = (
            [("", task.matrix)]
            if isinstance(task.matrix, dict)
            else [(f"/{g}", grid) for g, grid in enumerate(task.matrix)]
        )
        for grid_pointer, grid in grids:
            for key in grid:
                target = canonical(key, task, pipeline)
                of_preset = target.pool == "workflows" and target.entry == workflow
                if not of_preset and key not in moved:
                    continue
                if inlined is None:
                    message = (
                        f"Task '{task.name}' varies `{key}`, a setting of preset '{workflow}'. The copy has no such "
                        "setting, and Flow doesn't rewrite matrix keys onto the steps a preset expands to."
                    )
                    remediation = (
                        f"Keep task '{task.name}' on the preset, or take `{key}` out of its matrix before reassigning "
                        "it."
                    )
                else:
                    message = (
                        f"Task '{task.name}' varies `{key}`, a setting of preset '{workflow}'. The inlined steps have "
                        "no such setting, and Flow doesn't rewrite matrix keys onto the steps a preset expands to."
                        if of_preset
                        else f"Task '{task.name}' varies `{key}`, which reads an output of step '{inlined}'. Inlining "
                        "moves that output to the inlined steps, and Flow doesn't rewrite the addresses a matrix "
                        "sweeps."
                    )
                    remediation = (
                        f"Keep step '{inlined}' running the preset, or take `{key}` out of task '{task.name}''s matrix "
                        "before inlining the step."
                    )
                escaped = key.replace("~", "~0").replace("/", "~1")
                reasons.append(
                    DecomposeReason(
                        code="unsupported_matrix_rewrite",
                        category="scope_limit",
                        message=message,
                        document_pointer=f"/tasks/{index}/matrix{grid_pointer}/{escaped}",
                        preset=ref,
                        remediation=remediation,
                        details={"task": task.name, "key": key},
                    )
                )
    return reasons


def _unique(name: str, taken: set[str]) -> str:
    """`name`, or `name-2`, `name-3`, … : the first that `taken` doesn't hold."""
    candidate, number = name, 2
    while candidate in taken:
        candidate, number = f"{name}-{number}", number + 1
    return candidate


def _decision(choice: StructuralChoice, entry_pointer: str) -> DecomposeStructuralDecision:
    """`choice` as the response reports it, with a sentence saying what it chose."""
    value = choice.value
    shown = None if isinstance(value, dict | list) else f"`{json.dumps(value)}`"
    stands_for = f"`{choice.effective}`" if value is None and choice.effective is not None else None
    if choice.written:
        state = (shown or "set") + (f", so {stands_for}" if stands_for else "")
    elif stands_for:
        state = f"unset, so {stands_for}"
    else:
        state = f"unset, so its default {shown}" if shown else "unset, so it takes its default"
    parts = []
    if choice.included:
        parts.append(f"{', '.join(f'`{step}`' for step in choice.included)} included")
    if choice.excluded:
        parts.append(f"{', '.join(f'`{step}`' for step in choice.excluded)} not included")
    said = f"`{choice.setting}` is {state}" + (f": {'; '.join(parts)}" if parts else "") + "."
    return DecomposeStructuralDecision(
        setting=choice.setting,
        pointer=f"{entry_pointer}/{choice.setting.replace('.', '/')}",
        value=value,
        written=choice.written,
        effective_value=choice.effective if value is None else None,
        included_steps=list(choice.included),
        excluded_steps=list(choice.excluded),
        rule=choice.rule,
        explanation=f"{said} {choice.rule}",
    )


def _representational(
    preset: type[Preset], entry: Any, new_name: str, pointer: str
) -> list[DecomposeRepresentationalChange]:
    """What the copy represents differently from the preset."""
    changes = [
        DecomposeRepresentationalChange(
            code="result_type",
            pointer=pointer,
            message=f"The copy's results carry type '{new_name}', a custom workflow, where the preset's carry "
            f"'{entry.type}'.",
        )
    ]
    if preset.output_ports():
        names = ", ".join(f"`{port.name}`" for port in preset.output_ports())
        changes.append(
            DecomposeRepresentationalChange(
                code="outputs_not_carried",
                pointer=pointer,
                message=f"Preset '{entry.name}' hands {names} to a workflow running it as a step; a custom workflow "
                "can't run as a step, so the copy declares no outputs. Steps running the preset keep it.",
            )
        )
    return changes
