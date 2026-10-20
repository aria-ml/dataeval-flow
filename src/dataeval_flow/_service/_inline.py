"""Preparing a preset step's inlining for an editor: the step replaced by its preset's steps, written as the runtime
splices them, with a scope keeping what the preset declared over them (preset decomposition spec §6.2–§6.3). Reads no
data, stores nothing and queues nothing."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from dataeval_flow import __version__
from dataeval_flow._service._decompose import (
    _STEP,
    DecomposeChange,
    DecomposeError,
    DecomposeNames,
    DecomposePresetRef,
    DecomposeReady,
    DecomposeReason,
    DecomposeRepresentationalChange,
    DecomposeUnsupported,
    _decision,
    _matrix_gaps,
    _plugin_gaps,
    _resolve_candidate,
    _unique,
)
from dataeval_flow.config._schemas._matrix import MatrixRange
from dataeval_flow.steps._address import Address, parse_address

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from dataeval_flow._chain._graph import ChainGraph
    from dataeval_flow._chain._presets import Splice
    from dataeval_flow._service._decompose import DecomposeRequest
    from dataeval_flow.config import PipelineConfig
    from dataeval_flow.config._schemas import TaskConfig
    from dataeval_flow.steps._workflow import CustomWorkflowConfig, StepEntry
    from dataeval_flow.workflows._preset import Preset

__all__ = ["inline"]

# What a preset's expansion declares beyond its steps that a scope carries; its `ontology` never reached the envelope.
_DECLARED = ("groups", "reference", "record", "verdict", "next_steps", "preflight")


def inline(pipeline: PipelineConfig, request: DecomposeRequest) -> DecomposeReady | DecomposeUnsupported:
    """Prepare `request`'s inlining of a preset step of a custom workflow in `pipeline`, the request's draft as loaded:
    the step's preset's steps as ordinary steps named `<step>-<inner>`, reading what the runtime splice reads, every
    later read of the step's outputs following them, and a scope keeping what the preset declared over them, or only
    its start where an input reads a step's output, which can hold nothing.

    Raises
    ------
    DecomposeError
        Where the request takes another operation's fields, names no step of a custom workflow running a preset, or
        names one that doesn't resolve.
    ContractError
        ``too_many_nodes`` where the candidate resolves to more nodes than the service resolves.
    """
    from dataeval_flow.workflows._preset import expand_preset, structural_choices

    pointer = request.target_pointer
    index, position, parent, entry, preset = _target(pipeline, request)
    step = parent.steps[position]
    graph, splice = _resolved(parent, step, pipeline)
    workflow, evaluators = expand_preset(entry, preset)
    inlined = [inner.model_dump(mode="json") for inner in workflow.steps]
    taken = {slot.name for slot in parent.inputs} | {scope.name for scope in parent.scopes}
    taken |= {other.name for other in parent.steps if other is not step}
    renamed: dict[str, str] = {}
    for inner in inlined:
        renamed[inner["name"]] = _unique(f"{step.name}-{inner['name']}", taken)
        taken.add(renamed[inner["name"]])
    # Each read of the step's outputs, `c.clean`, to the inner output that holds it under its new name, `c-clean`.
    addresses: dict[str, str] = {}
    for alias, target in graph.aliases.items():
        if alias.startswith(f"{step.name}."):
            made = parse_address(target.removeprefix(splice.prefix))
            addresses[alias] = str(replace(made, name=renamed[made.name]))

    ref = DecomposePresetRef(name=entry.name, type=entry.type)
    reasons: list[DecomposeReason] = _plugin_gaps(preset, pointer, ref)
    for number, task in _running(pipeline, parent.name):
        moved = _swept_moves(task, parent, addresses)
        reasons += _matrix_gaps(pipeline, entry.name, [(number, task)], ref, inlined=step.name, moved=moved)
    if reasons:
        return DecomposeUnsupported(operation=request.operation, target_pointer=pointer, reasons=reasons)

    pool = {existing.name for existing in pipeline.evaluators or ()}
    copies: dict[str, str] = {}
    for evaluator in evaluators:
        copies[evaluator.name] = _unique(f"{parent.name}-{step.name}-{evaluator.name}", pool)
        pool.add(copies[evaluator.name])
    _write_inlined(inlined, graph, splice, renamed, copies)
    base = f"/workflows/{index}"
    document = pipeline.model_dump(mode="json", by_alias=True)
    written = document["workflows"][index]
    changes = [
        DecomposeChange(
            op="add" if offset else "replace",
            pointer=f"{base}/steps/{position + offset}",
            summary=f"Step '{new}', preset '{entry.name}''s `{old}`"
            + (f", in place of step '{step.name}'." if not offset else "."),
        )
        for offset, (old, new) in enumerate(renamed.items())
    ]
    changes += _splice_in(written, position, inlined, step.name, _outer(addresses), base)
    representational = [
        DecomposeRepresentationalChange(
            code="step_names",
            pointer=base,
            message=f"Results name step '{step.name}''s steps `{step.name}-…`, where they named them "
            f"`{step.name}/…`, and the envelope no longer records preset '{entry.name}' under `presets`.",
        )
    ]
    declares = workflow.declared_chain() is not None
    # A slot bound to a step's output can hold nothing at run time; the splice then skips its steps together.
    held = {slot.name for slot in parent.inputs}
    if declares or any(parse_address(text).name not in held for text in splice.slots.values()):
        scopes = written.setdefault("scopes", [])
        scope = _scope(workflow.model_dump(mode="json"), inlined, renamed, dict(splice.slots))
        scope["name"] = _unique(step.name, {item["name"] for item in (*scopes, *written["steps"])})
        scope_pointer = f"{base}/scopes/{len(scopes)}"
        scopes.append(scope)
        changes.append(
            DecomposeChange(
                op="add",
                pointer=scope_pointer,
                summary=f"Scope '{scope['name']}', judging the inlined steps on step '{step.name}''s inputs as preset "
                f"'{entry.name}' ({entry.type}) judged them."
                if declares
                else f"Scope '{scope['name']}', keeping preset '{entry.name}' ({entry.type})'s start on step "
                f"'{step.name}''s inputs: the inlined steps skip together when an input held nothing.",
            )
        )
        if "record" in scope:
            representational.append(
                DecomposeRepresentationalChange(
                    code="record_criteria",
                    pointer=scope_pointer,
                    message=f"Scope '{scope['name']}''s record lists its blocking and accepted checks, where preset "
                    f"'{entry.name}''s also listed each check's settings; the inlined steps carry those.",
                )
            )
    origins = written.setdefault("derived_from", [])
    origins.append(
        {
            "type": entry.type,
            "entry": entry.name,
            "settings": entry.model_dump(mode="json", by_alias=True),
            "steps": list(renamed.values()),
            "flow_version": __version__,
        }
    )
    changes.append(
        DecomposeChange(
            op="add",
            pointer=f"{base}/derived_from/{len(origins) - 1}",
            summary=f"Where the inlined steps came from: preset '{entry.name}' ({entry.type}), as it stood.",
        )
    )
    evaluator_pool = document.get("evaluators") or []
    document["evaluators"] = evaluator_pool
    for evaluator in evaluators:
        evaluator_pool.append({**evaluator.model_dump(mode="json", by_alias=True), "name": copies[evaluator.name]})
        changes.append(
            DecomposeChange(
                op="add",
                pointer=f"/evaluators/{len(evaluator_pool) - 1}",
                summary=f"Evaluator '{copies[evaluator.name]}', the inlined steps' own '{evaluator.name}'.",
            )
        )

    candidate, resolution = _resolve_candidate(document, set())
    return DecomposeReady(
        operation=request.operation,
        target_pointer=pointer,
        pipeline=candidate.model_dump(mode="json", by_alias=True),
        valid=not resolution.diagnostics,
        diagnostics=resolution.diagnostics,
        names=DecomposeNames(workflow=parent.name, steps=renamed, evaluators=copies, addresses=addresses),
        changes=changes,
        structure=[
            _decision(choice, f"/workflows/{list(pipeline.workflows or ()).index(entry)}")
            for choice in structural_choices(entry, preset) or ()
        ],
        representational_changes=representational,
    )


def _target(
    pipeline: PipelineConfig, request: DecomposeRequest
) -> tuple[int, int, CustomWorkflowConfig, Any, type[Preset]]:
    """The parent's index, the step's position in it, the parent, and the preset entry the step runs and its preset;
    refused where the request takes another operation's fields or names no step of a custom workflow running a
    preset."""
    from dataeval_flow.steps._workflow import CustomWorkflowConfig
    from dataeval_flow.workflows._preset import preset_of

    pointer = request.target_pointer
    for field in ("new_name", "reassign_tasks"):
        if getattr(request, field):
            raise DecomposeError(["body", field], f"`inline_step` takes no `{field}`.")
    match = _STEP.fullmatch(pointer)
    if match is None:
        raise DecomposeError(["body", "target_pointer"], f"`{pointer}` isn't `/workflows/<index>/steps/<index>`.")
    workflows = list(pipeline.workflows or ())
    index, position = int(match.group(1)), int(match.group(2))
    parent = workflows[index] if index < len(workflows) else None
    if not isinstance(parent, CustomWorkflowConfig) or position >= len(parent.steps):
        raise DecomposeError(["body", "target_pointer"], f"`{pointer}` names no step of a custom workflow.")
    step = parent.steps[position]
    entry = next((item for item in workflows if step.kind == "workflow" and item.name == step.target), None)
    preset = preset_of(entry)
    if entry is None or preset is None:
        raise DecomposeError(["body", "target_pointer"], f"Step '{step.name}' runs no preset.")
    return index, position, parent, entry, preset


def _resolved(parent: CustomWorkflowConfig, step: StepEntry, pipeline: PipelineConfig) -> tuple[ChainGraph, Splice]:
    """`parent`'s graph and `step`'s splice in it; refused where the graph stops at or before the step, whose bindings
    are then unknown."""
    from dataeval_flow._chain._graph import resolve_graph

    try:
        graph, error = resolve_graph(parent, pipeline)
    except ValueError as raised:  # a preset's chain that won't build, as `/v1/resolve` takes it
        graph, error = None, raised
    splice = next((item for item in graph.splices if item.name == step.name), None) if graph is not None else None
    if graph is None or splice is None:
        raise DecomposeError(
            ["body", "target_pointer"],
            f"Step '{step.name}' of workflow '{parent.name}' doesn't resolve, so its steps can't be inlined: {error}",
        )
    return graph, splice


def _running(pipeline: PipelineConfig, workflow: str) -> list[tuple[int, TaskConfig]]:
    """Each task running `workflow`, enabled or not, with its index."""
    return [
        (number, task)
        for number, task in enumerate(pipeline.tasks or ())
        if task.kind == "workflow" and task.workflow == workflow
    ]


def _write_inlined(
    inlined: list[dict[str, Any]],
    graph: ChainGraph,
    splice: Splice,
    renamed: Mapping[str, str],
    copies: Mapping[str, str],
) -> None:
    """Rewrite `inlined`, the preset's steps as written, in place as the runtime splice runs them: renamed, each
    evaluator its copy, each address moved as `_inner` moves it, and `optional` and `extractor` where the splice gave
    them and the step has none."""
    move = _inner(splice.slots, renamed)
    specs = {spec.name: spec for spec in graph.steps}
    for inner in inlined:
        spec = specs[f"{splice.prefix}{inner['name']}"]
        _move_addresses(inner, move)
        inner["name"] = renamed[inner["name"]]
        if "evaluator" in inner:
            inner["evaluator"] = copies[inner["evaluator"]]
        if spec.optional and not inner.get("optional"):
            inner["optional"] = True
        if spec.extractor is not None and inner.get("extractor") is None:
            inner["extractor"] = spec.extractor


def _splice_in(
    written: dict[str, Any],
    position: int,
    inlined: list[dict[str, Any]],
    name: str,
    move: Callable[[Address], Address],
    base: str,
) -> list[DecomposeChange]:
    """Put `inlined` in place of step `name` at `position` of `written`, the parent as written at `base`, and move each
    later step's reads and each scope's inputs with `move`; what that changed in them."""
    later = written["steps"][position + 1 :]
    written["steps"] = [*written["steps"][:position], *inlined, *later]
    changes: list[DecomposeChange] = []
    for offset, reader in enumerate(later, start=position + len(inlined)):
        for field in _move_addresses(reader, move):
            at = f"{base}/steps/{offset}/{field}"
            changes.append(
                DecomposeChange(
                    op="add",
                    pointer=at,
                    summary=f"Step '{reader['name']}' keeps the keys it derived from the addresses it read: "
                    + ", ".join(f"`{key}`" for key in reader["keys"])
                    + ".",
                )
                if field == "keys"
                else DecomposeChange(
                    op="replace",
                    pointer=at,
                    summary=f"Step '{reader['name']}''s `{field}` reads the inlined steps where it read step '{name}'.",
                )
            )
    for offset, scope in enumerate(written.get("scopes", [])):
        inputs = {slot: _move_text(text, move) for slot, text in scope["inputs"].items()}
        if inputs != scope["inputs"]:
            scope["inputs"] = inputs
            changes.append(
                DecomposeChange(
                    op="replace",
                    pointer=f"{base}/scopes/{offset}/inputs",
                    summary=f"Scope '{scope['name']}''s `inputs` read the inlined steps where they read step '{name}'.",
                )
            )
    return changes


def _scope(
    expanded: dict[str, Any], inlined: list[dict[str, Any]], renamed: Mapping[str, str], inputs: dict[str, str]
) -> dict[str, Any]:
    """The scope keeping what `expanded`, a preset's expansion as written, declares over `inlined`, its steps as
    inlined: on `inputs`, the slots as the step bound them, with each `accepted` check step under its new name. A key
    that is also a check type the steps run stays as written: types win, as the load-time check reads them first."""
    scope: dict[str, Any] = {
        "inputs": inputs,
        "steps": list(renamed.values()),
        **{key: expanded[key] for key in _DECLARED if key in expanded},
    }
    if "verdict" in scope:
        checks = {inner["check"] for inner in inlined if "check" in inner}
        accepted: dict[str, str] = {}
        for key, reason in scope["verdict"]["accepted"].items():
            name, bracket, element = key.partition("[")
            accepted[key if key in checks else renamed.get(name, name) + bracket + element] = reason
        scope["verdict"]["accepted"] = accepted
    return scope


def _outer(addresses: Mapping[str, str]) -> Callable[[Address], Address]:
    """The move of a read outside the inlined steps: an address whose base `addresses` names becomes the address it
    names, its key kept; any other address stays."""
    moved = {parse_address(old): parse_address(new) for old, new in addresses.items()}
    return lambda address: replace(moved[address.base], key=address.key) if address.base in moved else address


def _inner(slots: Mapping[str, str], renamed: Mapping[str, str]) -> Callable[[Address], Address]:
    """The move of a read inside the inlined steps, as the runtime splice moves it (`_chain._presets._moved`): a slot
    to the address the step bound it to, its key kept; a step of the preset to its new name, its output and key kept."""
    bound = {slot: parse_address(text) for slot, text in slots.items()}

    def move(address: Address) -> Address:
        if address.name in bound:
            outer = bound[address.name]
            return outer if address.key is None else replace(outer, key=address.key)
        return replace(address, name=renamed[address.name])

    return move


def _move_addresses(step: dict[str, Any], move: Callable[[Address], Address]) -> list[str]:
    """Move each address `step`, a step as written, reads, in place; the fields it changed.

    An evaluator or workflow step reads `input`; a transform, combine or check step reads its type's input ports, each
    an address, a list of them, or a mapping keyed by them (values and order kept). An unknown type moves nothing. A
    `collect` without `keys:` whose derived keys would change gains `keys:` with the ones it had, and `keys` is among
    the fields.
    """
    from dataeval_flow.steps._registry import inline_registry
    from dataeval_flow.steps._workflow import KIND_KEYS

    kind = next(key for key in KIND_KEYS if key in step)
    if kind in ("evaluator", "workflow"):
        fields = ["input"]
    else:
        try:
            fields = [port.name for port in inline_registry(kind).get(step[kind]).input_ports()]
        except ValueError:  # an unknown type, which the candidate's resolution reports
            return []
    keys = _collect_keys(step)
    changed: list[str] = []
    for field in fields:
        value = step.get(field)
        if isinstance(value, str):
            moved: Any = _move_text(value, move)
        elif isinstance(value, list):
            moved = [_move_text(item, move) if isinstance(item, str) else item for item in value]
        elif isinstance(value, dict):
            moved = {_move_text(key, move): item for key, item in value.items()}
        else:
            continue
        if moved != value:
            step[field] = moved
            changed.append(field)
    # ponytail: `collect` is the only built-in deriving names from address text; a plugin transform doing so would
    # need the same.
    if keys is not None and changed and _collect_keys(step) != keys:
        step["keys"] = list(keys)
        changed.append("keys")
    return changed


def _collect_keys(step: dict[str, Any]) -> tuple[str, ...] | None:
    """The keys a `collect` step without `keys:` derives from the addresses it reads; ``None`` for any other step, or
    one reading text that is no address."""
    from dataeval_flow.steps.transforms._collect import CollectConfig, collect_keys

    if step.get("transform") != "collect" or step.get("keys") is not None:
        return None
    try:
        return collect_keys(CollectConfig.model_construct(input=list(step["input"]), keys=None))
    except ValueError:
        return None


def _move_text(text: str, move: Callable[[Address], Address]) -> str:
    """`text` moved, where it is an address `move` moves; as it is otherwise."""
    try:
        address = parse_address(text)
    except ValueError:  # no address, which the candidate's resolution reports
        return text
    moved = move(address)
    return text if moved == address else str(moved)


def _swept_moves(task: TaskConfig, parent: CustomWorkflowConfig, addresses: Mapping[str, str]) -> list[str]:
    """The keys of `task`'s matrix that vary an address field of a step of `parent` whose written value, or a value the
    matrix sweeps, reads an address `addresses` moves. A key names such a field as `canonical` reads it:
    `steps.<step>.<field>…`, or the same after `workflows.<parent>.`."""
    move = _outer(addresses)
    steps = {step.name: step.model_dump(mode="json") for step in parent.steps}
    grids = [task.matrix] if isinstance(task.matrix, dict) else list(task.matrix or ())
    found: list[str] = []
    # ponytail: a key found in one grid is refused in every grid holding it; per grid if a plugin step makes it matter.
    for grid in grids:
        for key, values in grid.items():
            path = key.split(".")
            path = path[2:] if path[:2] == ["workflows", parent.name] else path
            step = steps.get(path[1]) if len(path) > 2 and path[0] == "steps" else None
            if step is None or key in found:
                continue
            field = path[2]
            swept = values.values() if isinstance(values, MatrixRange) else values
            if any(field in _move_addresses({**step, field: value}, move) for value in [step.get(field), *swept]):
                found.append(key)
    return found
