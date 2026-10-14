"""A pipeline's resolved graphs for an editor: each task's steps, ports and bindings, and where they don't connect
(builder spec §5). Reads no data and queues nothing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dataeval_flow._chain._graph import (
    ChainGraph,
    GraphError,
    StepSpec,
    one_step_graph,
    output_type,
    resolve_graph,
    task_problems,
)
from dataeval_flow.steps._catalog import class_path, dataset_kinds, input_shape, port_entry
from dataeval_flow.steps._port import Port
from dataeval_flow.steps._step import Transform
from dataeval_flow.steps._workflow import CustomWorkflowConfig

if TYPE_CHECKING:
    from dataeval_flow.config import PipelineConfig
    from dataeval_flow.config._schemas import TaskConfig

__all__ = ["Resolution", "resolve"]


@dataclass(frozen=True)
class Resolution:
    """What resolving a pipeline found: the graphs to draw, and every problem, located."""

    tasks: list[dict[str, Any]]
    workflows: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]

    @property
    def node_count(self) -> int:
        """How many nodes the graphs hold, for the service's bound."""
        return sum(len(graph["nodes"]) for graph in (*self.tasks, *self.workflows))


def resolve(pipeline: PipelineConfig, selected: Sequence[TaskConfig]) -> Resolution:
    """Resolve `pipeline` as the config check does, reporting each problem where the check raises the first (builder
    spec §5.3): what it refuses, this reports; what it accepts, this reports nothing about."""
    pointers = _Pointers(pipeline)
    graphs: dict[str, ChainGraph] = {}
    failed: dict[str, GraphError] = {}
    for entry in pipeline.workflows or ():
        graph, error = _entry_graph(entry, pipeline)
        if graph is not None:
            graphs[entry.name] = graph
        if error is not None:
            failed[entry.name] = error
    runs = {task.name: task.workflow for task in selected if task.kind == "workflow"}
    diagnostics: list[dict[str, Any]] = []
    for name, error in failed.items():
        owners = [task for task, workflow in runs.items() if workflow == name] or [None]
        diagnostics.extend(_diagnostic(error, name, owner, pointers) for owner in owners)
    resolved = {name: graph for name, graph in graphs.items() if name not in failed}
    problems: Sequence[tuple[str | None, str]]
    try:
        problems = task_problems(pipeline, resolved)
    except ValueError as error:  # what the config check refuses outside the graph walk
        problems = [(None, str(error))]
    diagnostics.extend(_task_problem(task, message, pointers) for task, message in problems)
    tasks = [_task(task, pipeline, graphs, failed, pointers) for task in selected]
    used = set(runs.values())
    workflows = [
        {"name": name, **_body(name, pointers.entries[name], graph, name not in failed, pointers)}
        for name, graph in graphs.items()
        if name not in used
    ]
    return Resolution(tasks, workflows, diagnostics)


def _entry_graph(entry: Any, pipeline: PipelineConfig) -> tuple[ChainGraph | None, GraphError | None]:
    """A workflow entry's graph with no slot keys, as the config check builds it; no graph for one that isn't a
    custom workflow or a preset."""
    from dataeval_flow.workflows._preset import expand_preset, preset_of

    try:
        if isinstance(entry, CustomWorkflowConfig):
            return resolve_graph(entry, pipeline)
        preset = preset_of(entry)
        if preset is None:
            return None, None
        chain, evaluators = expand_preset(entry, preset)
        return resolve_graph(chain, pipeline, evaluators=evaluators)
    except ValueError as error:  # the check refuses any ValueError, a preset's chain that won't build among them
        return ChainGraph(entry.name, (), ()), GraphError(str(error))


class _Pointers:
    """Where each named thing is in the pipeline, as JSON Pointers into its normalized form.

    Names match `STEP_NAME_PATTERN` and pool names, which hold neither `/` nor `~`, so no pointer needs escaping.
    """

    def __init__(self, pipeline: PipelineConfig) -> None:
        self.entries = {entry.name: entry for entry in pipeline.workflows or ()}
        self.workflows = {entry.name: index for index, entry in enumerate(pipeline.workflows or ())}
        self.evaluators = {entry.name: index for index, entry in enumerate(pipeline.evaluators or ())}
        self.tasks = {task.name: index for index, task in enumerate(pipeline.tasks or ())}

    def step(self, workflow: str, name: str | None) -> str | None:
        """Step `name` of custom workflow `workflow`; ``None`` where the document holds no such step."""
        entry = self.entries.get(workflow)
        if name is None or not isinstance(entry, CustomWorkflowConfig):
            return None
        index = next((index for index, step in enumerate(entry.steps) if step.name == name), None)
        return None if index is None else f"/workflows/{self.workflows[workflow]}/steps/{index}"

    def evaluator(self, name: str) -> str | None:
        index = self.evaluators.get(name)
        return None if index is None else f"/evaluators/{index}"

    def origin(self, owner: Any, spec: StepSpec) -> dict[str, Any]:
        """Where `spec` is written (builder spec §5.4). `owner` is the custom workflow or preset entry it comes from,
        or ``None`` for an evaluator task's one step; a preset's steps are generated, never pointed into."""
        config: Any = spec.config
        if owner is None:
            return {"step_pointer": None, "definition_pointer": self.evaluator(config.name), "generated": False}
        if not isinstance(owner, CustomWorkflowConfig):
            return {
                "step_pointer": None,
                "definition_pointer": f"/workflows/{self.workflows[owner.name]}",
                "generated": True,
            }
        outer, _, inner = spec.name.partition("/")
        step = self.step(owner.name, outer)
        if inner:
            target = next(entry.target for entry in owner.steps if entry.name == outer)
            return {
                "step_pointer": step,
                "definition_pointer": f"/workflows/{self.workflows[target]}",
                "generated": True,
            }
        definition = self.evaluator(config.name) if spec.kind == "evaluator" else None
        return {"step_pointer": step, "definition_pointer": definition, "generated": False}


def _diagnostic(error: GraphError, workflow: str, owner: str | None, pointers: _Pointers) -> dict[str, Any]:
    step = pointers.step(workflow, error.step)
    document = f"{step}/{error.port}" if step is not None and error.port is not None else step
    return {
        "code": error.code,
        "severity": "error",
        "message": str(error),
        "document_pointer": document or f"/workflows/{pointers.workflows[workflow]}",
        "task": owner,
        "node_id": None if error.step is None else f"{owner or workflow}:{error.step}",
        "port": error.port,
    }


def _task_problem(task: str | None, message: str, pointers: _Pointers) -> dict[str, Any]:
    return {
        "code": "task_problem",
        "severity": "error",
        "message": message,
        "document_pointer": "/tasks" if task is None else f"/tasks/{pointers.tasks[task]}",
        "task": task,
        "node_id": None,
        "port": None,
    }


def _task(
    task: TaskConfig,
    pipeline: PipelineConfig,
    graphs: dict[str, ChainGraph],
    failed: dict[str, GraphError],
    pointers: _Pointers,
) -> dict[str, Any]:
    names = task.source_names
    owner: Any = None
    if task.kind == "evaluator":
        instance = next(entry for entry in pipeline.evaluators or () if entry.name == task.workflow)
        graph, complete = one_step_graph(task, instance, names), True
    else:
        owner = pointers.entries[task.workflow]
        graph, complete = graphs[owner.name], owner.name not in failed
        slot = owner.list_slot if isinstance(owner, CustomWorkflowConfig) else None
        if complete and task.matrix is None and slot is not None and owner.binding_problem(len(names)) is None:
            bound = tuple(names[len(owner.single_slots) :])
            try:
                graph, error = resolve_graph(owner, pipeline, slot_keys={slot.name: bound})
            except ValueError as raised:  # as in `_entry_graph`
                error = raised
            complete = error is None  # a binding problem, which `task_problems` reports
    return {
        "name": task.name,
        "kind": task.kind,
        "workflow": task.workflow,
        "source_bindings": _source_bindings(task, graph, pointers),
        **_body(task.name, owner, graph, complete, pointers),
        "matrix": _matrix(task, pipeline),
    }


def _source_bindings(task: TaskConfig, graph: ChainGraph, pointers: _Pointers) -> list[dict[str, Any]]:
    """Which of `task`'s sources each input slot reads: one each in order, the last list slot the rest."""
    names = task.source_names
    pointer = f"/tasks/{pointers.tasks[task.name]}/sources"
    return [
        {
            "slot": slot.name,
            "list": slot.is_list,
            "sources": names[index:] if slot.is_list else names[index : index + 1],
            "document_pointer": pointer,
        }
        for index, slot in enumerate(graph.slots)
    ]


def _body(name: str, owner: Any, graph: ChainGraph, complete: bool, pointers: _Pointers) -> dict[str, Any]:
    return {
        "nodes": [_node(name, owner, spec, pointers) for spec in graph.steps],
        "execution_order": [f"{name}:{spec.name}" for spec in graph.steps],
        "output_aliases": [_alias(name, graph, alias, target) for alias, target in graph.aliases.items()],
        "complete": complete,
    }


def _alias(name: str, graph: ChainGraph, alias: str, target: str) -> dict[str, Any]:
    spec, port = next(
        (spec, port) for spec in graph.steps for port in spec.outputs if spec.output_address(port) == target
    )
    return {"address": alias, "node_id": f"{name}:{spec.name}", "port": port.name}


def _node(name: str, owner: Any, spec: StepSpec, pointers: _Pointers) -> dict[str, Any]:
    config: Any = spec.config
    empty = spec.impl.empty_outputs(config) if issubclass(spec.impl, Transform) else frozenset()
    return {
        "node_id": f"{name}:{spec.name}",
        "path": [name, *spec.name.split("/")],
        "step": {"kind": spec.kind, "type": spec.type},
        "origin": pointers.origin(owner, spec),
        "settings": config.model_dump(mode="json", by_alias=True),
        "inputs": [
            {
                **port_entry(binding.port, input_shape(spec.impl, binding.port)).model_dump(mode="json", by_alias=True),
                "addresses": [str(address) for address in binding.addresses],
            }
            for binding in spec.bindings
        ],
        "outputs": [_output(spec, port, empty) for port in spec.outputs],
        "optional": spec.optional,
        "broadcast": spec.broadcast,
    }


def _output(spec: StepSpec, port: Port, empty: frozenset[str]) -> dict[str, Any]:
    value = output_type(spec, port)
    unavailable = port.name in empty
    return {
        "port": port.name,
        "type": str(port.type),
        "list": value.is_list,
        "classes": [class_path(cls) for cls in port.classes],
        "kinds": dataset_kinds(port),
        "address": spec.output_address(port),
        "keys": None if value.keys is None else list(value.keys),
        "availability": "unavailable" if unavailable else "available",
        "reason": f"`{spec.name}`'s settings leave `{port.name}` empty." if unavailable else None,
    }


def _matrix(task: TaskConfig, pipeline: PipelineConfig) -> dict[str, Any] | None:
    """A matrix task's runs, numbered from 1, as its matrix expands; the graph above is the task as written."""
    if task.matrix is None:
        return None
    from dataeval_flow._matrix._build import expand_matrix

    return {
        "runs": [
            {"number": plan.number, "label": plan.label, "values": plan.values}
            for plan in expand_matrix(task, pipeline)
        ]
    }
