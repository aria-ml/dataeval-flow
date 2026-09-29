"""The static graph of a custom workflow: each step resolved and each address typed, before any data is read."""

__all__ = [
    "ChainGraph",
    "GraphError",
    "PortBinding",
    "StepSpec",
    "ValueType",
    "binding_problems",
    "build_graph",
    "one_step_graph",
    "task_problems",
]

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from dataeval_flow._input_spec import SourceCount
from dataeval_flow.steps._address import Address, parse_address
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import InlineStep, Step, StepKind, Transform, port_addresses
from dataeval_flow.steps._workflow import CustomWorkflowConfig, InputSlot, StepEntry

if TYPE_CHECKING:
    from dataeval_flow.config._models import PipelineConfig
    from dataeval_flow.config._schemas import TaskConfig

_ARTICLE = {
    DataType.DATASET: "a Dataset",
    DataType.OUTPUT: "an Output",
    DataType.EXPORT: "an export record",
    DataType.WORKFLOW_RESULT: "a workflow result",
    DataType.FINDINGS: "findings",
}


class GraphError(ValueError):
    """A custom workflow whose steps do not connect, found before any data is read."""


@dataclass(frozen=True)
class ValueType:
    """What an address holds, as far as it is known before running."""

    # ``builtins.type``: past ``type`` below, a bare ``type`` names the field, not the builtin.
    type: DataType
    classes: tuple[builtins.type, ...] = ()
    is_list: bool = False
    keys: tuple[str, ...] | None = None
    step: str | None = None


@dataclass(frozen=True)
class PortBinding:
    """The addresses one input port of a step reads."""

    port: Port
    addresses: tuple[Address, ...]


@dataclass(frozen=True)
class StepSpec:
    """One step, resolved: what runs, with which config, reading which addresses."""

    name: str
    kind: StepKind
    # ``builtins.type``: past ``type`` below, a bare ``type`` names the field, not the builtin.
    type: str
    impl: builtins.type[Step]
    config: BaseModel
    bindings: tuple[PortBinding, ...]
    outputs: tuple[Port, ...]
    extractor: str | None = None
    optional: bool = False
    broadcast: bool = False
    keys: tuple[str, ...] | None = None

    def addresses(self, port: str) -> tuple[Address, ...]:
        """The addresses input `port` reads; ``()`` when none."""
        return next((binding.addresses for binding in self.bindings if binding.port.name == port), ())

    def output_address(self, port: Port) -> str:
        """Where `port`'s output is stored: the step's name when it has one output, else ``name.port``."""
        return self.name if len(self.outputs) == 1 else f"{self.name}.{port.name}"


@dataclass(frozen=True)
class ChainGraph:
    """A workflow's inputs and its steps, in run order."""

    name: str
    slots: tuple[InputSlot, ...]
    steps: tuple[StepSpec, ...]
    one_step: bool = False


def build_graph(
    workflow: CustomWorkflowConfig,
    pipeline: "PipelineConfig",
    slot_keys: Mapping[str, tuple[str, ...]] | None = None,
) -> ChainGraph:
    """Resolve and type-check every step of `workflow` against `pipeline`.

    `slot_keys` holds a list slot's keys, the names of the sources a task binds to it. Without them, a key read from
    the slot, or from a list broadcast over it, is taken on trust.

    Raises
    ------
    GraphError
        Naming the step and the address that does not connect.
    """
    keys = slot_keys or {}
    types: dict[str, ValueType] = {
        slot.name: ValueType(DataType.DATASET, is_list=slot.is_list, keys=keys.get(slot.name))
        for slot in workflow.inputs
    }
    later = {entry.name for entry in workflow.steps}
    specs: dict[str, StepSpec] = {}
    empty: dict[str, frozenset[str]] = {}
    for entry in workflow.steps:
        later.discard(entry.name)
        spec = _resolve(entry, workflow, pipeline, types, later, specs, empty)
        specs[entry.name] = spec
        fixed: dict[str, tuple[str, ...]] = {}
        if issubclass(spec.impl, Transform):
            fixed = dict(spec.impl.output_keys(spec.config))
            empty[entry.name] = spec.impl.empty_outputs(spec.config)
        for port in spec.outputs:
            keys = fixed.get(port.name) if port.is_list else spec.keys
            types[spec.output_address(port)] = ValueType(
                port.type, port.classes, port.is_list or spec.broadcast, keys, step=spec.name
            )
    return ChainGraph(workflow.name, tuple(workflow.inputs), tuple(specs.values()))


def one_step_graph(task: "TaskConfig", instance: BaseModel, source_names: Sequence[str]) -> ChainGraph:
    """The graph an ``evaluator:`` task or a workflow-type task runs: its sources, read by one step named after it."""
    from dataeval_flow.evaluators._base import EvaluatorConfig
    from dataeval_flow.evaluators._registry import get_evaluator
    from dataeval_flow.workflows._registry import get_workflow

    is_evaluator = isinstance(instance, EvaluatorConfig)
    impl: type[Step] = (get_evaluator if is_evaluator else get_workflow)(instance.type)  # type: ignore[attr-defined]
    slots = tuple(InputSlot.model_construct(name=name, is_list=False) for name in source_names)
    (port,) = impl.input_ports()
    spec = StepSpec(
        name=task.name,
        kind="evaluator" if is_evaluator else "workflow",
        type=instance.type,  # type: ignore[attr-defined]
        impl=impl,
        config=instance,
        bindings=(PortBinding(port, tuple(Address(name) for name in source_names)),),
        outputs=impl.output_ports(),
    )
    return ChainGraph(task.name, slots, (spec,), one_step=True)


def task_problems(pipeline: "PipelineConfig", graphs: Mapping[str, ChainGraph]) -> list[str]:
    """Why a task cannot run the custom workflow it names: a list key its sources do not bind, a missing extractor,
    or two exports to one place. `graphs` holds each custom workflow's graph, by name, as :func:`build_graph` built
    it."""
    workflows = {
        workflow.name: workflow for workflow in pipeline.workflows or () if isinstance(workflow, CustomWorkflowConfig)
    }
    problems: list[str] = []
    owners: dict[str, str] = {export.name: f"export '{export.name}'" for export in pipeline.exports or ()}
    for task in pipeline.tasks or ():
        graph = graphs.get(task.workflow) if task.kind == "workflow" else None
        if graph is None:
            continue
        problems.extend(binding_problems(task, workflows[task.workflow], pipeline))
        problems.extend(_task_graph_problems(task, graph, owners))
    return problems


def binding_problems(task: "TaskConfig", workflow: CustomWorkflowConfig, pipeline: "PipelineConfig") -> list[str]:
    """Every address `task`'s sources leave naming nothing: a list key no source it binds to the list slot has.

    A list slot is keyed by the names of the sources a task binds to it, so its keys are known only once a task binds
    (spec §4.1). A task naming too few sources is refused by :meth:`CustomWorkflowConfig.binding_problem` instead.
    """
    slot = workflow.list_slot
    names = task.source_names
    if slot is None or workflow.binding_problem(len(names)) is not None:
        return []
    bound = tuple(names[len(workflow.single_slots) :])
    try:
        build_graph(workflow, pipeline, slot_keys={slot.name: bound})
    except GraphError as error:
        return [f"Task '{task.name}' binds sources {', '.join(bound)} to `{slot.name}`. {error}"]
    return []


def _task_graph_problems(task: "TaskConfig", graph: ChainGraph, owners: dict[str, str]) -> list[str]:
    """A task's problems running one workflow graph: missing extractors, and export destinations already claimed."""
    problems: list[str] = []
    for spec in graph.steps:
        config: Any = spec.config
        needs_extractor = spec.kind in ("evaluator", "workflow") and config.requires_extractor()
        if needs_extractor and not (spec.extractor or task.extractor):
            kinds = ", ".join(sorted(str(kind) for kind in config.wanted_kinds() if kind.needs_extractor))
            problems.append(
                f"Task '{task.name}' runs workflow '{graph.name}', whose step '{spec.name}' needs an extractor to "
                f"produce {kinds}; name one with `extractor:` on the task or the step."
            )
        if issubclass(spec.impl, Transform):
            problems.extend(_export_clashes(task, spec, spec.impl, owners))
    return problems


def _export_clashes(
    task: "TaskConfig", spec: StepSpec, impl: type[Transform[Any]], owners: dict[str, str]
) -> list[str]:
    """Every directory `impl`'s run of `spec` writes to that an earlier export or step already claimed."""
    problems: list[str] = []
    for destination in impl.destinations(spec.config, task=task.name, step=spec.name):
        owner = f"task '{task.name}' step '{spec.name}'"
        if destination in owners:
            problems.append(
                f"{owners[destination]} and {owner} both export to `datasets/{destination}`; "
                "give one of them a different `to:`."
            )
        owners.setdefault(destination, owner)
    return problems


def _resolve(
    entry: StepEntry,
    workflow: CustomWorkflowConfig,
    pipeline: "PipelineConfig",
    types: dict[str, ValueType],
    later: set[str],
    specs: dict[str, StepSpec],
    empty: dict[str, frozenset[str]],
) -> StepSpec:
    kind = entry.kind
    if kind in ("evaluator", "workflow"):
        config, impl, addresses = _pooled(entry, pipeline)
        type_id = config.type
    else:
        config, impl, addresses = _inline(entry, pipeline)
        type_id = entry.target

    bindings, broadcast, keys = _bind_inputs(
        entry, kind, type_id, config, impl, addresses, workflow, types, later, empty
    )

    if broadcast and any(port.is_list for port in impl.output_ports()):
        raise GraphError(
            f"Step '{entry.name}' reads a list on a port that takes one item, so it runs once per element; but it "
            "outputs lists, and lists do not nest."
        )
    if broadcast and issubclass(impl, Transform) and not impl.broadcasts:
        address, value = _first_list(bindings, entry, workflow, types, later, empty)
        element = f"{address}[{value.keys[0] if value.keys else '<key>'}]"
        raise GraphError(
            f"Step '{entry.name}' reads `{address}`, a list, but transform '{type_id}' does not run once per element "
            f"of a list: name one element, such as `{element}`."
        )
    if issubclass(impl, InlineStep):
        _same_node(entry, impl, addresses, specs, types)
        bound = {
            str(address): _typed(address, entry, workflow, types, later, empty).classes
            for binding in bindings
            if binding.port.type is DataType.OUTPUT
            for address in binding.addresses
        }
        problem = impl.bound_problem(config, bound)
        if problem is not None:
            raise GraphError(f"Step '{entry.name}': {problem}")
    _check_extractor(entry, kind, type_id, config, pipeline)
    return StepSpec(
        name=entry.name,
        kind=kind,
        type=type_id,
        impl=impl,
        config=config,
        bindings=bindings,
        outputs=impl.output_ports(),
        extractor=entry.extractor,
        optional=entry.optional,
        broadcast=broadcast,
        keys=tuple(keys) if broadcast and keys is not None else None,
    )


def _inline(entry: StepEntry, pipeline: "PipelineConfig") -> tuple[Any, type[Step], dict[str, tuple[Address, ...]]]:
    """An inline step's resolved config, its implementation, and its input ports' addresses."""
    from dataeval_flow.steps._registry import inline_registry

    impl = inline_registry(entry.kind).get(entry.target)
    try:
        config = impl.resolved(entry.config, pipeline)
    except ValueError as error:
        raise GraphError(f"Step '{entry.name}': {error}") from error
    addresses = {port.name: port_addresses(config, port) for port in impl.input_ports()}
    return config, impl, addresses


def _pooled(entry: StepEntry, pipeline: "PipelineConfig") -> tuple[Any, type[Step], dict[str, tuple[Address, ...]]]:
    """An evaluator or workflow step's pool entry, its implementation, and its `input` addresses."""
    from dataeval_flow.evaluators._registry import get_evaluator
    from dataeval_flow.workflows._registry import get_workflow

    kind = entry.kind
    pool = pipeline.evaluators if kind == "evaluator" else pipeline.workflows
    config = next((item for item in pool or () if item.name == entry.target), None)
    if config is None:
        raise GraphError(f"Step '{entry.name}' names {kind} '{entry.target}', which `{kind}s:` does not define.")
    if isinstance(config, CustomWorkflowConfig):
        raise GraphError(
            f"Step '{entry.name}' names workflow '{entry.target}', a custom workflow: only a workflow type (`type:`) "
            "runs as a step."
        )
    if entry.input is None:
        raise GraphError(f"Step '{entry.name}' reads nothing: give it `input:`.")
    raw = [entry.input] if isinstance(entry.input, str) else list(entry.input)
    try:
        found = tuple(parse_address(text) for text in raw)
    except ValueError as error:
        raise GraphError(f"Step '{entry.name}': {error}") from error
    impl = (get_evaluator if kind == "evaluator" else get_workflow)(config.type)
    return config, impl, {"input": found}


def _bind_inputs(
    entry: StepEntry,
    kind: StepKind,
    type_id: str,
    config: Any,
    impl: type[Step],
    addresses: dict[str, tuple[Address, ...]],
    workflow: CustomWorkflowConfig,
    types: dict[str, ValueType],
    later: set[str],
    empty: dict[str, frozenset[str]],
) -> tuple[tuple[PortBinding, ...], bool, list[str] | None]:
    """Each input port's addresses, typed and checked against the port; and whether reading any of them broadcasts."""
    bindings: list[PortBinding] = []
    broadcast = False
    keys: list[str] | None = []
    for port in impl.input_ports():
        found = addresses.get(port.name, ())
        if port.count is not None and kind in ("evaluator", "workflow"):
            problem = _count_problem(port.count, found, config)
            if problem is not None:
                raise GraphError(f"Step '{entry.name}' runs {kind} '{entry.target}' ({type_id}), which {problem}")
        elif port.count is not None and not port.count.allows(len(found)):
            raise GraphError(
                f"Step '{entry.name}' runs {kind} '{type_id}', whose `{port.name}` takes {port.count.phrase}, but the "
                f"step names {len(found)}."
            )
        for address in found:
            value = _typed(address, entry, workflow, types, later, empty)
            _accepts(port, value, entry, address)
            if value.is_list and not port.is_list:
                broadcast = True
                keys = _broadcast_keys(keys, value)
        bindings.append(PortBinding(port, found))
    return tuple(bindings), broadcast, keys


def _first_list(
    bindings: tuple[PortBinding, ...],
    entry: StepEntry,
    workflow: CustomWorkflowConfig,
    types: dict[str, ValueType],
    later: set[str],
    empty: dict[str, frozenset[str]],
) -> tuple[Address, ValueType]:
    """The first address a step reads a list from on a port that takes one item, and what it holds."""
    return next(
        (address, value)
        for binding in bindings
        if not binding.port.is_list
        for address in binding.addresses
        if (value := _typed(address, entry, workflow, types, later, empty)).is_list
    )


def _count_problem(count: SourceCount, found: tuple[Address, ...], config: Any) -> str | None:
    """Why `found`'s length breaks `count`'s rule, or the pool entry's own rule; ``None`` if neither does."""
    if not count.allows(len(found)):
        return f"takes {count.phrase}, but the step names {len(found)}."
    return config.check_inputs(len(found))


def _broadcast_keys(keys: list[str] | None, value: ValueType) -> list[str] | None:
    """`keys` extended with any of `value`'s own keys not already in it, or ``None`` once any source's are unknown."""
    if keys is None or value.keys is None:
        return None
    return keys + [key for key in value.keys if key not in keys]


def _check_extractor(entry: StepEntry, kind: StepKind, type_id: str, config: Any, pipeline: "PipelineConfig") -> None:
    """Refuse an extractor on a step that embeds nothing, or a name `extractors:` does not define."""
    if entry.extractor is None:
        return
    if kind not in ("evaluator", "workflow"):
        raise GraphError(f"Step '{entry.name}' runs {kind} '{type_id}', which embeds nothing; remove `extractor:`.")
    if entry.extractor not in {extractor.name for extractor in pipeline.extractors or ()}:
        raise GraphError(
            f"Step '{entry.name}' names extractor '{entry.extractor}', which `extractors:` does not define."
        )
    if kind in ("evaluator", "workflow") and not config.inputs.accepts_extractor:
        raise GraphError(
            f"Step '{entry.name}' runs {kind} '{entry.target}' ({type_id}), which does not use an extractor; "
            "remove `extractor:`."
        )


def _typed(
    address: Address,
    entry: StepEntry,
    workflow: CustomWorkflowConfig,
    types: dict[str, ValueType],
    later: set[str],
    empty: dict[str, frozenset[str]],
) -> ValueType:
    """What `address` holds, or why it names nothing a step may read."""
    if address.name in later:
        raise GraphError(
            f"Step '{entry.name}' reads `{address}`, but step '{address.name}' runs later: a step reads only "
            "inputs and earlier steps."
        )
    value = _resolved_output(address, entry, workflow, types, empty)
    return _keyed(value, address, entry)


def _resolved_output(
    address: Address,
    entry: StepEntry,
    workflow: CustomWorkflowConfig,
    types: dict[str, ValueType],
    empty: dict[str, frozenset[str]],
) -> ValueType:
    """The value at `address`'s step and, when named, its output; or why neither names one."""
    whole = types.get(address.name)
    named = sorted(key.split(".", 1)[1] for key in types if key.startswith(f"{address.name}."))
    if address.output is None:
        if whole is not None:
            return whole
        if named:
            raise GraphError(
                f"Step '{entry.name}' reads `{address}`, but step '{address.name}' has outputs "
                f"{', '.join(named)}: name one, such as `{address.name}.{named[0]}`."
            )
        raise GraphError(
            f"Step '{entry.name}' reads `{address}`, which is no input or earlier step of workflow '{workflow.name}'."
        )
    if whole is not None:
        raise GraphError(
            f"Step '{entry.name}' reads `{address}`, but `{address.name}` has one output: name it `{address.name}`."
        )
    found = types.get(f"{address.name}.{address.output}")
    if found is None:
        where = f"has outputs {', '.join(named)}" if named else "is no input or earlier step"
        raise GraphError(f"Step '{entry.name}' reads `{address}`, but `{address.name}` {where}.")
    if address.output in empty.get(address.name, frozenset()):
        raise GraphError(
            f"Step '{entry.name}' reads `{address.base}`, which step '{address.name}' leaves empty with its settings."
        )
    return found


def _keyed(value: ValueType, address: Address, entry: StepEntry) -> ValueType:
    """`value`, narrowed to one element when `address` names a list key; `value` unchanged otherwise."""
    if address.key is None:
        return value
    if not value.is_list:
        raise GraphError(f"Step '{entry.name}' reads `{address}`, but `{address.base}` is not a list.")
    if value.keys is not None and address.key not in value.keys:
        raise GraphError(
            f"Step '{entry.name}' reads `{address}`, but `{address.base}` has elements {', '.join(value.keys)}, "
            f"not `{address.key}`."
        )
    return replace(value, is_list=False, keys=None)


def _accepts(port: Port, value: ValueType, entry: StepEntry, address: Address) -> None:
    """Refuse a value `port` does not take."""
    if value.type != port.type:
        raise GraphError(
            f"Step '{entry.name}' reads `{address}` on `{port.name}`, which takes {_ARTICLE[port.type]}, but "
            f"`{address}` is {_ARTICLE[value.type]}."
        )
    if value.classes and not any(port.accepts_class(cls) for cls in value.classes):
        wanted = ", ".join(cls.__name__ for cls in port.classes)
        given = ", ".join(cls.__name__ for cls in value.classes)
        raise GraphError(f"Step '{entry.name}' reads `{address}` on `{port.name}`, which takes {wanted}, not {given}.")
    if port.is_list and not value.is_list:
        raise GraphError(
            f"Step '{entry.name}' reads `{address}` on `{port.name}`, which takes a whole list, but `{address}` "
            "is one item."
        )


def _same_node(
    entry: StepEntry,
    impl: type[InlineStep],
    addresses: dict[str, tuple[Address, ...]],
    specs: dict[str, StepSpec],
    types: dict[str, ValueType],
) -> None:
    """Refuse an Output a transform applies to a Dataset other than exactly the one it was computed on."""
    on = tuple(str(address) for address in addresses.get("input", ()))
    for port in impl.same_node:
        for address in addresses.get(port, ()):
            computed = _computed_on(address, specs, types)
            # A ranking may read a reference set after the Dataset it indexes; then only its first input counts.
            if (computed[:1] if impl.same_node_first_input else computed) != on:
                where = ", ".join(f"`{item}`" for item in computed) or "no Dataset"
                target = ", ".join(f"`{item}`" for item in on)
                raise GraphError(
                    f"Step '{entry.name}' reads `{address}`, which was computed on {where}, not on {target}: it "
                    "applies only to the Dataset it was computed on."
                )


def _computed_on(address: Address, specs: dict[str, StepSpec], types: dict[str, ValueType]) -> tuple[str, ...]:
    """The addresses of the Datasets the Output at `address` was computed on: its producer's `input`.

    An element of a producer run once per element, such as `dupes[0]`, was computed on that element of each list the
    producer ran over, so each of those lists is named by the same key: `k.train[0]`.
    """
    producer = specs.get(address.name)
    if producer is None:
        return ()
    binding = next((binding for binding in producer.bindings if binding.port.name == "input"), None)
    if binding is None:
        return ()
    if address.key is None or not producer.broadcast or binding.port.is_list:
        return tuple(str(item) for item in binding.addresses)
    return tuple(
        str(replace(item, key=address.key)) if _is_list(item, types) else str(item) for item in binding.addresses
    )


def _is_list(address: Address, types: dict[str, ValueType]) -> bool:
    """Whether `address` names a whole list, rather than one item or one element of a list."""
    value = types.get(str(address))
    return value is not None and value.is_list
