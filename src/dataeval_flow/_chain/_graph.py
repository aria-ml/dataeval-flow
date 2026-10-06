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
    "verdict_steps",
]

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from dataeval_flow._input_spec import InputKind, SourceCount
from dataeval_flow.steps._address import Address, pair_keys, pair_members, parse_address
from dataeval_flow.steps._by import ByConfig
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import InlineStep, Step, StepKind, Transform, port_addresses
from dataeval_flow.steps._workflow import CustomWorkflowConfig, InputSlot, StepEntry

if TYPE_CHECKING:
    from dataeval_flow._chain._presets import Splice, Spliced
    from dataeval_flow.config._models import PipelineConfig
    from dataeval_flow.config._schemas import TaskConfig
    from dataeval_flow.evaluators._base import EvaluatorConfig
    from dataeval_flow.workflows._preset import Preset, PresetChain

_ARTICLE = {
    DataType.DATASET: "a Dataset",
    DataType.OUTPUT: "an Output",
    DataType.EXPORT: "an export record",
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
    by: ByConfig | None = None
    """For a `PerClassOutput`, an evaluate step's Output with `by:`, that step's `by:`: whether it keys by class or by
    group. ``None`` for anything else."""


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
    by: ByConfig | None = None
    pairs: bool = False

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
    aliases: Mapping[str, str] = field(default_factory=dict)
    """Each declared output of a preset step, such as `cleaning.clean`, to the output address in the spliced
    chain that holds it, `cleaning/clean` or `splits/split.train`."""
    splices: tuple["Splice", ...] = ()
    """Each preset step's splice, in run order (audit-as-a-step spec §4.1)."""

    def aliases_of(self, address: str) -> tuple[str, ...]:
        """The addresses that read what `address` holds: each declared output of a preset step made there."""
        return tuple(alias for alias, target in self.aliases.items() if target == address)


def build_graph(
    workflow: CustomWorkflowConfig,
    pipeline: "PipelineConfig",
    slot_keys: Mapping[str, tuple[str, ...]] | None = None,
    *,
    evaluators: Sequence["EvaluatorConfig[Any]"] = (),
    slot_types: Mapping[str, ValueType] | None = None,
) -> ChainGraph:
    """Resolve and type-check every step of `workflow` against `pipeline`.

    `slot_keys` holds a list slot's keys, the names of the sources a task binds to it. Without them, a key read from
    the slot, or from a list broadcast over it, is taken on trust. `evaluators` are entries a preset's steps name,
    found before the pipeline's own. `slot_types` types a slot as given: a preset's slot, spliced in as a step,
    holds whatever that step reads.

    Raises
    ------
    GraphError
        Naming the step and the address that does not connect.
    """
    verdicts = verdict_steps(workflow, pipeline)
    if len(verdicts) > 1:
        named = ", ".join(f"`{name}`" for name in verdicts[:-1]) + f" and `{verdicts[-1]}`"
        raise GraphError(
            f"Workflow '{workflow.name}' runs {_count_word(len(verdicts))} steps that give a verdict, {named}; a "
            "workflow gives one verdict."
        )
    keys = slot_keys or {}
    given = slot_types or {}
    types: dict[str, ValueType] = {
        slot.name: given.get(slot.name) or ValueType(DataType.DATASET, is_list=slot.is_list, keys=keys.get(slot.name))
        for slot in workflow.inputs
    }
    later = {entry.name for entry in workflow.steps}
    specs: dict[str, StepSpec] = {}
    empty: dict[str, frozenset[str]] = {}
    aliases: dict[str, str] = {}
    splices: list[Splice] = []
    for entry in workflow.steps:
        later.discard(entry.name)
        preset = _preset_step(entry, pipeline)
        if preset is not None:
            spliced = _splice(entry, *preset, workflow, pipeline, types, later, empty)
            specs.update((spec.name, spec) for spec in spliced.steps)
            aliases.update(spliced.aliases)
            types.update(spliced.types)
            empty[entry.name] = spliced.empty
            splices.append(spliced.splice)
            continue
        spec = _resolve(entry, workflow, pipeline, types, later, specs, empty, evaluators)
        specs[entry.name] = spec
        fixed: dict[str, tuple[str, ...]] = {}
        if issubclass(spec.impl, Transform):
            fixed = dict(spec.impl.output_keys(spec.config))
            empty[entry.name] = spec.impl.empty_outputs(spec.config)
        for port in spec.outputs:
            keys = fixed.get(port.name) if port.is_list else spec.keys
            types[spec.output_address(port)] = ValueType(
                port.type,
                port.classes,
                port.is_list or spec.broadcast,
                keys,
                step=spec.name,
                by=spec.by if spec.kind == "evaluator" else None,
            )
    return ChainGraph(
        workflow.name, tuple(workflow.inputs), tuple(specs.values()), aliases=aliases, splices=tuple(splices)
    )


def verdict_steps(workflow: CustomWorkflowConfig, pipeline: "PipelineConfig") -> list[str]:
    """The steps of `workflow` that run a preset whose chain gives a verdict, in order."""
    found: list[str] = []
    for entry in workflow.steps:
        preset = _preset_step(entry, pipeline)
        if preset is not None and preset[1].chain(preset[0]).blocking is not None:
            found.append(entry.name)
    return found


def _count_word(count: int) -> str:
    return {2: "two", 3: "three", 4: "four", 5: "five"}.get(count, str(count))


def one_step_graph(task: "TaskConfig", instance: "EvaluatorConfig[Any]", source_names: Sequence[str]) -> ChainGraph:
    """The graph an ``evaluator:`` task runs: its sources, read by one step named after it."""
    from dataeval_flow.evaluators._registry import get_evaluator

    impl = get_evaluator(instance.type)
    slots = tuple(InputSlot.model_construct(name=name, is_list=False) for name in source_names)
    (port,) = impl.input_ports()
    spec = StepSpec(
        name=task.name,
        kind="evaluator",
        type=instance.type,
        impl=impl,
        config=instance,
        bindings=(PortBinding(port, tuple(Address(name) for name in source_names)),),
        outputs=impl.output_ports(),
    )
    return ChainGraph(task.name, slots, (spec,), one_step=True)


def task_problems(pipeline: "PipelineConfig", graphs: Mapping[str, ChainGraph]) -> list[str]:
    """Why a task cannot run the graph it names: a list key its sources do not bind, a missing extractor, or two
    exports to one place. `graphs` holds the graph of each custom workflow and each preset entry, by name, as
    :func:`build_graph` built it."""
    workflows = {
        workflow.name: workflow for workflow in pipeline.workflows or () if isinstance(workflow, CustomWorkflowConfig)
    }
    extractors = {extractor.name: extractor for extractor in pipeline.extractors or ()}
    problems: list[str] = []
    owners: dict[str, str] = {export.name: f"export '{export.name}'" for export in pipeline.exports or ()}
    for task in pipeline.tasks or ():
        graph = graphs.get(task.workflow) if task.kind == "workflow" else None
        if graph is None:
            continue
        if task.matrix is not None:
            # Its runs are checked one by one; only the destinations its export steps claim are checked here, against
            # every other task's and export's (task-matrix spec §5.6).
            for spec in graph.steps:
                if issubclass(spec.impl, Transform):
                    problems.extend(_export_clashes(task, spec, spec.impl, owners))
            continue
        # A preset's chain reads no key of its list slot, so only a custom workflow's list slot has keys to bind.
        workflow = workflows.get(task.workflow)
        if workflow is not None:
            problems.extend(binding_problems(task, workflow, pipeline))
        problems.extend(_task_graph_problems(task, graph, owners, extractors))
    return problems


def binding_problems(
    task: "TaskConfig",
    workflow: CustomWorkflowConfig,
    pipeline: "PipelineConfig",
    evaluators: "Sequence[EvaluatorConfig[Any]]" = (),
) -> list[str]:
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
        build_graph(workflow, pipeline, slot_keys={slot.name: bound}, evaluators=evaluators)
    except GraphError as error:
        held = f"sources {', '.join(bound)}" if bound else "no source"
        return [f"Task '{task.name}' binds {held} to `{slot.name}`. {error}"]
    return []


def _task_graph_problems(
    task: "TaskConfig", graph: ChainGraph, owners: dict[str, str], extractors: Mapping[str, Any]
) -> list[str]:
    """A task's problems running one workflow graph: missing extractors (an optional step with none is skipped at
    run time instead, data-splitting spec §5.2), a model extractor on a step that reads one
    row per item or whose settings cannot read its rows, `by: predicted` without a model, and export destinations
    already claimed."""
    from dataeval_flow._predictions import runs_model

    problems: list[str] = []
    for spec in graph.steps:
        config: Any = spec.config
        needs_extractor = spec.kind == "evaluator" and config.requires_extractor()
        if needs_extractor and not (spec.extractor or task.extractor) and not spec.optional:
            kinds = ", ".join(sorted(str(kind) for kind in config.wanted_kinds() if kind.needs_extractor))
            problems.append(
                f"Task '{task.name}' runs workflow '{graph.name}', whose step '{spec.name}' needs an extractor to "
                f"produce {kinds}; name one with `extractor:` on the task or the step."
            )
        name = spec.extractor or task.extractor
        model = name if name is not None and runs_model(extractors.get(name)) else None
        if needs_extractor and model is not None and not config.inputs.detection_rows:
            problems.append(
                f"Task '{task.name}' runs workflow '{graph.name}', whose step '{spec.name}' embeds with `{model}`, "
                f"which runs a model whose rows may be detections; `{spec.type}` needs one row per item: only drift "
                "and OOD evaluators read it."
            )
        rows_problem = config.model_rows_problem() if needs_extractor and model is not None else None
        if rows_problem is not None and config.inputs.detection_rows:
            problems.append(
                f"Task '{task.name}' runs workflow '{graph.name}', whose step '{spec.name}' embeds with `{model}`, "
                f"and `{spec.type}` {rows_problem}"
            )
        if spec.kind == "evaluator" and spec.by is not None and spec.by.predicted is not None and model is None:
            problems.append(
                f"Task '{task.name}' runs workflow '{graph.name}', whose step '{spec.name}' has `by: predicted`, which "
                "needs a model's predictions: name an `uncertainty` extractor on the step, the task, or a preset's "
                "detector."
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
    evaluators: Sequence["EvaluatorConfig[Any]"],
) -> StepSpec:
    kind = entry.kind
    if kind in ("evaluator", "workflow"):
        config, impl, addresses = _pooled(entry, pipeline, evaluators)
        type_id = config.type
        read = sorted(config.wanted_kinds() - {InputKind.EMBEDDINGS, InputKind.LABELS}) if entry.by is not None else []
        if read:
            raise GraphError(
                f"Step '{entry.name}' has `by:`, which slices embeddings and labels, and `{type_id}` reads "
                f"{', '.join(read)}."
            )
    else:
        config, impl, addresses = _inline(entry, pipeline)
        type_id = entry.target

    bindings, broadcast, keys = _bind_inputs(
        entry, kind, type_id, config, impl, addresses, workflow, types, later, empty
    )

    if broadcast and any(port.is_list for port in impl.output_ports()):
        address, value = _first_list(bindings, entry, workflow, types, later, empty)
        element = f"{address}[{value.keys[0] if value.keys else '<key>'}]"
        raise GraphError(
            f"Step '{entry.name}' reads `{address}`, a list, on a port that takes one item, so it would run once per "
            f"element; but it outputs lists, and lists do not nest: name one element, such as `{element}`."
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
        _datasets_agree(entry, impl, addresses, specs, types)
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
    by = entry.by
    if kind == "check" and by is not None:
        # A check's bare `by:` keys by its input's keys, so it takes the `by:` of the step that made them: by
        # class, by group, or by predicted class.
        typed = (_typed(address, entry, workflow, types, later, empty) for b in bindings for address in b.addresses)
        by = next((value.by for value in typed if value.by is not None), by)
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
        by=by,
        pairs=entry.pairs,
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


def _pooled(
    entry: StepEntry, pipeline: "PipelineConfig", evaluators: Sequence["EvaluatorConfig[Any]"] = ()
) -> tuple[Any, type[Step], dict[str, tuple[Address, ...]]]:
    """An evaluator step's pool entry, its implementation, and its `input` addresses.

    A preset's own evaluator entries, `evaluators`, are found before the pipeline's. A `workflow:` step comes here only
    when its entry is missing or a custom workflow, and is refused: a workflow type's chain is spliced in instead.
    """
    from dataeval_flow.evaluators._registry import get_evaluator

    kind = entry.kind
    pool = [*evaluators, *(pipeline.evaluators or ())] if kind == "evaluator" else pipeline.workflows
    config = next((item for item in pool or () if item.name == entry.target), None)
    if config is None:
        raise GraphError(f"Step '{entry.name}' names {kind} '{entry.target}', which `{kind}s:` does not define.")
    if isinstance(config, CustomWorkflowConfig):
        raise GraphError(
            f"Step '{entry.name}' names workflow '{entry.target}', a custom workflow: only a workflow type (`type:`) "
            "runs as a step."
        )
    return config, get_evaluator(config.type), {"input": _input_addresses(entry)}


def _input_addresses(entry: StepEntry) -> tuple[Address, ...]:
    """The addresses an evaluator or workflow step's `input` names, refused when it names none or one is no address."""
    if entry.input is None:
        raise GraphError(f"Step '{entry.name}' reads nothing: give it `input:`.")
    raw = [entry.input] if isinstance(entry.input, str) else list(entry.input)
    try:
        return tuple(parse_address(text) for text in raw)
    except ValueError as error:
        raise GraphError(f"Step '{entry.name}': {error}") from error


def _preset_step(entry: StepEntry, pipeline: "PipelineConfig") -> "tuple[Any, type[Preset]] | None":
    """The pool entry and the preset a `workflow:` step runs; ``None`` where its entry is missing or a custom
    workflow."""
    from dataeval_flow.workflows._preset import preset_of

    if entry.kind != "workflow":
        return None
    config = next((item for item in pipeline.workflows or () if item.name == entry.target), None)
    preset = preset_of(config)
    return (config, preset) if preset is not None else None


def _splice(
    entry: StepEntry,
    config: Any,
    preset: "type[Preset]",
    workflow: CustomWorkflowConfig,
    pipeline: "PipelineConfig",
    types: dict[str, ValueType],
    later: set[str],
    empty: dict[str, frozenset[str]],
) -> "Spliced":
    """A preset step's chain, spliced in, refused unless the step reads one Dataset per slot."""
    from dataeval_flow._chain._presets import splice_preset

    if entry.pairs:
        raise GraphError(
            f"Step '{entry.name}' runs workflow '{entry.target}' ({config.type}), a preset, which runs no pairs: "
            "remove `pairs:`."
        )

    names = preset.slot_names()
    found = _input_addresses(entry)
    if len(found) != len(names):
        slots = ", ".join(
            f"`{name}`" + (" (a list)" if not isinstance(slot, str) and slot.is_list else "")
            for slot, name in zip(preset.slots, names, strict=True)
        )
        raise GraphError(
            f"Step '{entry.name}' runs workflow '{entry.target}' ({config.type}), whose inputs are {slots}, but the "
            f"step names {len(found)}."
        )
    bound: list[tuple[Address, ValueType]] = []
    for slot, name, address in zip(preset.slots, names, found, strict=True):
        value = _typed(address, entry, workflow, types, later, empty)
        if value.type is not DataType.DATASET:
            raise GraphError(
                f"Step '{entry.name}' reads `{address}`, which is {_ARTICLE[value.type]}, but workflow "
                f"'{entry.target}' ({config.type}) reads Datasets."
            )
        if not isinstance(slot, str) and slot.is_list and not value.is_list:
            raise GraphError(
                f"Step '{entry.name}' binds `{address}` to `{name}`, which takes a list of Datasets, but `{address}` "
                "is one Dataset."
            )
        bound.append((address, value))
    by_slot = dict(zip(names, bound, strict=True))
    chain = preset.chain(config)
    _check_splice(entry, config, preset, chain, by_slot)
    _check_reference(entry, config, preset, chain, pipeline, by_slot)
    _check_extractor(entry, "workflow", config.type, config, pipeline)
    return splice_preset(entry, config, preset, pipeline, by_slot)


def _check_splice(
    entry: StepEntry,
    config: Any,
    preset: "type[Preset]",
    chain: "PresetChain",
    bound: Mapping[str, tuple[Address, ValueType]],
) -> None:
    """Refuse, on a preset step, `optional:` where its chain gives a verdict (audit-as-a-step spec D9); an `accepted`
    key naming an element no list bound to a slot has, where its keys are known (§4.1); and, where its chain declares a
    reference, a record or a verdict, an element of such a list keyed like one of its single slots, which the record's
    columns and the preflight's names would confuse."""
    if entry.optional and chain.blocking is not None:
        raise GraphError(
            f"Step '{entry.name}' runs workflow '{entry.target}' ({config.type}), which gives a verdict, so it may not "
            "be optional: a step of it that failed would read as not assessed, and the verdict would pass."
        )
    lists = [(address, value) for address, value in bound.values() if value.is_list and value.keys]
    keys = [key for _, value in lists for key in value.keys or ()]
    for accepted in chain.accepted if lists else ():
        if "[" in accepted and (element := accepted[accepted.index("[") + 1 : -1]) not in keys:
            raise GraphError(
                f"Step '{entry.name}' runs workflow '{entry.target}' ({config.type}), whose `accepted` names "
                f"`{accepted}`, but `{lists[0][0]}` has no element `{element}`. Its elements: {', '.join(keys)}."
            )
    if chain.reference is None and chain.record is None and chain.blocking is None:
        return
    singles = {
        name for slot, name in zip(preset.slots, bound, strict=True) if isinstance(slot, str) or not slot.is_list
    }
    for name, (address, value) in bound.items():
        clash = next((key for key in value.keys or () if key in singles), None) if name not in singles else None
        if clash is not None:
            raise GraphError(
                f"Step '{entry.name}' binds `{address}`, whose element `{clash}` has the name of its slot `{clash}`: "
                "name the element otherwise with `keys:`."
            )


def _check_reference(
    entry: StepEntry,
    config: Any,
    preset: "type[Preset]",
    chain: "PresetChain",
    pipeline: "PipelineConfig",
    bound: Mapping[str, tuple[Address, ValueType]],
) -> None:
    """Refuse, on a preset step whose chain declares a reference or a verdict, a list bound to a slot taking one Dataset
    (audit-as-a-step spec D3), and, under a reference, a metadata policy naming `reference_split` (D7)."""
    if chain.reference is not None or chain.blocking is not None:
        for slot, (name, (address, value)) in zip(preset.slots, bound.items(), strict=True):
            single = isinstance(slot, str) or not slot.is_list
            if single and value.is_list:
                element = f"{address}[{value.keys[0] if value.keys else '<key>'}]"
                raise GraphError(
                    f"Step '{entry.name}' binds `{address}`, a list, to `{name}` of workflow '{entry.target}' "
                    f"({config.type}), which takes one Dataset there: name one element, such as `{element}`."
                )
    if chain.reference is not None and (named := getattr(config, "metadata", None)) is not None:
        policy = next((item for item in pipeline.metadata or () if item.name == named), None)
        if policy is not None and policy.reference_split is not None:
            raise GraphError(
                f"Step '{entry.name}' runs workflow '{entry.target}' ({config.type}), whose metadata policy "
                f"'{named}' names reference_split={policy.reference_split!r}: run as a step, a preset encodes like "
                f"the Dataset bound to `{chain.reference}`, and a reference_split, which names a task source, cannot "
                "single out a split part. Remove reference_split from the policy, or run the preset as a task."
            )


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
    """Each input port's addresses, typed and checked against the port; whether reading any of them broadcasts; and
    the keys it runs over. With `pairs:`, the one list a port reads alone is read two elements a run (audit spec
    §9.2)."""
    bindings: list[PortBinding] = []
    broadcast = False
    keys: list[str] | None = []
    paired = lists = 0
    for port in impl.input_ports():
        found = addresses.get(port.name, ())
        values = [_typed(address, entry, workflow, types, later, empty) for address in found]
        pair = entry.pairs and len(values) == 1 and values[0].is_list and not port.is_list
        named, said = (2, "`pairs:` hands it two") if pair else (len(found), f"the step names {len(found)}")
        if pair and port.count is None:
            raise GraphError(
                f"Step '{entry.name}' has `pairs: true`, but `{port.name}` of {kind} '{type_id}' reads one item, "
                "not two."
            )
        if port.count is not None and kind == "evaluator":
            problem = _count_problem(port.count, named, said, config)
            if problem is not None:
                raise GraphError(f"Step '{entry.name}' runs {kind} '{entry.target}' ({type_id}), which {problem}")
        elif port.count is not None and not port.count.allows(named):
            raise GraphError(
                f"Step '{entry.name}' runs {kind} '{type_id}', whose `{port.name}` takes {port.count.phrase}, but "
                f"{said}."
            )
        for address, value in zip(found, values, strict=True):
            _accepts(port, value, entry, address)
            if value.is_list and not port.is_list:
                broadcast, lists = True, lists + 1
                keys = _pair_keys(value.keys, entry, address) if pair else _broadcast_keys(keys, value)
        paired += pair
        bindings.append(PortBinding(port, found))
    if entry.pairs and (paired != 1 or lists != 1):
        raise GraphError(
            f"Step '{entry.name}' has `pairs: true`, so it reads one list, alone on one input, and runs once per pair "
            "of its elements."
        )
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


def _count_problem(count: SourceCount, named: int, said: str, config: Any) -> str | None:
    """Why `named` addresses break `count`'s rule, or the pool entry's own rule; ``None`` if neither does."""
    if not count.allows(named):
        return f"takes {count.phrase}, but {said}."
    return config.check_inputs(named)


def _pair_keys(keys: tuple[str, ...] | None, entry: StepEntry, address: Address) -> list[str] | None:
    """The keys of a pairwise run over a list with `keys`: each unordered pair, in list order; ``None`` if unknown."""
    if keys is None:
        return None
    try:
        return pair_keys(keys)
    except ValueError as error:
        raise GraphError(
            f"Step '{entry.name}' pairs `{address}`, whose {error}: rename a source so no two pairs share a key."
        ) from None


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
            f"Step '{entry.name}' reads `{address}`, but `{address.base}` has elements "
            f"{', '.join(value.keys) or 'none'}, not `{address.key}`."
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
    if value.by is not None and (entry.kind != "check" or entry.by is None):
        raise GraphError(
            f"Step '{entry.name}' reads `{address}`, which holds Outputs per key (`by:`): only a check with "
            "`by:` reads them."
        )
    if entry.kind == "check" and entry.by is not None and value.by is None:
        raise GraphError(f"Step '{entry.name}' has `by:`, but `{address}` holds one Output, not one per key.")
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


def _datasets_agree(
    entry: StepEntry,
    impl: type[InlineStep],
    addresses: dict[str, tuple[Address, ...]],
    specs: dict[str, StepSpec],
    types: dict[str, ValueType],
) -> None:
    """Refuse Outputs a step requires to share their Datasets, or to have been computed on its own Dataset ports."""
    for port in impl.shared_datasets:
        found = {str(address): _computed_on(address, specs, types) for address in addresses.get(port, ())}
        if len(set(found.values())) > 1:
            listed = "; ".join(f"`{address}` on {_where(on)}" for address, on in found.items())
            raise GraphError(
                f"Step '{entry.name}' reads Outputs on `{port}` computed on different Datasets ({listed}): it reads "
                "the Outputs of one comparison."
            )
    for port, dataset_ports in impl.computed_on.items():
        wanted = tuple(str(address) for name in dataset_ports for address in addresses.get(name, ()))
        for address in addresses.get(port, ()):
            computed = _computed_on(address, specs, types)
            if computed != wanted:
                raise GraphError(
                    f"Step '{entry.name}' reads `{address}`, which was computed on {_where(computed)}, not on "
                    f"{_where(wanted)}: it applies only to the Datasets it was computed on."
                )


def _where(addresses: tuple[str, ...]) -> str:
    return ", ".join(f"`{item}`" for item in addresses) or "no Dataset"


def _computed_on(address: Address, specs: dict[str, StepSpec], types: dict[str, ValueType]) -> tuple[str, ...]:
    """The addresses of the Datasets the Output at `address` was computed on: its producer's `input`.

    An element of a producer run once per element, such as `dupes[0]`, was computed on that element of each list the
    producer ran over, so each of those lists is named by the same key: `k.train[0]`. An Output whose producer read
    Outputs alone, as `ood-union` does, was computed on what those were, where they agree; ``()`` where they do not.
    """
    producer = specs.get(address.name)
    if producer is None:
        return ()
    binding = next((binding for binding in producer.bindings if binding.port.name == "input"), None)
    if binding is None:
        return ()
    if address.key is None or not producer.broadcast or binding.port.is_list:
        read = binding.addresses
    else:
        parts = pair_members(address.key) if producer.pairs else (address.key,)
        read = tuple(
            narrowed
            for item in binding.addresses
            for narrowed in ([replace(item, key=part) for part in parts] if _is_list(item, types) else [item])
        )
    if read and all(_holds_output(item, types) for item in read):
        found = {_computed_on(item, specs, types) for item in read}
        return found.pop() if len(found) == 1 else ()
    return tuple(str(item) for item in read)


def _holds_output(address: Address, types: dict[str, ValueType]) -> bool:
    """Whether `address`, or the list it names an element of, holds Outputs."""
    value = types.get(str(address.base))
    return value is not None and value.type is DataType.OUTPUT


def _is_list(address: Address, types: dict[str, ValueType]) -> bool:
    """Whether `address` names a whole list, rather than one item or one element of a list."""
    value = types.get(str(address))
    return value is not None and value.is_list
