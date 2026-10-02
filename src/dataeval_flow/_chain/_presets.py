"""A preset run as a step of a custom workflow: its chain spliced into the graph, each step as `<step>/<inner>`."""

__all__ = ["Spliced", "splice_preset"]

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from dataeval_flow._chain._graph import GraphError, PortBinding, StepSpec, ValueType, build_graph
from dataeval_flow.steps._address import Address, parse_address
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import Transform

if TYPE_CHECKING:
    from dataeval_flow.config._models import PipelineConfig
    from dataeval_flow.steps._workflow import StepEntry
    from dataeval_flow.workflows._preset import Preset


@dataclass(frozen=True)
class Spliced:
    """A preset step's chain, as steps of the custom workflow that runs it."""

    steps: tuple[StepSpec, ...]
    aliases: dict[str, str]
    """Each declared output's address, such as `cleaning.clean`, to the output address of the spliced step that makes
    it."""
    types: dict[str, ValueType]
    """What each declared output's address holds."""
    empty: frozenset[str] = frozenset()
    """The declared outputs these settings leave empty, which no step outside may read."""


def splice_preset(
    entry: "StepEntry",
    config: Any,
    preset: "type[Preset]",
    pipeline: "PipelineConfig",
    bound: Mapping[str, tuple[Address, ValueType]],
) -> Spliced:
    """The steps preset entry `config` expands to, as steps of the custom workflow that runs it as step `entry`.

    `bound` holds, by slot, the address `entry` reads for it and what that holds. Each step of the chain is named
    `<entry>/<step>` and reads what its slot is bound to, so over a list it runs once per element. `entry`'s
    `optional` holds for every step of the chain, and its `extractor` for every one that embeds and names none.
    """
    from dataeval_flow.workflows._preset import expand_preset

    chain, evaluators = expand_preset(config, preset)
    try:
        graph = build_graph(
            chain, pipeline, evaluators=evaluators, slot_types={slot: value for slot, (_, value) in bound.items()}
        )
    except GraphError as error:
        raise GraphError(f"Step '{entry.name}' runs workflow '{entry.target}' ({config.type}): {error}") from error
    renamed = {spec.name: f"{entry.name}/{spec.name}" for spec in graph.steps}
    steps = tuple(
        replace(
            spec,
            name=renamed[spec.name],
            bindings=tuple(
                PortBinding(binding.port, tuple(_moved(address, bound, renamed) for address in binding.addresses))
                for binding in spec.bindings
            ),
            optional=spec.optional or entry.optional,
            extractor=spec.extractor or (entry.extractor if _embeds(spec) else None),
        )
        for spec in graph.steps
    )
    mapped = preset.chain(config).outputs
    aliases: dict[str, str] = {}
    types: dict[str, ValueType] = {}
    empty: set[str] = set()
    made = {spec.name: spec for spec in graph.steps}
    for port in preset.outputs:
        target = mapped.get(port.name, port.name)
        found = _output(made, target)
        if found is None:
            where = (
                f"at `{target}`, but `{target}` is no Dataset its chain makes"
                if port.name in mapped
                else f"but no step of its chain named `{port.name}` makes one Dataset"
            )
            raise GraphError(
                f"Step '{entry.name}' runs workflow '{entry.target}' ({config.type}), which declares output "
                f"`{port.name}`{' ' if port.name in mapped else ', '}{where}."
            )
        spec, output = found
        inner = renamed[spec.name] if len(spec.outputs) == 1 else f"{renamed[spec.name]}.{output.name}"
        transform = spec.impl if issubclass(spec.impl, Transform) else None
        keys = dict(transform.output_keys(spec.config)).get(output.name) if output.is_list and transform else spec.keys
        address = f"{entry.name}.{port.name}"
        aliases[address] = inner
        types[address] = ValueType(
            DataType.DATASET,
            output.classes,
            is_list=output.is_list or spec.broadcast,
            keys=keys,
            step=renamed[spec.name],
        )
        if transform is not None and output.name in transform.empty_outputs(spec.config):
            empty.add(port.name)
    return Spliced(steps, aliases, types, frozenset(empty))


def _output(made: Mapping[str, StepSpec], target: str) -> "tuple[StepSpec, Port] | None":
    """The step and Dataset output `target` names in a preset's chain: `step.output`, or a step with one output;
    ``None`` where it names no Dataset."""
    address = parse_address(target)
    spec = made.get(address.name)
    if spec is None or address.key is not None:
        return None
    if address.output is None:
        port = spec.outputs[0] if len(spec.outputs) == 1 else None
    else:
        port = next((port for port in spec.outputs if port.name == address.output), None)
    return (spec, port) if port is not None and port.type is DataType.DATASET else None


def _moved(address: Address, bound: Mapping[str, tuple[Address, ValueType]], renamed: Mapping[str, str]) -> Address:
    """`address`, read inside the chain, as the custom workflow names it: a slot as what it is bound to, a step by
    its spliced name."""
    if address.name in bound:
        outer, _ = bound[address.name]
        return outer if address.key is None else replace(outer, key=address.key)
    return replace(address, name=renamed[address.name])


def _embeds(spec: StepSpec) -> bool:
    """Whether `spec` runs an evaluator or workflow that takes an extractor."""
    config: Any = spec.config
    return spec.kind in ("evaluator", "workflow") and config.inputs.accepts_extractor
