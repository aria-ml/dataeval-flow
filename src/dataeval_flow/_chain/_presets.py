"""A preset run as a step of a custom workflow: its chain spliced into the graph, each step as `<step>/<inner>`."""

__all__ = ["Spliced", "splice_preset"]

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from dataeval_flow._chain._graph import GraphError, PortBinding, StepSpec, ValueType, build_graph
from dataeval_flow.steps._address import Address
from dataeval_flow.steps._port import DataType

if TYPE_CHECKING:
    from dataeval_flow.config._models import PipelineConfig
    from dataeval_flow.steps._workflow import StepEntry
    from dataeval_flow.workflows._preset import Preset


@dataclass(frozen=True)
class Spliced:
    """A preset step's chain, as steps of the custom workflow that runs it."""

    steps: tuple[StepSpec, ...]
    aliases: dict[str, str]
    """Each declared output's address, such as `cleaning.clean`, to the address of the spliced step that makes it."""
    types: dict[str, ValueType]
    """What each declared output's address holds."""


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
    aliases: dict[str, str] = {}
    types: dict[str, ValueType] = {}
    made = {spec.name: spec for spec in graph.steps}
    for port in preset.outputs:
        spec = made.get(port.name)
        (output, *more) = spec.outputs if spec is not None else (None,)
        if spec is None or more or output is None or output.type is not DataType.DATASET or output.is_list:
            raise GraphError(
                f"Step '{entry.name}' runs workflow '{entry.target}' ({config.type}), which declares output "
                f"`{port.name}`, but no step of its chain named `{port.name}` makes one Dataset."
            )
        address = f"{entry.name}.{port.name}"
        aliases[address] = renamed[spec.name]
        types[address] = ValueType(
            DataType.DATASET, output.classes, is_list=spec.broadcast, keys=spec.keys, step=renamed[spec.name]
        )
    return Spliced(steps, aliases, types)


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
