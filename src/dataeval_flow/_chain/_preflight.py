"""Preflight: what each step needs resolved before any step runs, and the Dataset kinds reaching each (spec §5.4)."""

__all__ = ["check_kinds", "detect_kind", "step_contexts"]

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dataeval_flow._chain._graph import ChainGraph, GraphError, StepSpec
from dataeval_flow._chain._nodes import Node, NodeList
from dataeval_flow._chain._run import StepContext
from dataeval_flow.steps._step import Transform

if TYPE_CHECKING:
    from dataeval_flow.config._models import PipelineConfig
    from dataeval_flow.workflows._context import DatasetContext


def step_contexts(
    graph: ChainGraph,
    pipeline: "PipelineConfig",
    data_dir: Path | None,
    slot_contexts: Mapping[str, Sequence["DatasetContext"]],
) -> dict[str, StepContext]:
    """Each step's metadata policy, stats policy and ontology, resolved as ``_run_single_task`` resolves a task's.

    `slot_contexts` holds each input slot's source contexts; a list slot's holds one per bound source. A step's
    value range and band groups come from the slots it descends from.
    """
    from dataeval_flow._orchestrator import (
        _apply_dataset_value_range,
        _resolve_metadata_policy,
        _resolve_ontology,
        _resolve_stats_policy,
    )

    reach = _slots_reached(graph)
    resolved: dict[str, StepContext] = {}
    for spec in graph.steps:
        contexts = [context for slot in reach[spec.name] for context in slot_contexts.get(slot, ())]
        instance: Any = spec.config
        target = getattr(instance, "name", spec.name)
        policy = _resolve_metadata_policy(instance, pipeline, data_dir)
        policy = _apply_dataset_value_range(policy, [context.value_range for context in contexts], target, spec.kind)
        stats = _resolve_stats_policy(instance, pipeline, {context.name: context for context in contexts})
        if policy is not None and stats is not None:
            policy = replace(policy, stats=stats)
        ontology = _resolve_ontology(instance, pipeline, data_dir)
        resolved[spec.name] = StepContext(metadata_policy=policy, stats_policy=stats, ontology=ontology)
    return resolved


def _slots_reached(graph: ChainGraph) -> dict[str, list[str]]:
    """For each step, the input slots its Datasets descend from, in slot order."""
    slots = [slot.name for slot in graph.slots]
    origin: dict[str, set[str]] = {name: {name} for name in slots}
    reached: dict[str, list[str]] = {}
    for spec in graph.steps:
        found: set[str] = set()
        for binding in spec.bindings:
            for address in binding.addresses:
                found |= origin.get(address.name, set()) | origin.get(str(address.base), set())
        reached[spec.name] = [slot for slot in slots if slot in found]
        origin[spec.name] = found
        for port in spec.outputs:
            origin[spec.output_address(port)] = found
    return reached


def detect_kind(dataset: Any) -> str | None:
    """The Dataset kind DataEval detects from `dataset`'s first datum; ``None`` for an empty one."""
    from dataeval.exceptions import MaiteShapeError
    from dataeval.utils.data import validate_dataset

    if len(dataset) == 0:
        return None
    try:
        return str(validate_dataset(dataset, expected="any_target", caller="dataeval-flow"))
    except MaiteShapeError:
        return "image_only"


def check_kinds(graph: ChainGraph, inputs: Mapping[str, Node | NodeList]) -> dict[str, str | None]:
    """The Dataset kind at each Dataset address, refusing any port that does not take the kind reaching it."""
    kinds: dict[str, str | None] = {}
    for slot in graph.slots:
        value = inputs[slot.name]
        nodes = list(value.present.values()) if isinstance(value, NodeList) else [value]
        found = {detect_kind(node.value) for node in nodes}
        for node in nodes:
            node.kind = detect_kind(node.value) if len(found) > 1 else next(iter(found), None)
        kinds[slot.name] = next(iter(found)) if len(found) == 1 else None
    for spec in graph.steps:
        input_kinds = _check_step(spec, kinds)
        if issubclass(spec.impl, Transform):
            try:
                made = spec.impl().output_kinds(spec.config, input_kinds)  # type: ignore[call-arg,arg-type]
            except ValueError as error:
                raise GraphError(f"Step '{spec.name}': {error}") from error
            for port in spec.outputs:
                if port.type == "dataset":
                    kinds[spec.output_address(port)] = made.get(port.name)
    return kinds


def _check_step(spec: StepSpec, kinds: Mapping[str, str | None]) -> dict[str, str | None]:
    input_kinds: dict[str, str | None] = {}
    for binding in spec.bindings:
        known = {str(address): kinds.get(str(address.base)) for address in binding.addresses}
        distinct = {kind for kind in known.values() if kind is not None}
        if len(distinct) > 1:
            listed = ", ".join(f"`{address}` is {kind}" for address, kind in known.items() if kind is not None)
            raise GraphError(
                f"Step '{spec.name}' reads Datasets of different kinds on `{binding.port.name}`: {listed}."
            )
        for address in binding.addresses:
            kind = kinds.get(str(address.base))
            input_kinds.setdefault(binding.port.name, kind)
            allowed = binding.port.kinds
            if allowed is not None and kind is not None and kind not in allowed:
                raise GraphError(
                    f"Step '{spec.name}' reads `{address}`, a {kind} Dataset, but `{binding.port.name}` takes "
                    f"{', '.join(sorted(allowed))}."
                )
    return input_kinds
