"""Preflight: what each step needs resolved before any step runs, and the Dataset kinds reaching each (spec §5.4)."""

__all__ = ["check_kinds", "detect_kind", "step_contexts"]

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dataeval_flow._chain._graph import ChainGraph, GraphError, StepSpec
from dataeval_flow._chain._nodes import Node, NodeList
from dataeval_flow._chain._run import StepContext
from dataeval_flow.steps._address import Address
from dataeval_flow.steps._step import Transform

if TYPE_CHECKING:
    from dataeval_flow._stats import ResolvedStatsPolicy
    from dataeval_flow.config._models import PipelineConfig
    from dataeval_flow.workflows._context import DatasetContext

_Read = tuple[str, str | None]
"""A Dataset a step reads: the address of the node or list holding it, past any preset alias, and its element key."""


def step_contexts(
    graph: ChainGraph,
    pipeline: "PipelineConfig",
    data_dir: Path | None,
    slot_contexts: Mapping[str, Sequence["DatasetContext"]],
) -> dict[str, StepContext]:
    """Each step's metadata policy, stats policy and ontology, resolved as ``_run_single_task`` resolves a task's, and
    its stats unions.

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
    return _with_stats_unions(graph, resolved)


def _with_stats_unions(graph: ChainGraph, contexts: Mapping[str, StepContext]) -> dict[str, StepContext]:
    """`contexts`, with each evaluator step that reads statistics given the union to ask for at each Dataset it reads.

    Each such step asks the cache for its own families, and the cache computes only what it lacks, so no statistic is
    computed twice; but each step's request reads the Dataset again. The union of every request made of a Dataset,
    asked for first, reads it once. Requests whose scope fragments differ cannot share a cache entry, so a step's
    union takes in only those whose fragment is its own. A step keeps only the unions wider than its own request.
    """
    requested = {
        spec.name: (policy, _reads(graph, spec))
        for spec in graph.steps
        if (policy := _stats_request(spec, contexts[spec.name])) is not None
    }
    readers: dict[_Read, list[ResolvedStatsPolicy]] = {}
    for policy, reads in requested.values():
        for read in reads:
            readers.setdefault(read, []).append(policy)
    planned = dict(contexts)
    for name, (policy, reads) in requested.items():
        unions: dict[str, ResolvedStatsPolicy] = {}
        for read in reads:
            for (base, key), requests in _requests_of(read, readers).items():
                union = _union(policy, requests)
                if union.request != policy.request:
                    unions[str(Address(base, key=key))] = union
        if unions:
            planned[name] = replace(contexts[name], stats_unions=unions)
    return planned


def _stats_request(spec: StepSpec, context: StepContext) -> "ResolvedStatsPolicy | None":
    """The stats policy evaluator step `spec` asks the cache for, as ``produce_stats`` computes it.

    ``None`` for a step that reads no statistics, and for one whose request raises: that step fails when it runs, as
    it would have without a plan.
    """
    from dataeval_flow._input_spec import InputKind
    from dataeval_flow._stats import stats_policy_for

    config: Any = spec.config
    if spec.kind != "evaluator" or InputKind.STATS not in config.wanted_kinds():
        return None
    try:
        return stats_policy_for(context, **config.stats_request())
    except Exception:  # noqa: BLE001 - its run raises it again, failing only the step and what reads it
        return None


def _reads(graph: ChainGraph, spec: StepSpec) -> list[_Read]:
    """The Datasets evaluator step `spec` reads on its `input`, each named past any preset alias."""
    return [
        (graph.aliases.get(str(address.base), str(address.base)), address.key) for address in spec.addresses("input")
    ]


def _requests_of(
    read: _Read, readers: Mapping[_Read, list["ResolvedStatsPolicy"]]
) -> dict[_Read, list["ResolvedStatsPolicy"]]:
    """The requests made of each Dataset `read` reaches, keyed by the read that names it.

    An element named alone is also read by each step running over its whole list. A whole list is read element by
    element, so each element another step names alone is read by that step as well.
    """
    base, key = read
    whole = readers.get((base, None), [])
    if key is not None:
        return {read: [*readers[read], *whole]}
    named = {
        (base, element): [*whole, *requests]
        for (other, element), requests in readers.items()
        if other == base and element is not None
    }
    return {read: whole, **named}


def _union(policy: "ResolvedStatsPolicy", requests: Sequence["ResolvedStatsPolicy"]) -> "ResolvedStatsPolicy":
    """`policy`, widened to measure, view by view, every family any of `requests` sharing its cache entry measures."""
    fragment = policy.scope_fragment()
    measure = dict(policy.measure)
    for request in requests:
        if request.scope_fragment() != fragment:
            continue
        for view, flags in request.measure:
            measure[view] = measure[view] | flags if view in measure else flags
    return replace(policy, measure=tuple(measure.items()))


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
            for alias in graph.aliases_of(spec.output_address(port)):
                origin[alias] = found
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
    """The Dataset kind at each Dataset address, refusing any port that does not take a kind reaching it.

    A list address may carry several distinct kinds at once (it broadcasts per element at run time), so every
    kind it carries is checked against the port, even though the address's own entry in the returned mapping
    collapses to ``None`` once its elements disagree.
    """
    kinds: dict[str, str | None] = {}
    reaching: dict[str, set[str]] = {}
    for slot in graph.slots:
        value = inputs[slot.name]
        nodes = list(value.present.values()) if isinstance(value, NodeList) else [value]
        detected = [(node, detect_kind(node.value)) for node in nodes]
        found = {kind for _, kind in detected}
        for node, kind in detected:
            node.kind = kind if len(found) > 1 else next(iter(found), None)
        reaching[slot.name] = {kind for kind in found if kind is not None}
        kinds[slot.name] = next(iter(found)) if len(found) == 1 else None
    for spec in graph.steps:
        input_kinds = _check_step(spec, kinds, reaching)
        if issubclass(spec.impl, Transform):
            try:
                made = spec.impl().output_kinds(spec.config, input_kinds)  # type: ignore[call-arg,arg-type]
            except ValueError as error:
                raise GraphError(f"Step '{spec.name}': {error}") from error
            for port in spec.outputs:
                if port.type == "dataset":
                    kind = made.get(port.name)
                    kinds[spec.output_address(port)] = kind
                    reaching[spec.output_address(port)] = {kind} if kind is not None else set()
                    for alias in graph.aliases_of(spec.output_address(port)):
                        kinds[alias] = kinds[spec.output_address(port)]
                        reaching[alias] = reaching[spec.output_address(port)]
    return kinds


def _check_step(
    spec: StepSpec, kinds: Mapping[str, str | None], reaching: Mapping[str, set[str]]
) -> dict[str, str | None]:
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
            if allowed is None:
                continue
            for candidate in sorted(reaching.get(str(address.base), set())):
                if candidate not in allowed:
                    raise GraphError(
                        f"Step '{spec.name}' reads `{address}`, a {candidate} Dataset, but `{binding.port.name}` "
                        f"takes {', '.join(sorted(allowed))}."
                    )
    return input_kinds
