"""The executor: run a graph's steps in config order over its bound inputs (spec §4.2, §5.5, §5.6)."""

__all__ = ["ChainRun", "ExtractorSetup", "RunSettings", "StepContext", "bind_inputs", "input_node", "run_chain"]

import contextlib
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dataeval_flow._chain._graph import ChainGraph, StepSpec
from dataeval_flow._chain._identity import element_key, output_key, settings_of, short_digest, step_key
from dataeval_flow._chain._nodes import Missing, Node, NodeList, Root
from dataeval_flow._chain._reads import MetadataRead, ReadingContext, note_read, noting_reads
from dataeval_flow._result import LabelSpaceRecord, LineageRecord, failure_message
from dataeval_flow.steps._address import Address
from dataeval_flow.steps._by import roll_up
from dataeval_flow.steps._check import Check, CheckContext
from dataeval_flow.steps._combine import Combine, CombineContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._result import StepResult, StepStatus
from dataeval_flow.steps._step import StepSkipped, Transform, TransformContext
from dataeval_flow.workflows._base import Finding

if TYPE_CHECKING:
    from dataeval_flow._policy import ResolvedPolicy
    from dataeval_flow._sources import ResolvedSource
    from dataeval_flow._stats import ResolvedStatsPolicy
    from dataeval_flow.config._models import PipelineConfig
    from dataeval_flow.config.extractors._base import ExtractorConfig
    from dataeval_flow.evaluators._evaluator import Evaluator
    from dataeval_flow.workflows._base import Workflow
    from dataeval_flow.workflows._context import DatasetContext, ResolvedOntology

_logger = logging.getLogger(__name__)

_Value = Node | NodeList | Missing
"""What an address holds at run time."""


@dataclass(frozen=True)
class ExtractorSetup:
    """An extractor as a task or step uses it: its config, the preprocessing it applies, and its batch size."""

    config: "ExtractorConfig"
    transforms: Callable[..., Any] | None = None
    batch_size: int | None = None


@dataclass(frozen=True)
class StepContext:
    """What preflight resolved for one step: its metadata and stats policies, its ontology, and its stats unions."""

    metadata_policy: "ResolvedPolicy | None" = None
    stats_policy: "ResolvedStatsPolicy | None" = None
    ontology: "ResolvedOntology | None" = None
    stats_unions: "Mapping[str, ResolvedStatsPolicy]" = field(default_factory=dict)
    """By the address of a Dataset an evaluator step reads, the union of the statistics every evaluator step reading
    that Dataset asks for, where it is wider than this step's own request. A list the step runs over is keyed by the
    list's address; an element of it another step names alone, by the element's."""


@dataclass(frozen=True)
class RunSettings:
    """What every step of one task's run shares."""

    task: str
    pipeline: "PipelineConfig"
    data_dir: Path | None = None
    cache_dir: Path | None = None
    output_dir: Path | None = None
    extractors: Mapping[str | None, ExtractorSetup | None] = field(default_factory=dict)
    """By extractor name. ``None`` holds the task's own extractor, which every node's context already carries."""
    step_contexts: Mapping[str, StepContext] = field(default_factory=dict)
    runners: "Mapping[str, Workflow[Any, Any] | Evaluator[Any, Any]]" = field(default_factory=dict)
    """By step name, the instance an evaluator or workflow step runs, where the caller made it already, as a one-step
    task makes its own. Any other such step makes a fresh instance of its type."""


@dataclass
class ChainRun:
    """What running a graph produced."""

    steps: dict[str, StepResult]
    nodes: dict[str, Node | NodeList | Missing]
    lineage: list[LineageRecord]
    label_space: list[LabelSpaceRecord]
    reads: list[MetadataRead] = field(default_factory=list)
    """Each Metadata a step read, in the order read, for the result's binning record."""


def input_node(address: str, context: "DatasetContext", *, source: str, cache_name: str, cache_key: str) -> Node:
    """A chain input: the source's context, named for `address`, keyed exactly as the source is today."""
    if context.name != address:
        context = replace(context, name=address)
    root = Root(
        source=source,
        cache_name=cache_name,
        value_range=context.value_range,
        channel_groups=context.channel_groups,
        label_source=context.label_source,
    )
    return Node(address, DataType.DATASET, context=context, key=cache_key, roots=(root,), source=source)


def bind_inputs(
    graph: ChainGraph,
    source_names: Sequence[str],
    contexts: Mapping[str, "DatasetContext"],
    resolved: Mapping[str, "ResolvedSource"],
) -> dict[str, Node | NodeList]:
    """Bind a task's sources to the graph's slots in order; a list slot takes the rest, keyed by source name."""
    bound: dict[str, Node | NodeList] = {}
    names = list(source_names)
    for index, slot in enumerate(graph.slots):
        if slot.is_list:
            rest = names[index:]
            bound[slot.name] = NodeList(
                slot.name,
                {name: _source_node(f"{slot.name}[{name}]", name, contexts, resolved) for name in rest},
            )
        else:
            bound[slot.name] = _source_node(slot.name, names[index], contexts, resolved)
    return bound


def _source_node(
    address: str, name: str, contexts: Mapping[str, "DatasetContext"], resolved: Mapping[str, "ResolvedSource"]
) -> Node:
    source = resolved[name]
    return input_node(address, contexts[name], source=name, cache_name=source.cache_name, cache_key=source.cache_key)


def run_chain(graph: ChainGraph, inputs: Mapping[str, Node | NodeList], settings: RunSettings) -> ChainRun:
    """Run each step in order. Never raises for a step's failure: it becomes the step's status.

    Each Metadata a step reads is noted, for the result's binning record.
    """
    nodes: dict[str, _Value] = dict(inputs)
    # A one-step graph is a rerouted task: its result carries no lineage, so measure nothing it would not.
    tracked = not graph.one_step
    lineage = [_lineage(node) for node in _datasets(inputs.values())] if tracked else []
    steps: dict[str, StepResult] = {}
    label_space: list[LabelSpaceRecord] = []
    with noting_reads() as reads:
        for spec in graph.steps:
            record, produced, records = _run_step(spec, nodes, settings, lineage, label_space, steps)
            steps[spec.name] = record
            nodes.update(produced)
            # A preset step's declared output reads what the spliced step made.
            for address, value in produced.items():
                for alias in graph.aliases_of(address):
                    nodes[alias] = value
            if tracked:
                lineage.extend(_lineage(node) for node in _datasets(produced.values()))
            label_space.extend(records)
    return ChainRun(steps, nodes, lineage, label_space, reads)


def _lookup(nodes: Mapping[str, _Value], address: Address) -> _Value:
    base = str(address.base)
    value = nodes.get(base, Missing("was not produced"))
    if address.key is None or isinstance(value, Missing):
        return value
    if isinstance(value, NodeList):
        return value.elements.get(address.key, Missing(f"has no element `{address.key}`"))
    return Missing("is not a list")


def _run_step(
    spec: StepSpec,
    nodes: Mapping[str, _Value],
    settings: RunSettings,
    lineage: Sequence[LineageRecord],
    applied: Sequence[LabelSpaceRecord],
    steps: Mapping[str, StepResult],
) -> tuple[StepResult, dict[str, _Value], list[LabelSpaceRecord]]:
    """Run `spec` once, or once per key of the lists it broadcasts over; its outputs come back by address.

    `applied` holds the label spaces the steps before it applied, in chain order, and `steps` their records, which
    say why an input holds nothing. A check reading an input that holds nothing is not assessed, rather than skipped.
    """
    inputs_text = [str(address) for binding in spec.bindings for address in binding.addresses]
    bound = {binding.port.name: [_lookup(nodes, address) for address in binding.addresses] for binding in spec.bindings}
    gap = _first_gap(spec, bound, steps)
    if gap is not None:
        address, missing = gap
        if spec.kind == "check":
            record, outputs = _unassessed(spec, inputs_text, _gap_text(address, missing, steps), None)
            return record, _by_address(spec, outputs), []
        reason = f"needs `{address}`, which {missing.reason}"
        return _skipped(spec, inputs_text, reason), _by_address(spec, _missing_outputs(spec, "was skipped")), []
    keys = _broadcast_keys(spec, bound)
    if keys is None:
        record, outputs, records = _attempt(spec, _shaped(spec, bound), settings, None, inputs_text, lineage, applied)
        return record, _by_address(spec, outputs), records
    return _broadcast(spec, bound, keys, settings, inputs_text, lineage, applied, steps)


def _first_gap(
    spec: StepSpec, bound: Mapping[str, list[_Value]], steps: Mapping[str, StepResult]
) -> tuple[Address, Missing] | None:
    """The first address `spec` reads that holds nothing, and why; ``None`` when every one holds something.

    A check is never skipped for want of input (spec §9.1), so a list it takes whole holds nothing when no element
    of it exists.
    """
    for binding in spec.bindings:
        for address, value in zip(binding.addresses, bound[binding.port.name], strict=True):
            if isinstance(value, Missing):
                return address, value
            if spec.kind == "check" and binding.port.is_list and isinstance(value, NodeList) and not value.present:
                return address, _empty_list(address, value, steps)
    return None


def _empty_list(address: Address, value: NodeList, steps: Mapping[str, StepResult]) -> Missing:
    """Why a list holds no element: none exists, or the first one's gap."""
    if not value.elements:
        return Missing("holds no element")
    key, first = next(iter(value.elements.items()))
    inner = _gap_text(replace(address, key=key), first, steps) if isinstance(first, Missing) else ""
    return Missing(f"holds no element; {inner}" if inner else "holds no element")


def _gap_text(address: Address, missing: Missing, steps: Mapping[str, StepResult]) -> str:
    """What a check could not assess: "`count` failed: ValueError: ...", with the cause its producer recorded."""
    record = steps.get(address.name)
    if record is not None and address.key is not None and record.elements is not None:
        record = record.elements.get(address.key)
    cause = None
    if record is not None:
        cause = "; ".join(record.errors) if record.status == "failed" else record.reason
    return f"`{address}` {missing.reason}" + (f": {cause}" if cause else "")


def _check_title(spec: StepSpec) -> str:
    """What a check's finding is titled where the check makes none itself: its `subject`, if any, else its title."""
    return getattr(spec.config, "subject", None) or spec.impl.title  # type: ignore[attr-defined]  # every check has one


def _unassessed(
    spec: StepSpec, inputs_text: list[str], gap: str, element: str | None
) -> tuple[StepResult, dict[str, _Value]]:
    """A check whose input holds nothing: never skipped, it reports one ``info`` finding saying why (spec §9.1)."""
    finding = Finding(
        severity="info",
        title=_check_title(spec) + (f" by {spec.by.label}" if spec.by is not None else ""),
        brief="not assessed",
        description=f"Not assessed: {gap}.",
        step=_finding_step(spec, element),
    )
    record = StepResult(
        name=spec.name,
        kind=spec.kind,
        type=spec.type,
        inputs=inputs_text,
        status="ok",
        output=[finding],
        summary=_tally([finding]),
        optional=spec.optional,
    )
    (port,) = spec.outputs
    node = Node(_at(spec, port, element), DataType.FINDINGS, payload=[finding], step=spec.name, step_type=spec.type)
    return record, {port.name: node}


def _broadcast_keys(spec: StepSpec, bound: Mapping[str, list[_Value]]) -> list[str] | None:
    """The keys `spec` runs once each for, in order: of every list on a port that takes one item; ``None`` if none."""
    lists = [
        value
        for binding in spec.bindings
        if not binding.port.is_list
        for value in bound[binding.port.name]
        if isinstance(value, NodeList)
    ]
    if not lists:
        return None
    keys: list[str] = []
    for value in lists:
        keys.extend(key for key in value.elements if key not in keys)
    return keys


def _broadcast(
    spec: StepSpec,
    bound: Mapping[str, list[_Value]],
    keys: list[str],
    settings: RunSettings,
    inputs_text: list[str],
    lineage: Sequence[LineageRecord],
    applied: Sequence[LabelSpaceRecord],
    steps: Mapping[str, StepResult],
) -> tuple[StepResult, dict[str, _Value], list[LabelSpaceRecord]]:
    """Run `spec` once per key, zipping its lists by key; each output is a list with those keys."""
    elements: dict[str, StepResult] = {}
    per_output: dict[str, dict[str, Node | Missing]] = {port.name: {} for port in spec.outputs}
    label_space: list[LabelSpaceRecord] = []
    for key in keys:
        chosen, gap = _pick(spec, bound, key)
        element_inputs = _element_inputs(spec, bound, key)
        if gap is not None and spec.kind == "check":
            elements[key], outputs = _unassessed(spec, element_inputs, _element_gap(gap, key, steps), key)
        elif gap is not None:
            elements[key] = _skipped(spec, element_inputs, _element_reason(gap, key))
            outputs = _missing_outputs(spec, "was skipped")
        else:
            elements[key], outputs, records = _attempt(
                spec, _shaped(spec, chosen), settings, key, element_inputs, lineage, applied
            )
            label_space.extend(records)
        for port in spec.outputs:
            per_output[port.name][key] = outputs[port.name]  # type: ignore[assignment]  # lists do not nest
    record = StepResult(
        name=spec.name,
        kind=spec.kind,
        type=spec.type,
        inputs=inputs_text,
        status=_overall(elements.values()),
        elements=elements,
        elapsed=sum(element.elapsed for element in elements.values()),
        optional=spec.optional,
    )
    produced: dict[str, _Value] = {
        spec.output_address(port): NodeList(spec.output_address(port), per_output[port.name]) for port in spec.outputs
    }
    return record, produced, label_space


def _pick(
    spec: StepSpec, bound: Mapping[str, list[_Value]], key: str
) -> tuple[dict[str, list[Any]], tuple[Address, Missing | None] | None]:
    """Each port's values for element `key`, and the first list holding nothing there: its address, and why.

    The why is ``None`` where that list has no element `key` at all; the gap is ``None`` when every list holds one.
    """
    chosen: dict[str, list[Any]] = {}
    gap: tuple[Address, Missing | None] | None = None
    for binding in spec.bindings:
        picked: list[Any] = []
        for address, value in zip(binding.addresses, bound[binding.port.name], strict=True):
            if isinstance(value, NodeList) and not binding.port.is_list:
                element = value.elements.get(key)
                if gap is None and not isinstance(element, Node):
                    gap = (address, element)
                picked.append(element)
            else:
                picked.append(value)
        chosen[binding.port.name] = picked
    return chosen, gap


def _element_inputs(spec: StepSpec, bound: Mapping[str, list[_Value]], key: str) -> list[str]:
    """The addresses element `key` of a broadcast reads: each list on a port that takes one item, narrowed to `key`."""
    return [
        f"{address}[{key}]" if isinstance(value, NodeList) and not binding.port.is_list else str(address)
        for binding in spec.bindings
        for address, value in zip(binding.addresses, bound[binding.port.name], strict=True)
    ]


def _element_reason(gap: tuple[Address, Missing | None], key: str) -> str:
    """Why element `key` of a step cannot run: its list has no such element, or holds nothing there."""
    address, missing = gap
    if missing is None:
        return f"`{address}` has no element `{key}`"
    return f"needs `{address}[{key}]`, which {missing.reason}"


def _element_gap(gap: tuple[Address, Missing | None], key: str, steps: Mapping[str, StepResult]) -> str:
    """What element `key` of a check could not assess, with the cause its producer recorded for that element."""
    address, missing = gap
    if missing is None:
        return f"`{address}` has no element `{key}`"
    return _gap_text(replace(address, key=key), missing, steps)


def _overall(elements: Iterable[StepResult]) -> StepStatus:
    """A broadcast step's status: failed if any element failed, skipped if every one did, else ok."""
    statuses = {element.status for element in elements}
    if "failed" in statuses:
        return "failed"
    return "skipped" if statuses == {"skipped"} else "ok"


def _shaped(spec: StepSpec, bound: Mapping[str, list[Any]]) -> dict[str, Any]:
    """Each port's value as a step reads it: one node, a list of nodes, or a whole keyed list."""
    shaped: dict[str, Any] = {}
    for binding in spec.bindings:
        values = bound[binding.port.name]
        many = binding.port.count is not None or len(values) > 1
        shaped[binding.port.name] = values if many else (values[0] if values else None)
    return shaped


def _attempt(
    spec: StepSpec,
    inputs: Mapping[str, Any],
    settings: RunSettings,
    element: str | None,
    inputs_text: list[str],
    lineage: Sequence[LineageRecord],
    applied: Sequence[LabelSpaceRecord],
) -> tuple[StepResult, dict[str, _Value], list[LabelSpaceRecord]]:
    """Run one invocation of a step; its outputs come back by port name. Every exception becomes the step's failure."""
    start = time.monotonic()
    records: list[LabelSpaceRecord] = []
    result: Any = None
    details: dict[str, Any] | None = None
    try:
        if spec.kind in ("evaluator", "workflow"):
            result = _pooled(spec, inputs, settings, element)
            if not result.success:
                failed = _failed(spec, inputs_text, list(result.errors), start, result)
                return failed, _missing_outputs(spec, _failure_word(spec)), []
            outputs = _pooled_outputs(spec, result, inputs, element)
        elif spec.kind == "combine":
            outputs = _combine(spec, inputs, settings, element)
        elif spec.kind == "check":
            outputs = _check(spec, inputs, settings, element)
        else:
            outputs, records, details = _transform(spec, inputs, settings, element, lineage, applied)
    except StepSkipped as skip:
        return _skipped(spec, inputs_text, skip.reason), _missing_outputs(spec, "was skipped"), []
    except Exception as error:  # a step's failure must not stop the chain
        message = failure_message(error)
        if spec.optional:  # a failure the config expects: a warning, without the traceback
            _logger.warning("Optional step '%s' failed, so it is skipped: %s", spec.name, message)
        else:
            _logger.exception("Step '%s' failed", spec.name)
        failed = _failed(spec, inputs_text, [message], start, result)
        return failed, _missing_outputs(spec, _failure_word(spec)), []
    value = {name: _live(item) for name, item in outputs.items()}
    summary = _step_summary(spec, outputs)
    record = StepResult(
        name=spec.name,
        kind=spec.kind,
        type=spec.type,
        inputs=inputs_text,
        status="ok",
        output=next(iter(value.values())) if len(value) == 1 else value,
        elapsed=time.monotonic() - start,
        result=result,
        optional=spec.optional,
        summary=summary,
        details=details,
    )
    return record, outputs, records


def _summarize(item: Node | NodeList | Missing) -> Any:
    """A transform output's JSON: a Dataset's size and digest, a list's per key, a record's dump."""
    if isinstance(item, NodeList):
        return {key: _summarize(value) for key, value in item.elements.items()}
    if isinstance(item, Missing):
        return None
    if item.type is DataType.DATASET:
        return {"items": len(item.value), "digest": short_digest(item.key or item.address)}
    payload = item.payload
    return payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload


def _step_summary(spec: StepSpec, outputs: Mapping[str, _Value]) -> Any:
    """What a step's JSON says it made: a transform's or combine's outputs, a check's tally; ``None`` otherwise."""
    if spec.kind == "check":
        (findings,) = outputs.values()
        return _tally(_live(findings))
    if spec.kind not in ("transform", "combine"):
        return None
    summary = {name: _summarize(item) for name, item in outputs.items()}
    return next(iter(summary.values())) if len(summary) == 1 else summary


def _tally(findings: Sequence[Finding]) -> dict[str, int]:
    """A check step's JSON output: how many findings it made, and how many are warnings."""
    return {"findings": len(findings), "warnings": sum(finding.severity == "warning" for finding in findings)}


def _finding_step(spec: StepSpec, element: str | None) -> str:
    """What a check's finding names as its step: the step, with the element's key where it ran once per element."""
    return spec.name if element is None else f"{spec.name}[{element}]"


def _combine(
    spec: StepSpec, inputs: Mapping[str, Any], settings: RunSettings, element: str | None
) -> dict[str, _Value]:
    """Run a combine; each Output it makes is stored with the Datasets it was computed on, and their size."""
    impl: Combine[Any] = spec.impl()  # type: ignore[assignment]
    step = settings.step_contexts.get(spec.name, StepContext())
    context = CombineContext(
        task=settings.task,
        step=spec.name,
        derive_metadata=lambda node: _metadata(node, step.metadata_policy, _policy_name(spec)),
        derive_stats=lambda node: _stats(node, step.stats_policy),
    )
    made = impl.run(spec.config, inputs, context)
    missing = [port.name for port in spec.outputs if port.name not in made]
    if missing:
        raise TypeError(
            f"combine '{spec.type}' returned no `{missing[0]}`: a combine returns every output port it declares."
        )
    computed_on = _computed_on(inputs)
    return {
        port.name: Node(
            _at(spec, port, element),
            port.type,
            payload=made[port.name],
            step=spec.name,
            step_type=spec.type,
            inputs=tuple(node.address for node in computed_on),
            computed_on=computed_on,
            config=spec.config,
        )
        for port in spec.outputs
    }


def _check(spec: StepSpec, inputs: Mapping[str, Any], settings: RunSettings, element: str | None) -> dict[str, _Value]:
    """Run a check; each finding is stamped with the step, and with the element's key where it ran once per element."""
    impl: Check[Any] = spec.impl()  # type: ignore[assignment]
    context = CheckContext(task=settings.task, step=spec.name)
    if spec.by is not None:
        (port,) = (binding.port for binding in spec.bindings)
        node = inputs[port.name]
        per_class = node.value
        found_by_key = {
            key: _findings(spec, impl.run(spec.config, {port.name: replace(node, payload=output)}, context))
            for key, output in per_class.outputs.items()
        }
        found = [roll_up(found_by_key, per_class.skipped, title=_check_title(spec), by=spec.by)]
    else:
        found = _findings(spec, impl.run(spec.config, inputs, context))
    stamp = _finding_step(spec, element)
    findings = [finding.model_copy(update={"step": stamp}) for finding in found]
    (port,) = spec.outputs
    node = Node(_at(spec, port, element), DataType.FINDINGS, payload=findings, step=spec.name, step_type=spec.type)
    return {port.name: node}


def _findings(spec: StepSpec, returned: Any) -> list[Finding]:
    """What one run of a check returned, refused unless it is a list of findings."""
    if isinstance(returned, Finding):
        raise TypeError(f"check '{spec.type}' returned a Finding, not a list of findings.")
    found = list(returned)
    strays = [type(item).__name__ for item in found if not isinstance(item, Finding)]
    if strays:
        raise TypeError(f"check '{spec.type}' returned {', '.join(strays)}, not findings.")
    return found


def _computed_on(inputs: Mapping[str, Any]) -> tuple[Node, ...]:
    """The Dataset nodes an Output made from `inputs` was computed on.

    Made from Datasets, it was computed on them; made from Outputs alone, on what the first of those was.
    """
    datasets = _datasets(inputs.values())
    if datasets:
        return tuple(datasets)
    first = next(iter(_among(inputs.values(), DataType.OUTPUT)), None)
    return first.computed_on if first is not None else ()


def _at(spec: StepSpec, port: Port, element: str | None) -> str:
    """Where one run of `spec` stores `port`'s output: its address, narrowed to `element` in a broadcast."""
    return spec.output_address(port) + (f"[{element}]" if element is not None else "")


def _failure_word(spec: StepSpec) -> str:
    return "was skipped" if spec.optional else "failed"


def _failed(spec: StepSpec, inputs_text: list[str], errors: list[str], start: float, result: Any) -> StepResult:
    optional = spec.optional
    return StepResult(
        name=spec.name,
        kind=spec.kind,
        type=spec.type,
        inputs=inputs_text,
        status="skipped" if optional else "failed",
        reason=f"failed: {'; '.join(errors)}" if optional else None,
        errors=errors,
        elapsed=time.monotonic() - start,
        result=result,
        optional=optional,
    )


def _skipped(spec: StepSpec, inputs_text: list[str], reason: str) -> StepResult:
    return StepResult(
        name=spec.name,
        kind=spec.kind,
        type=spec.type,
        inputs=inputs_text,
        status="skipped",
        reason=reason,
        optional=spec.optional,
    )


def _missing_outputs(spec: StepSpec, reason: str) -> dict[str, _Value]:
    """Each of `spec`'s outputs, by port name, holding nothing for `reason`."""
    return {port.name: Missing(reason) for port in spec.outputs}


def _by_address(spec: StepSpec, outputs: Mapping[str, _Value]) -> dict[str, _Value]:
    """`outputs`, by port name, keyed instead by the address each is stored at."""
    return {spec.output_address(port): outputs[port.name] for port in spec.outputs}


def _live(item: _Value) -> Any:
    if isinstance(item, NodeList):
        return {key: _live(value) for key, value in item.elements.items()}
    return item.value if isinstance(item, Node) else None


def _pooled(spec: StepSpec, inputs: Mapping[str, Any], settings: RunSettings, element: str | None) -> Any:
    """An evaluator's or a workflow type's result, from a WorkflowContext over its input nodes.

    An evaluator step reading a Dataset other evaluator steps read also gets, by node, the union of their stats
    requests that preflight planned, so the first of them computes the statistics every one reads in one pass.
    """
    nodes = _datasets(inputs.values())
    setup = settings.extractors.get(spec.extractor)
    contexts = {node.address: _context_for(node, spec, setup) for node in nodes}
    step = settings.step_contexts.get(spec.name, StepContext())
    context = ReadingContext(
        dataset_contexts=contexts,
        batch_size=setup.batch_size if setup is not None else None,
        metadata_policy=step.metadata_policy,
        ontology=step.ontology,
        stats_policy=step.stats_policy,
        policy_name=_policy_name(spec),
    )
    # Run as the orchestrator runs a task's target, so a step and a task run it the same way.
    from dataeval_flow._orchestrator import _run_target

    runner = settings.runners[spec.name] if spec.name in settings.runners else spec.impl()
    unions = {node.address: union for node in nodes if (union := _stats_union(step, node, element)) is not None}
    # Each is passed only when set, so without them the call is exactly a task's.
    extra: dict[str, Any] = {"stats_unions": unions} if unions else {}
    if spec.by is not None:
        extra["by"] = spec.by
    return _run_target(runner, spec.config, context, **extra)  # type: ignore[arg-type]


def _stats_union(step: StepContext, node: Node, element: str | None) -> "ResolvedStatsPolicy | None":
    """The stats union preflight planned for `node`: by its address, or, as element `element` of a list the step runs
    over, by the list's; ``None`` where it planned none."""
    union = step.stats_unions.get(node.address)
    if union is None and element is not None:
        listed = (
            planned
            for address, planned in step.stats_unions.items()
            if str(Address(address, key=element)) == node.address
        )
        union = next(listed, None)
    return union


def _context_for(node: Node, spec: StepSpec, setup: ExtractorSetup | None) -> "DatasetContext":
    context = node.context
    if context is None:
        raise RuntimeError(f"Dataset node `{node.address}` has no context")
    if spec.extractor is None or setup is None:
        return context
    return replace(context, extractor=setup.config, transforms=setup.transforms, batch_size=setup.batch_size)


def _pooled_outputs(spec: StepSpec, result: Any, inputs: Mapping[str, Any], element: str | None) -> dict[str, _Value]:
    (port,) = spec.outputs
    payload = result.output if port.type is DataType.OUTPUT else result
    datasets = _datasets(inputs.values())
    on = tuple(node.address for node in datasets)
    return {
        port.name: Node(
            _at(spec, port, element),
            port.type,
            payload=payload,
            step=spec.name,
            step_type=spec.type,
            inputs=on,
            result=result,
            computed_on=tuple(datasets) if port.type is DataType.OUTPUT else (),
            config=spec.config if port.type is DataType.OUTPUT else None,
        )
    }


def _transform(
    spec: StepSpec,
    inputs: Mapping[str, Any],
    settings: RunSettings,
    element: str | None,
    lineage: Sequence[LineageRecord],
    applied: Sequence[LabelSpaceRecord],
) -> tuple[dict[str, _Value], list[LabelSpaceRecord], dict[str, Any] | None]:
    impl: Transform[Any] = spec.impl()  # type: ignore[assignment]
    step = settings.step_contexts.get(spec.name, StepContext())
    sources = _datasets(inputs.values())
    ancestors = {record.name for node in sources for record in _ancestry(node.address, lineage)}
    context = TransformContext(
        task=settings.task,
        step=spec.name,
        output_dir=settings.output_dir,
        pipeline=settings.pipeline,
        data_dir=settings.data_dir,
        derive_metadata=lambda node: _metadata(node, step.metadata_policy, _policy_name(spec)),
        lineage=lambda address: _ancestry(address, lineage),
        label_space=tuple(record for record in applied if record.source in ancestors),
        element=element,
    )
    made = impl.run(spec.config, inputs, context)
    _check_datasets(spec, made)
    first = _at(spec, spec.outputs[0], element)
    records = impl.label_space(spec.config, inputs, made, address=first)
    digest = impl.digest(spec.config, inputs, made)
    details = impl.details(spec.config, inputs, made)
    base = step_key(
        spec.type, settings_of(spec.config, spec.impl.input_ports()), [node.key or "" for node in sources], digest
    )
    outputs: dict[str, _Value] = {}
    for port in spec.outputs:
        address = _at(spec, port, element)
        key = base if len(spec.outputs) == 1 else output_key(base, port.name)
        value = made[port.name]
        if port.type is DataType.DATASET and port.is_list:
            outputs[port.name] = NodeList(
                address,
                {
                    k: _made_node(f"{address}[{k}]", ds, element_key(key, k), sources, spec, settings)
                    for k, ds in value.items()
                },
            )
        elif port.type is DataType.DATASET:
            outputs[port.name] = _made_node(address, value, key, sources, spec, settings)
        else:
            on = tuple(node.address for node in sources)
            outputs[port.name] = Node(address, port.type, payload=value, step=spec.name, step_type=spec.type, inputs=on)
    return outputs, records, details


def _check_datasets(spec: StepSpec, made: Mapping[str, Any]) -> None:
    """Refuse a Dataset output with no length here, where it fails only its step: lineage and the JSON measure it."""
    for port in spec.outputs:
        if port.type is not DataType.DATASET:
            continue
        value = made[port.name]
        for dataset in value.values() if port.is_list else [value]:
            if not isinstance(dataset, Sized):
                raise TypeError(
                    f"transform '{spec.type}' returned {type(dataset).__name__} for `{port.name}`, which has no "
                    "length: a Dataset must have one."
                )


def _made_node(
    address: str, dataset: Any, key: str, sources: list[Node], spec: StepSpec, settings: RunSettings
) -> Node:
    """A Dataset a transform made, with a context that derives under its own key."""
    from dataeval_flow._cache import DatasetCache
    from dataeval_flow._orchestrator import _label_source_of, _merge_channel_groups
    from dataeval_flow.workflows._context import DatasetContext

    roots: list[Root] = []
    for node in sources:
        roots.extend(root for root in node.roots if root not in roots)
    ranges = {root.value_range for root in roots if root.value_range is not None}
    setup = settings.extractors.get(None)
    context = DatasetContext(
        name=address,
        dataset=dataset,
        extractor=setup.config if setup is not None else None,
        transforms=setup.transforms if setup is not None else None,
        batch_size=setup.batch_size if setup is not None else None,
        label_source=_label_source_of(_label_sources(roots)),
        value_range=next(iter(ranges)) if len(ranges) == 1 else None,
        channel_groups=_merge_channel_groups((root.channel_groups for root in roots), f"Step '{spec.name}'"),
        cache=DatasetCache.get_or_create(
            settings.cache_dir, name="+".join(root.cache_name for root in roots), cache_key=key
        ),
    )
    return Node(
        address,
        DataType.DATASET,
        context=context,
        key=key,
        roots=tuple(roots),
        step=spec.name,
        step_type=spec.type,
        inputs=tuple(node.address for node in sources),
    )


def _label_sources(roots: Sequence[Root]) -> list[str | None]:
    """Each root's label provenance, one per operand: a root merged from several gives each of its own."""
    found: list[str | None] = []
    for root in roots:
        if root.label_source is None or isinstance(root.label_source, str):
            found.append(root.label_source)
        else:
            found.extend(root.label_source)
    return found


def _policy_name(spec: StepSpec) -> str | None:
    """The metadata policy the step's entry names, ``None`` where it names none: how the binning record tells two
    readings of one Dataset apart."""
    name = getattr(spec.config, "metadata", None)
    return name if isinstance(name, str) else None


def _metadata(node: Node, policy: Any, policy_name: str | None) -> Any:
    """`node`'s Metadata under `policy`, cached on the node, and noted for the chain's binning record."""
    from dataeval_flow._cache import active_cache, get_or_compute_metadata, selection_repr

    dataset = node.value
    cache = node.context.cache if node.context is not None else None
    scope = active_cache(cache, selection_repr(dataset)) if cache is not None else contextlib.nullcontext()
    with scope:
        metadata = get_or_compute_metadata(dataset, policy)
    note_read(node.address, policy_name, policy, metadata)
    return metadata


def _stats(node: Node, policy: "ResolvedStatsPolicy | None") -> Any:
    """`node`'s per-image statistics under `policy`, every statistic where the step names none, cached on the node."""
    from dataeval.flags import ImageStats

    from dataeval_flow._cache import active_cache, get_or_compute_stats, selection_repr
    from dataeval_flow._stats import ResolvedStatsPolicy

    dataset = node.value
    cache = node.context.cache if node.context is not None else None
    value_range = node.context.value_range if node.context is not None else None
    scope = active_cache(cache, selection_repr(dataset)) if cache is not None else contextlib.nullcontext()
    with scope:
        return get_or_compute_stats(
            policy if policy is not None else ResolvedStatsPolicy.of_flags(ImageStats.ALL),
            dataset=dataset,
            per_image=True,
            per_target=False,
            value_range=value_range,
        )


def _datasets(values: Iterable[Any]) -> list[Node]:
    """The Dataset nodes among `values`: each node, each node of a port fed several, and each present element of a
    list. Lineage records no Output, and a step's inputs are the Datasets it read."""
    return _among(values, DataType.DATASET)


def _among(values: Iterable[Any], data_type: DataType) -> list[Node]:
    """The nodes of `data_type` among `values`, however a port holds them."""
    found: list[Node] = []
    for value in values:
        items = value.present.values() if isinstance(value, NodeList) else value if isinstance(value, list) else [value]
        found.extend(item for item in items if isinstance(item, Node) and item.type is data_type)
    return found


def _ancestry(address: str, lineage: Sequence[LineageRecord]) -> list[LineageRecord]:
    """`address`'s lineage record, then its ancestors', nearest first."""
    records = {record.name: record for record in lineage}
    found: list[LineageRecord] = []
    queue = [address]
    while queue:
        name = queue.pop(0)
        record = records.get(name)
        if record is None or record in found:
            continue
        found.append(record)
        queue.extend(record.inputs)
    return found


def _lineage(node: Node) -> LineageRecord:
    return LineageRecord(
        name=node.address,
        step=node.step,
        type=node.step_type,
        inputs=list(node.inputs),
        source=node.source,
        digest=short_digest(node.key or node.address),
        items=len(node.value),
    )
