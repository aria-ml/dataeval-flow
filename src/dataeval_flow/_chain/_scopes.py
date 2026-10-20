"""A scope of a custom workflow as a splice: its steps started on inputs of their own and judged together, as a preset
run as a step is (preset decomposition spec §6.1)."""

__all__ = ["scope_splice"]

from typing import TYPE_CHECKING, Any

from dataeval_flow._chain._graph import _ARTICLE, GraphError, ValueType, _typed
from dataeval_flow._chain._presets import Splice
from dataeval_flow.steps._address import Address, parse_address
from dataeval_flow.steps._port import DataType
from dataeval_flow.steps._workflow import CustomWorkflowConfig, StepEntry, StepScopeConfig

if TYPE_CHECKING:
    from dataeval_flow.config._models import PipelineConfig


def scope_splice(
    scope: StepScopeConfig,
    workflow: CustomWorkflowConfig,
    pipeline: "PipelineConfig",
    types: dict[str, ValueType],
    later: set[str],
    empty: dict[str, frozenset[str]],
) -> Splice:
    """Scope `scope` of `workflow` as a splice, as the graph reaches its first step: refused unless each input is a
    Dataset made before then, and where it declares what a preset step's chain may, refused as that step would be.

    `types`, `later` and `empty` are the graph's as it stands: what each address holds, the steps not yet reached, and
    the outputs each step leaves empty.
    """
    entry = StepEntry.model_construct(name=scope.name)
    bound: dict[str, tuple[Address, ValueType]] = {}
    for slot, text in scope.inputs.items():
        address = parse_address(text)
        try:
            value = _typed(address, entry, workflow, types, later, empty)
        except GraphError as error:
            # Each of `_typed`'s messages opens with the reader, `Step '<name>' `: here, the scope's input.
            said = str(error).removeprefix(f"Step '{scope.name}' ")
            raise GraphError(
                f"Scope '{scope.name}' input `{slot}` {said}", code=error.code, scope=scope.name, port=slot
            ) from error
        if value.type is not DataType.DATASET:
            raise GraphError(
                f"Scope '{scope.name}' input `{slot}` reads `{address}`, which is {_ARTICLE[value.type]}, not a "
                "Dataset.",
                code="wrong_type",
                scope=scope.name,
                port=slot,
            )
        bound[slot] = (address, value)
    chain = scope.declared_chain()
    if chain.reference is not None or chain.record is not None or chain.blocking is not None:
        _check_lists(scope, bound)
    lists = [(address, value) for address, value in bound.values() if value.is_list and value.keys]
    keys = [key for _, value in lists for key in value.keys or ()]
    for accepted in chain.accepted if lists else ():
        if "[" in accepted and (element := accepted[accepted.index("[") + 1 : -1]) not in keys:
            raise GraphError(
                f"Scope '{scope.name}': `accepted` names `{accepted}`, but `{lists[0][0]}` has no element "
                f"`{element}`. Its elements: {', '.join(keys)}.",
                code="list_mismatch",
                scope=scope.name,
            )
    held = [step for step in workflow.steps if step.name in scope.steps]
    if chain.reference is not None:
        _check_reference_split(scope, held, pipeline)
    return Splice(scope.name, None, None, chain, dict(scope.inputs), tuple(step.name for step in held), prefix="")


def _check_lists(scope: StepScopeConfig, bound: dict[str, tuple[Address, ValueType]]) -> None:
    """Refuse a list bound to the scope's reference (audit-as-a-step spec D3), and an element of a list keyed like one
    of its inputs bound to one Dataset, which the record's columns and the preflight's names would confuse."""
    singles = {slot for slot, (_, value) in bound.items() if not value.is_list}
    for slot, (address, value) in bound.items():
        if slot == scope.reference and value.is_list:
            element = f"{address}[{value.keys[0] if value.keys else '<key>'}]"
            raise GraphError(
                f"Scope '{scope.name}' binds `{address}`, a list, to its reference `{slot}`, which takes one Dataset: "
                f"name one element, such as `{element}`.",
                code="list_mismatch",
                scope=scope.name,
                port=slot,
            )
        clash = next((key for key in value.keys or () if key in singles), None) if value.is_list else None
        if clash is not None:
            raise GraphError(
                f"Scope '{scope.name}' binds `{address}`, whose element `{clash}` has the name of its input `{clash}`: "
                "name the element otherwise with `keys:`.",
                scope=scope.name,
                port=slot,
            )


def _check_reference_split(scope: StepScopeConfig, held: list[StepEntry], pipeline: "PipelineConfig") -> None:
    """Refuse, under the scope's reference, a step of it whose config names a metadata policy with `reference_split`
    (audit-as-a-step spec D7): an evaluator step's pool entry, or an inline step's own config."""
    pool = {item.name: item for item in pipeline.evaluators or ()}
    for step in held:
        config: Any = pool.get(step.target) if step.kind == "evaluator" else step.config
        named = getattr(config, "metadata", None)
        policy = next((item for item in pipeline.metadata or () if item.name == named), None)
        if policy is not None and policy.reference_split is not None:
            raise GraphError(
                f"Scope '{scope.name}' holds step '{step.name}', whose metadata policy '{named}' names "
                f"reference_split={policy.reference_split!r}: a scope encodes like the Dataset bound to "
                f"`{scope.reference}`, and a reference_split, which names a task source, cannot single out a split "
                "part. Remove reference_split from the policy.",
                scope=scope.name,
            )
