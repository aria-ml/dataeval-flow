"""`remove`: removal plans from Duplicates and Outliers, applied to the Dataset they were computed on."""

__all__ = ["RemoveConfig", "RemoveTransform"]

import inspect
from collections.abc import Mapping
from typing import Any, ClassVar, get_type_hints

from dataeval.quality import DuplicatesOutput, OutliersOutput
from dataeval.types import RemovalPlan
from pydantic import Field, TypeAdapter, ValidationError

from dataeval_flow._chain._identity import plan_digest
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import Transform, TransformConfig, TransformContext

_METHODS: tuple[tuple[type, str], ...] = ((DuplicatesOutput, "deduplicate"), (OutliersOutput, "prune"))
# A plan address's kind (``SourceIndex.kind``) to the level its removal is counted under.
_LEVELS: dict[str | None, str] = {None: "items", "instance": "detections", "track": "tracks", "unit": "frames"}


def _method(cls: type) -> str:
    """The plan method of an Output class; ``TypeError`` for a class that has none."""
    found = next((name for owner, name in _METHODS if issubclass(cls, owner)), None)
    if found is None:
        accepted = " or ".join(owner.__name__ for owner, _ in _METHODS)
        raise TypeError(f"{cls.__name__} has no removal plan: `remove` reads a {accepted}.")
    return found


def _spelled(hint: Any) -> str:
    """A type hint as its source writes it: ``Literal['first', 'last']``, ``Sequence[int] | None``."""
    text = hint.__name__ if isinstance(hint, type) else str(hint)
    return text.replace("typing.", "").replace("collections.abc.", "")


def _arguments_problem(owner: type, arguments: Mapping[str, Any]) -> str | None:
    """Why `arguments` do not fit `owner`'s plan method, by name and then by type; ``None`` when they fit.

    Types are checked only where the method's hints resolve at run time; otherwise names alone are.
    """
    method = _method(owner)
    function = getattr(owner, method)
    taken = [name for name in inspect.signature(function).parameters if name != "self"]
    where = f"{owner.__name__}.{method}"
    unknown = sorted(set(arguments) - set(taken))
    if unknown:
        return f"passes {', '.join(unknown)}, which {where} does not take; it takes {', '.join(taken)}."
    try:
        hints = get_type_hints(function)
    except NameError:  # a hint DataEval imports only for type checkers: the names above are all that can be checked
        hints = {}
    for name, value in arguments.items():
        hint = hints.get(name, Any)
        # Strict, since `run` passes each value as written: lax mode would accept a "2" that reaches the method a str.
        try:
            TypeAdapter(hint).validate_python(value, strict=True)
        except ValidationError:
            return f"passes {name}: {value!r}, but {where} takes {name} as {_spelled(hint)}."
    return None


class RemoveConfig(TransformConfig):
    """A `remove` step's settings: its input, and each Output's plan method arguments, keyed by the Output's address."""

    input: str = Field(description="The Dataset to remove from; every plan must have been computed on it.")
    plans: dict[str, dict[str, Any]] = Field(
        min_length=1,
        description=(
            "The Outputs whose plans to apply, by address, each with its plan method's arguments: `deduplicate`'s for "
            "a Duplicates Output (`dup_types`, `keep`, `exclude_groups`, `levels`), `prune`'s for an Outliers Output "
            "(`metrics`, `min_flags`)."
        ),
    )


class RemoveTransform(Transform[RemoveConfig]):
    """``remove``: each Output's plan, combined with ``|``, applied with ``Indices(plan, exclude=True)``.

    The engine makes one instance per invocation, so the plan :meth:`run` combines is the one :meth:`digest` and
    :meth:`details` read.
    """

    name: ClassVar[str] = "remove"
    title: ClassVar[str] = "Remove"
    description: ClassVar[str] = "Removes the items, detections or tracks that Duplicates and Outliers plans name."
    inputs: ClassVar[tuple[Port, ...]] = (
        Port("input", DataType.DATASET),
        Port("plans", DataType.OUTPUT, classes=(DuplicatesOutput, OutliersOutput)),
    )
    outputs: ClassVar[tuple[Port, ...]] = (Port("output", DataType.DATASET),)
    same_node: ClassVar[tuple[str, ...]] = ("plans",)

    _plan: RemovalPlan

    @classmethod
    def bound_problem(cls, config: RemoveConfig, classes: Mapping[str, tuple[type, ...]]) -> str | None:
        """Refuse an argument the Output's plan method does not take, or a value of a type it does not take."""
        for address, arguments in config.plans.items():
            owner = next((c for c in classes.get(address, ()) if any(issubclass(c, o) for o, _ in _METHODS)), None)
            problem = _arguments_problem(owner, arguments) if owner is not None else None
            if problem is not None:
                return f"`plans: {address}` {problem}"
        return None

    def run(
        self,
        config: RemoveConfig,
        inputs: Mapping[str, Any],
        context: TransformContext,  # noqa: ARG002
    ) -> Mapping[str, Any]:
        """The input without every row the plans name."""
        from dataeval.data import Indices, View

        # One node when `plans` names one address, a list of them otherwise, in the mapping's order.
        nodes = inputs["plans"] if isinstance(inputs["plans"], list) else [inputs["plans"]]
        plan = RemovalPlan()
        for arguments, node in zip(config.plans.values(), nodes, strict=True):
            output = node.value
            plan = plan | getattr(output, _method(type(output)))(**arguments)
        self._plan = plan
        return {"output": View(inputs["input"].value, Indices(plan, exclude=True))}

    def digest(
        self,
        config: RemoveConfig,  # noqa: ARG002
        inputs: Mapping[str, Any],  # noqa: ARG002
        outputs: Mapping[str, Any],  # noqa: ARG002
    ) -> str:
        """The combined plan: equal plans key alike, whatever settings made them."""
        return plan_digest(self._plan)

    def details(
        self,
        config: RemoveConfig,  # noqa: ARG002
        inputs: Mapping[str, Any],  # noqa: ARG002
        outputs: Mapping[str, Any],  # noqa: ARG002
    ) -> dict[str, Any]:
        """How many rows were removed at each level."""
        counts = dict.fromkeys(_LEVELS.values(), 0)
        for address in self._plan:
            counts[_LEVELS.get(address.kind, "detections")] += 1
        return {"removed": counts}

    def section(self, record: Any) -> list[Any]:
        """The counts removed at each level."""
        from dataeval_flow._blocks import Fields

        removed = (record.details or {}).get("removed", {})
        return [Fields(items=[(level.title(), count) for level, count in removed.items()])]
