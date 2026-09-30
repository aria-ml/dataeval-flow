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
# Each level as a report says it, singular and plural: an item of a Dataset is an image.
_NOUNS: dict[str, tuple[str, str]] = {
    "items": ("image", "images"),
    "detections": ("detection", "detections"),
    "tracks": ("track", "tracks"),
    "frames": ("frame", "frames"),
}


def _counted(plan: RemovalPlan) -> dict[str, int]:
    """How many rows a plan names at each level."""
    counts = dict.fromkeys(_LEVELS.values(), 0)
    for address in plan:
        counts[_LEVELS.get(address.kind, "detections")] += 1
    return counts


def _noun(level: str, count: int) -> str:
    """``count`` at ``level`` as a report says it: ``1 image``, ``2 detections``."""
    return f"{count} {_NOUNS[level][count != 1]}"


def _plan_count(levels: dict[str, int], *, single: bool) -> str:
    if not levels:
        return "0"
    return _list([str(count) if single else _noun(level, count) for level, count in levels.items()])


def _list(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"


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
    _named: dict[str, RemovalPlan]

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
        named: dict[str, RemovalPlan] = {}
        for (address, arguments), node in zip(config.plans.items(), nodes, strict=True):
            output = node.value
            named[address] = getattr(output, _method(type(output)))(**arguments)
            plan = plan | named[address]
        self._plan = plan
        self._named = named
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
        """How many rows were removed at each level, and how many each plan named, where plans may overlap."""
        by_plan = {
            address: {level: count for level, count in _counted(plan).items() if count}
            for address, plan in self._named.items()
        }
        return {"removed": _counted(self._plan), "by_plan": by_plan}

    def section(self, record: Any) -> list[Any]:
        """What was kept and removed, at the levels the Dataset has, and what each plan named."""
        from dataeval_flow._blocks import Paragraph

        details = record.details or {}
        removed = {level: count for level, count in details.get("removed", {}).items() if count}
        kept = len(record.output) if hasattr(record.output, "__len__") else None
        total = None if kept is None else kept + details.get("removed", {}).get("items", 0)
        lines = []
        if kept is not None:
            lines.append(f"Kept {kept} of {total} images.")
        if not removed:
            lines.append("Nothing removed.")
            return [Paragraph(text=" ".join(lines))]
        says = _list([_noun(level, count) for level, count in removed.items()])
        # Under one level the plans' counts need no noun; under several each says which level it counted.
        single = len(removed) == 1
        named = [
            f"{_plan_count(levels, single=single)} {'named by' if index == 0 else 'by'} `{address}`"
            for index, (address, levels) in enumerate(details.get("by_plan", {}).items())
        ]
        lines.append(f"Removed {says}{': ' + ', '.join(named) if named else ''}.")
        return [Paragraph(text=" ".join(lines))]
