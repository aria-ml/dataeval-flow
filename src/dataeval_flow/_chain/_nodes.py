"""What flows between steps at run time: nodes, keyed lists of them, and why one is missing."""

__all__ = ["Missing", "Node", "NodeList", "Root"]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dataeval_flow.steps._port import DataType

if TYPE_CHECKING:
    from dataeval_flow.workflows._context import DatasetContext


@dataclass(frozen=True)
class Root:
    """A chain input's source: what every node made from it inherits for caching and reading pixels."""

    source: str
    cache_name: str
    value_range: tuple[float, float] | None = None
    channel_groups: Mapping[str, tuple[int, ...]] | None = None
    label_source: str | Sequence[str] | None = None


@dataclass(frozen=True)
class Missing:
    """Why an address holds nothing at run time: `reason` completes "which ...", e.g. ``failed``."""

    reason: str


@dataclass
class Node:
    """One value at an address: a Dataset (read through its context), an Output, a workflow result or a record."""

    address: str
    type: DataType
    payload: Any = None
    context: "DatasetContext | None" = None
    key: str | None = None
    kind: str | None = None
    roots: tuple[Root, ...] = ()
    step: str | None = None
    step_type: str | None = None
    inputs: tuple[str, ...] = ()
    source: str | None = None
    result: Any = None
    computed_on: tuple["Node", ...] = ()
    """For an Output, the Dataset nodes it was computed on."""
    config: Any = None
    """For an Output, the settings of the step that made it: an evaluator's pool entry, or an inline step's config;
    ``None`` for anything else."""
    _dataset: Any = field(default=None, repr=False)

    @property
    def items(self) -> int | None:
        """For an Output, how many items the Datasets it was computed on hold together, measured when first read;
        ``None`` for anything else."""
        if self.type is not DataType.OUTPUT or not self.computed_on:
            return None
        return sum(len(node.value) for node in self.computed_on)

    @property
    def value(self) -> Any:
        """The Dataset, with its context's view applied, or the payload of any other node."""
        if self.type is not DataType.DATASET:
            return self.payload
        if self._dataset is None:
            from dataeval_flow._view import build_view

            context = self.context
            if context is None:
                raise RuntimeError(f"Dataset node `{self.address}` has no context")
            ops = context.view_operations
            self._dataset = build_view(context.dataset, list(ops)) if ops else context.dataset
        return self._dataset


@dataclass
class NodeList:
    """A keyed list of nodes, in order. An element is :class:`Missing` when the step making it failed or skipped."""

    address: str
    elements: dict[str, "Node | Missing"]
    reason: str | None = None
    """Why the list holds no element, where its readers should say so: an empty list input's `empty:`, carried on by
    each step run over it. ``None`` for any list that holds an element."""

    @property
    def present(self) -> dict[str, Node]:
        """The elements that exist, in order."""
        return {key: value for key, value in self.elements.items() if isinstance(value, Node)}
