"""The independent units ``feasibility`` hands DataEval: items that may share a source are one group.

DataEval judges a target on groups, not items: the bounds count a group once, and the splits keep a group whole. Two
copies of one image in different groups would let a model that saw one score on the other, so the bounds would be
tighter than the evidence allows and a ``demonstrated_feasible`` could be wrong. The groups are the union of three
things, each of which says that items share a source: a view that repeats an item, a metadata factor's value, and a
group of exact or near duplicates.
"""

__all__ = ["Grouping", "group_items"]

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Grouping:
    """The groups of a source's items, and what joined them."""

    units: NDArray[np.intp]
    """One group id per item, numbered from 0 by the first item each group holds."""
    groups: int
    """How many groups the items fall into."""
    largest_group: int
    """How many items the largest group holds."""
    duplicate_merges: int
    """How many fewer groups there are because duplicates joined them: the groups the views and the factor made, less
    the groups there are."""


class _Components:
    """Items joined into groups, the first item of a group standing for it whatever order the joins come in."""

    def __init__(self, count: int) -> None:
        self._parent = list(range(count))

    def find(self, item: int) -> int:
        """The first item of `item`'s group."""
        while self._parent[item] != item:
            self._parent[item] = self._parent[self._parent[item]]
            item = self._parent[item]
        return item

    def join(self, first: int, second: int) -> None:
        """Make the groups of two items one."""
        low, high = sorted((self.find(first), self.find(second)))
        self._parent[high] = low

    def join_sharing(self, keys: Iterable[Any]) -> None:
        """Join the items that hold one key, in the order given; an item whose key is ``None`` joins no one."""
        seen: dict[Any, int] = {}
        for item, key in enumerate(keys):
            if key is not None:
                self.join(seen.setdefault(key, item), item)

    def count(self) -> int:
        """How many groups there are."""
        return len({self.find(item) for item in range(len(self._parent))})


def group_items(roots: Sequence[int], factor: Sequence[Any] | None, duplicates: Iterable[Sequence[int]]) -> Grouping:
    """Group the items of a source by what says they share a source.

    Parameters
    ----------
    roots : Sequence[int]
        Each item's index in the dataset at the bottom of the source's views. Items with one root are one item a view
        repeats.
    factor : Sequence[Any] or None
        Each item's value of the metadata factor that names its source, with ``None`` for an item without one. Items
        with one value share a source. An item without a value joins no one by it.
    duplicates : Iterable[Sequence[int]]
        Groups of items that are exact or near duplicates of each other, as item indices. Their members share a source.

    Returns
    -------
    Grouping
        The group of each item, and how many groups, how large the largest and how many fewer there are because of
        ``duplicates``.

    Raises
    ------
    ValueError
        When ``factor`` does not hold one value per item, or a duplicate group names an item that is not there.
    """
    count = len(roots)
    if factor is not None and len(factor) != count:
        raise ValueError(f"The factor has {len(factor)} values for {count} items; it needs one per item.")
    components = _Components(count)
    components.join_sharing(int(root) for root in roots)
    if factor is not None:
        components.join_sharing(factor)
    before = components.count()
    for members in duplicates:
        for item in members:
            if not 0 <= item < count:
                raise ValueError(f"A duplicate group names item {item}, but the source has {count} items.")
        for other in members[1:]:
            components.join(members[0], other)

    ids: dict[int, int] = {}
    units = np.array([ids.setdefault(components.find(item), len(ids)) for item in range(count)], dtype=np.intp)
    return Grouping(
        units=units,
        groups=len(ids),
        largest_group=int(np.bincount(units).max()) if count else 0,
        duplicate_merges=before - len(ids),
    )
