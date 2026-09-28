"""The items an image cell names, and how a renderer names one it has no thumbnail for."""

__all__ = ["item_name", "refs_in"]

from dataeval_flow._blocks._models import Cell, ItemRef


def refs_in(value: Cell) -> list[ItemRef]:
    """An image cell's item references: its one item, its group's, or none where it holds something else."""
    if isinstance(value, ItemRef):
        return [value]
    if isinstance(value, list):
        return [ref for ref in value if isinstance(ref, ItemRef)]
    return []


def item_name(ref: ItemRef) -> str:
    """``41``, or ``41 box 3`` for a detection box: the item as its row's other cells number it."""
    return str(ref.index) if ref.target is None else f"{ref.index} box {ref.target}"
