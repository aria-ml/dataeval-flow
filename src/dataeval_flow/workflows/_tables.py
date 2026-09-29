"""Table shapes more than one workflow reports its findings in."""

__all__ = ["unlabelled_blocks"]

from collections.abc import Mapping, Sequence

from dataeval_flow._blocks import Block, Cell, Column, ItemRef, Section, Table
from dataeval_flow._tables import group_cells


def unlabelled_blocks(images: Mapping[str, Sequence[int]], *, header: str) -> list[Block]:
    """Each source's images with no labels, under their own heading: how many, and up to eight, named and pictured.

    *images* are each source's unlabelled images by index, and *header* names the source column
    (``Split``, or ``Source``). Sources with none are left out, and the section too where no source has any.
    """
    rows: list[dict[str, Cell]] = []
    for source, indices in images.items():
        if indices:
            items, shown = group_cells([ItemRef(source=source, index=index) for index in indices])
            rows.append({"source": source, "count": len(indices), "items": items, "image": shown})
    if not rows:
        return []
    columns = [
        Column(key="source", header=header),
        Column(key="count", header="Count"),
        Column(key="items", header="Items", align="left"),
        Column(key="image", kind="image"),
    ]
    return [Section(title="Images with no labels", blocks=[Table(columns=columns, rows=rows)])]
