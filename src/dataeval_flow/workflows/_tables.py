"""Table shapes more than one workflow reports its findings in."""

__all__ = ["PREVIEW", "ROW_CAP", "group_cells", "groups_table", "ranked_table", "unlabelled_blocks"]

from collections.abc import Mapping, Sequence
from typing import Any

from dataeval_flow._blocks import Block, Cell, Column, ItemRef, Paragraph, Section, Table
from dataeval_flow._blocks._items import item_name

# A table of items lists at most this many rows, with a paragraph naming the rest, and a renderer with
# little room shows the first few before a line counting the rest.
ROW_CAP = 500
PREVIEW = 10

# A group's cell shows at most this many of its items.
GROUP_SHOWN = 8


def ranked_table(values: Mapping[Any, float], *, headers: tuple[str, str]) -> Table:
    """*values* as name, value and bar columns, largest first: a count per class, or an MI per factor."""
    ranked = sorted(values.items(), key=lambda item: -item[1])
    rows: list[dict[str, Cell]] = [{"name": str(name), "value": value} for name, value in ranked]
    columns = [
        Column(key="name", header=headers[0]),
        Column(key="value", header=headers[1]),
        Column(key="value", kind="bar"),
    ]
    return Table(columns=columns, rows=rows)


def group_cells(refs: Sequence[ItemRef]) -> tuple[str, list[ItemRef]]:
    """Up to eight of a group's items: named for text (``0, 5, … 12 more``), and as references for thumbnails.

    Text leaves image columns out, so the names are what tell a text reader which items a group holds.
    """
    shown = list(refs[:GROUP_SHOWN])
    names = ", ".join(item_name(ref) for ref in shown)
    more = len(refs) - len(shown)
    return (f"{names}, … {more:,} more" if more else names), shown


def groups_table(groups: Sequence[tuple[str, int, Sequence[ItemRef]]], noun: str) -> list[Block]:
    """Duplicate groups, largest first: each one's kind, size, and up to eight of its items, named and pictured.

    *groups* are each group's kind (``exact`` or ``near``), its number among its kind in ``output.raw``,
    where every one of its items is, and its items. At most 500 are listed, with a paragraph naming the rest.
    """
    if not groups:
        return []
    # Stable, so groups of one size keep exact before near, and each kind its own order.
    ranked = sorted(groups, key=lambda group: -len(group[2]))
    rows: list[dict[str, Cell]] = []
    for kind, number, refs in ranked[:ROW_CAP]:
        items, shown = group_cells(refs)
        rows.append({"group": number, "kind": kind, "count": len(refs), "items": items, "image": shown})
    columns = [
        Column(key="group", header="Group"),
        Column(key="kind", header="Kind", align="left"),
        Column(key="count", header="Count"),
        Column(key="items", header="Items", align="left"),
        Column(key="image", kind="image"),
    ]
    blocks: list[Block] = [Table(columns=columns, rows=rows, preview=PREVIEW)]
    if len(ranked) > ROW_CAP:
        blocks.append(
            Paragraph(
                text=f"{len(ranked):,} groups of {noun}; the {ROW_CAP:,} largest are listed, and every one is in "
                "`output.raw`."
            )
        )
    return blocks


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
