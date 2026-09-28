"""Table shapes more than one workflow reports its findings in."""

__all__ = [
    "GROUP_SHOWN",
    "TableLimits",
    "group_cells",
    "groups_table",
    "limited_tables",
    "ranked_table",
    "table_limits",
    "uncovered_blocks",
    "unlabelled_blocks",
]

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from dataeval_flow._blocks import Block, Cell, Column, ItemRef, Paragraph, Section, Table
from dataeval_flow._blocks._items import item_name
from dataeval_flow._blocks._table import DEFAULT_PREVIEW, DEFAULT_ROWS


@dataclass(frozen=True)
class TableLimits:
    """How many rows a table of items lists, with a paragraph naming the rest, and how many of them a renderer
    with little room shows first, before a line counting the rest; ``None`` for every row.
    """

    rows: int | None = DEFAULT_ROWS
    preview: int | None = DEFAULT_PREVIEW


# The limits of the run in progress, which the orchestrator sets from the pipeline's `result:` block. A
# workflow run on its own, outside a pipeline, keeps the defaults.
_LIMITS: ContextVar[TableLimits] = ContextVar("table_limits")
_DEFAULT_LIMITS = TableLimits()


def table_limits() -> TableLimits:
    """The limits on the tables of items the run in progress builds."""
    return _LIMITS.get(_DEFAULT_LIMITS)


@contextmanager
def limited_tables(limits: TableLimits) -> Iterator[None]:
    """Build every table of items within *limits* until the block ends."""
    token = _LIMITS.set(limits)
    try:
        yield
    finally:
        _LIMITS.reset(token)


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


def group_cells(refs: Sequence[ItemRef], total: int | None = None) -> tuple[str, list[ItemRef]]:
    """Up to eight of a group's items: named for text (``0, 5, … 12 more``), and as references for thumbnails.

    Text leaves image columns out, so the names are what tell a text reader which items a group holds.
    *total* is the group's size where *refs* are only its first few.
    """
    shown = list(refs[:GROUP_SHOWN])
    names = ", ".join(item_name(ref) for ref in shown)
    more = (len(refs) if total is None else total) - len(shown)
    return (f"{names}, … {more:,} more" if more else names), shown


def groups_table(groups: Sequence[tuple[str, int, Sequence[ItemRef]]], noun: str) -> list[Block]:
    """Duplicate groups, largest first: each one's kind, size, and up to eight of its items, named and pictured.

    *groups* are each group's kind (``exact`` or ``near``), its number among its kind in ``output.raw``,
    where every one of its items is, and its items. At most ``result: max_rows``, 500 by default, are listed,
    with a paragraph naming the rest.
    """
    if not groups:
        return []
    # Stable, so groups of one size keep exact before near, and each kind its own order.
    ranked = sorted(groups, key=lambda group: -len(group[2]))
    limits = table_limits()
    rows: list[dict[str, Cell]] = []
    for kind, number, refs in ranked[: limits.rows]:
        items, shown = group_cells(refs)
        rows.append({"group": number, "kind": kind, "count": len(refs), "items": items, "image": shown})
    columns = [
        Column(key="group", header="Group"),
        Column(key="kind", header="Kind", align="left"),
        Column(key="count", header="Count"),
        Column(key="items", header="Items", align="left"),
        Column(key="image", kind="image"),
    ]
    blocks: list[Block] = [Table(columns=columns, rows=rows, preview=limits.preview)]
    if limits.rows is not None and len(ranked) > limits.rows:
        blocks.append(
            Paragraph(
                text=f"{len(ranked):,} groups of {noun}; the {limits.rows:,} largest are listed, and every one is in "
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


def uncovered_blocks(uncovered: Sequence[tuple[ItemRef, str | None, float | None]], noun: str) -> list[Block]:
    """Items in sparse regions of the embedding space, under their own heading, farthest first.

    *uncovered* are each item's reference, its class where known, and its distance to its k-th nearest
    neighbour where known; *noun* names them (``images``, ``detection crops``). Each row shows the item's
    thumbnail and name, its box where it is one, its class and its distance, and a column none of them
    has is left out. At most ``result: max_rows``, 500 by default, are listed, with a paragraph counting the rest.
    """
    if not uncovered:
        return []
    ranked = sorted(uncovered, key=lambda row: (-(row[2] or 0.0), row[0].index, row[0].target or 0))
    limits = table_limits()
    rows: list[dict[str, Cell]] = [
        {"image": ref, "item": ref.index, "box": ref.target, "class": name, "distance": distance}
        for ref, name, distance in ranked[: limits.rows]
    ]
    columns = [
        Column(key="image", kind="image"),
        Column(key="item", header="Item"),
        *([Column(key="box", header="Box")] if any(ref.target is not None for ref, _, _ in ranked) else []),
        *([Column(key="class", header="Class", align="left")] if any(name for _, name, _ in ranked) else []),
        *(
            [Column(key="distance", header="Distance", format="{:.4g}")]
            if any(d is not None for *_, d in ranked)
            else []
        ),
    ]
    blocks: list[Block] = [
        Paragraph(
            text="Farthest first, by each one's distance to its k-th nearest neighbour (k is `num_observations`)."
        ),
        Table(columns=columns, rows=rows, preview=limits.preview),
    ]
    if limits.rows is not None and len(ranked) > limits.rows:
        blocks.append(
            Paragraph(
                text=f"{len(ranked):,} {noun} uncovered; the {limits.rows:,} farthest are listed, and every one is in "
                "`output.raw`."
            )
        )
    return [Section(title=f"Uncovered {noun}", blocks=blocks)]
