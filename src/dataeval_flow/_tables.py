"""Row limits for tables of items, and the cell a group of items shows: shared by every report that lists items."""

__all__ = ["GROUP_SHOWN", "TableLimits", "group_cells", "limited_tables", "table_limits"]

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from dataeval_flow._blocks import ItemRef
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


def group_cells(refs: Sequence[ItemRef], total: int | None = None) -> tuple[str, list[ItemRef]]:
    """Up to eight of a group's items: named for text (``0, 5, … 12 more``), and as references for thumbnails.

    Text leaves image columns out, so the names are what tell a text reader which items a group holds.
    *total* is the group's size where *refs* are only its first few.
    """
    shown = list(refs[:GROUP_SHOWN])
    names = ", ".join(item_name(ref) for ref in shown)
    more = (len(refs) if total is None else total) - len(shown)
    return (f"{names}, … {more:,} more" if more else names), shown
