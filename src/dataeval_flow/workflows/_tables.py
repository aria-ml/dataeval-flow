"""Table shapes more than one workflow reports its findings in."""

__all__ = ["PREVIEW", "ROW_CAP", "ranked_table"]

from collections.abc import Mapping
from typing import Any

from dataeval_flow._blocks import Cell, Column, Table

# A table of items lists at most this many rows, with a paragraph naming the rest, and a renderer with
# little room shows the first few before a line counting the rest.
ROW_CAP = 500
PREVIEW = 10


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
