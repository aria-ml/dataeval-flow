"""The bias evaluators' report tables."""

__all__ = ["ranked_table"]

from collections.abc import Mapping
from typing import Any

from dataeval_flow._blocks import Cell, Column, Table


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
