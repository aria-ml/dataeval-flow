"""The bias evaluators' report tables."""

__all__ = ["balance_section", "metadata_summary_section", "ranked_table"]

from collections.abc import Mapping
from typing import Any

from dataeval_flow._blocks import Block, Cell, Column, Paragraph, Section, Table


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


def balance_section(output: Mapping[str, Any], *, detailed: bool) -> list[Block]:
    """A Balance Output's report: each factor ranked by its mutual information with the class, then the rest."""
    from dataeval_flow.evaluators._report import output_blocks

    data = dict(output.get("data") or {})
    balance = data.get("balance") or {}
    values = {
        str(row["factor_name"]): float(row["mi_value"])
        for row in balance.get("rows") or []
        if row["factor_name"] != "class_label" and isinstance(row.get("mi_value"), int | float)
    }
    if values:
        data.pop("balance")
    ranked: list[Block] = (
        [
            Section(
                title="Balance",
                brief="mutual information with the class",
                blocks=[ranked_table(values, headers=("Factor", "MI"))],
            )
        ]
        if values
        else []
    )
    return [*ranked, *output_blocks({**output, "data": data}, detailed=detailed)]


def metadata_summary_section(output: Mapping[str, Any]) -> list[Block]:
    """A Metadata Summary Output's report: legacy data-coverage's Metadata Distribution table over the kept factors, or
    a sentence when there are none (coverage spec §6.1)."""
    data = output.get("data") or {}
    factors = list(data.get("factors") or [])
    if not factors:
        return [Paragraph(text="No metadata factors available")]
    summary = data.get("summary") or {}
    rows: list[dict[str, Cell]] = []
    for name in factors:
        stats = summary.get(name) or {}
        if "unique_values" in stats:
            unique: Cell = stats["unique_values"]
        elif stats.get("mean") is not None:
            # An all-null column (e.g. a target-level factor read off image-level rows) has a null mean: absent.
            unique = f"μ={round(stats['mean'], 2)}"
        else:
            unique = "-"
        rows.append(
            {
                "factor": name,
                "type": stats.get("type", "unknown"),
                "unique": unique,
                "nulls": stats.get("null_count", 0),
            }
        )
    return [
        Table(
            columns=[
                Column(key="factor", header="Factor"),
                Column(key="type", header="Type"),
                Column(key="unique", header="Unique"),
                Column(key="nulls", header="Nulls"),
            ],
            rows=rows,
        )
    ]
