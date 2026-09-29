"""The scope evaluators' report tables."""

__all__ = ["coverage_section", "uncovered_blocks"]

from collections.abc import Mapping, Sequence
from typing import Any

from dataeval_flow._blocks import Block, Cell, Column, ItemRef, Paragraph, Section, Table
from dataeval_flow._tables import table_limits


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


def coverage_section(output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block]:
    """A Coverage Output's report: the items in sparse regions, pictured, then the per-class table and the rest."""
    from dataeval_flow.evaluators._report import output_blocks

    extras = dict(output.get("extras") or {})
    uncovered = extras.pop("uncovered_indices", None) or []
    refs = [(ItemRef(source=sources[0], index=int(index)), None, None) for index in uncovered]
    return [*uncovered_blocks(refs, "images"), *output_blocks({**output, "extras": extras}, detailed=detailed)]
