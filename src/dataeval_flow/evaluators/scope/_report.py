"""The scope evaluators' report tables."""

__all__ = [
    "coverage_section",
    "label_alignment_section",
    "prioritize_section",
    "representation_section",
    "uncovered_blocks",
]

from collections.abc import Mapping, Sequence
from typing import Any

from dataeval_flow._blocks import Block, Cell, Column, Fields, ItemRef, Paragraph, Section, Table
from dataeval_flow._tables import table_limits


def uncovered_blocks(
    uncovered: Sequence[tuple[ItemRef, str | None, float | None]], noun: str, *, listed_in: str = "output.raw"
) -> list[Block]:
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
                f"`{listed_in}`."
            )
        )
    return [Section(title=f"Uncovered {noun}", blocks=blocks)]


def coverage_section(output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block]:
    """A Coverage Output's report: the items in sparse regions, pictured, then the per-class table and the rest."""
    from dataeval_flow.evaluators._report import output_blocks

    extras = dict(output.get("extras") or {})
    uncovered = extras.pop("uncovered_indices", None) or []
    refs = [(ItemRef(source=sources[0], index=int(index)), None, None) for index in uncovered]
    return [
        *uncovered_blocks(refs, "images", listed_in="output.extras.uncovered_indices"),
        *output_blocks({**output, "extras": extras}, detailed=detailed),
    ]


def representation_section(output: Mapping[str, Any], *, detailed: bool) -> list[Block]:
    """A Representation Output's report: leaf coverage, the deficit and how many concepts fall short. The worklist
    itself is shown by the check that judges it (coverage spec §3.3), and by a detailed report, in full."""
    from dataeval_flow.evaluators._report import output_blocks

    extras = output.get("extras") or {}
    worklist = output.get("rows") or []  # the worklist serializes as a table
    coverage = extras.get("leaf_coverage")
    fields = Fields(
        items=[
            ("Leaf coverage", None if coverage is None else f"{round(float(coverage) * 100, 1)}%"),
            ("Total deficit", extras.get("total_deficit")),
            ("Concepts short", len(worklist)),
        ]
    )
    return [fields, *(output_blocks(dict(output), detailed=True) if detailed else [])]


def label_alignment_section(output: Mapping[str, Any], *, detailed: bool) -> list[Block]:
    """A Label Alignment Output's report: its mergeability and its counts. The correspondences are shown by the check
    that judges it (coverage spec §3.3), and by a detailed report, in full."""
    from dataeval_flow.evaluators._report import output_blocks

    data = output.get("data") or {}
    fields = Fields(
        items=[
            ("Mergeability", data.get("mergeability")),
            ("Correspondences", len(data.get("correspondences") or [])),
            ("Dropped", len(data.get("unaligned_source") or [])),
            ("Not covered", len(data.get("unaligned_target") or [])),
        ]
    )
    return [fields, *(output_blocks(dict(output), detailed=True) if detailed else [])]


# A ranking's first and last this many items are listed.
_ENDS = 25


def prioritize_section(output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block]:  # noqa: ARG001
    """The ranking's 25 highest-priority and 25 lowest-priority items: each one's rank, thumbnail, item and score.

    Rank is the position in the ranking, which a stratified or class-balanced policy doesn't keep in score order.
    Score shows where the method gives one. A ranking of 50 or fewer is split between the two.
    """
    indices = [int(index) for index in output.get("data") or ()]
    scores = (output.get("extras") or {}).get("scores")
    columns = [
        Column(key="rank", header="Rank"),
        Column(key="image", kind="image"),
        Column(key="item", header="Item"),
        *([Column(key="score", header="Score", format="{:.4g}")] if scores is not None else []),
    ]
    ends = [
        ("Highest priority", range(min(_ENDS, len(indices)))),
        ("Lowest priority", range(max(_ENDS, len(indices) - _ENDS), len(indices))),
    ]
    blocks: list[Block] = []
    for title, positions in ends:
        rows: list[dict[str, Cell]] = [
            {
                "rank": position + 1,
                "image": ItemRef(source=sources[0], index=indices[position]),
                "item": indices[position],
                "score": None if scores is None else scores[position],
            }
            for position in positions
        ]
        if rows:
            table = Table(columns=columns, rows=rows, preview=table_limits().preview)
            blocks.append(Section(title=title, blocks=[table]))
    return blocks
