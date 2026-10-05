"""The leakage check: items in duplicate groups spanning two splits, and group values two splits share (spec §10.2)."""

__all__ = ["LeakageCheck", "LeakageConfig"]

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import polars as pl
from dataeval.quality import DuplicatesOutput
from pydantic import Field

from dataeval_flow._blocks import Block, Cell, Column, ItemRef, Paragraph, Section, Table
from dataeval_flow._input_spec import SourceCount
from dataeval_flow._tables import group_cells, table_limits
from dataeval_flow.evaluators.quality import FactorLeakageOutput
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.checks._limits import Severity, exceeds
from dataeval_flow.workflows._base import Finding

_Group = tuple[str, list[tuple[int, int]]]  # a group's kind, and its members as (source number, item index)


class LeakageConfig(CheckConfig):
    """A `leakage` step's inputs, and how many items and group values may sit in two splits."""

    duplicates: str | list[str] = Field(
        description=(
            "The `duplicates` Outputs over two sources, each a list or one: train with each evaluation split, and "
            "evaluation pairs."
        )
    )
    factors: str | list[str] | None = Field(
        default=None,
        description="The `factor-leakage` Outputs over the same pairs; unset judges duplicates alone.",
    )
    exact: int | None = Field(
        default=0,
        ge=0,
        description=(
            "Most items, counted over every exact-duplicate group with members in two splits, that may leak before the "
            "finding warns; `null` judges nothing."
        ),
    )
    near: int | None = Field(
        default=0,
        ge=0,
        description="The same for near-duplicate groups; `null` judges nothing.",
    )
    groups: int | None = Field(
        default=0,
        ge=0,
        description=(
            "Most group values (a factor's value held by both splits of a pair) that may be shared before the finding "
            "warns; `null` judges nothing."
        ),
    )


def _nodes(value: Any) -> list[Any]:
    """The nodes a port holds: none, one, a keyed list's present elements, or several of any of these."""
    if value is None:
        return []
    if isinstance(value, list):
        return [node for item in value for node in _nodes(item)]
    present = getattr(value, "present", None)
    return list(present.values()) if present is not None else [value]


def _gaps(value: Any, port: str) -> list[str]:
    """What a port's keyed lists hold no node for, as "`port[key]` was not compared: why", so a missing element does not
    read as a pair that shared nothing."""
    if isinstance(value, list):
        return [gap for item in value for gap in _gaps(item, port)]
    present, elements = getattr(value, "present", None), getattr(value, "elements", None)
    if present is None or elements is None:
        return []
    return [
        f"`{port}[{key}]` was not compared: {getattr(element, 'reason', None) or 'missing'}."
        for key, element in elements.items()
        if key not in present
    ]


def _pair(node: Any) -> tuple[str, str]:
    """The addresses of the two Datasets a node was computed on."""
    on = node.computed_on
    return on[0].address, on[-1].address


def _spanning(node: Any) -> list[_Group]:
    """The item-level duplicate groups with members in two sources."""
    frame = node.value.data().filter(pl.col("level") == "item")
    return [
        (row["dup_type"], list(zip(row["dataset_indices"], row["item_indices"], strict=True)))
        for row in frame.iter_rows(named=True)
        if len(set(row["dataset_indices"] or ())) >= 2
    ]


def _group_blocks(a: str, b: str, groups: Sequence[_Group]) -> list[Block]:
    """One pair's spanning groups, largest first: each split's members beside the other's."""
    ranked = sorted(enumerate(groups), key=lambda entry: -len(entry[1][1]))
    limits = table_limits()
    rows: list[dict[str, Cell]] = []
    for number, (kind, members) in ranked[: limits.rows]:
        row: dict[str, Cell] = {"group": number, "kind": kind}
        for side, number_of, name in (("a", 0, a), ("b", 1, b)):
            refs = [ItemRef(source=name, index=index) for source, index in members if source == number_of]
            row[side], row[f"{side}_image"] = group_cells(refs)
        rows.append(row)
    columns = [
        Column(key="group", header="Group"),
        Column(key="kind", header="Kind", align="left"),
        Column(key="a", header=a, align="left"),
        Column(key="a_image", kind="image"),
        Column(key="b", header=b, align="left"),
        Column(key="b_image", kind="image"),
    ]
    blocks: list[Block] = [Table(columns=columns, rows=rows, preview=limits.preview)]
    if limits.rows is not None and len(ranked) > limits.rows:
        text = f"{len(ranked):,} groups; the {limits.rows:,} largest are listed, and every one is in `output.data`."
        blocks.append(Paragraph(text=text))
    return [Section(title=f"{a} vs {b}", brief=f"{len(groups)} groups", blocks=blocks)]


class LeakageCheck(Check[LeakageConfig]):
    """``leakage``: items or groups shared across splits, over `duplicates` Outputs and `factor-leakage` Outputs.

    Not assessed when no `duplicates` list holds an element: no pair of splits, so nothing to judge. `factors` may be
    empty, so group values that never ran leave the duplicates judged.
    """

    name: ClassVar[str] = "leakage"
    description: ClassVar[str] = "Warns when items or group values sit in two splits at once."
    title: ClassVar[str] = "Leakage"
    inputs: ClassVar[tuple[Port, ...]] = (
        Port("duplicates", DataType.OUTPUT, classes=(DuplicatesOutput,), is_list=True, count=SourceCount.ONE_OR_MORE),
        Port("factors", DataType.OUTPUT, classes=(FactorLeakageOutput,), is_list=True, may_be_empty=True),
    )

    def run(self, config: LeakageConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """Items in groups spanning two splits, and group values two splits share.

        Raises
        ------
        ValueError
            When a `duplicates` Output was not computed on exactly two sources.
        """
        duplicates, factors = _nodes(inputs.get("duplicates")), _nodes(inputs.get("factors"))
        counts: Counter[str] = Counter()
        blocks: list[Block] = []
        for node in duplicates:
            if len(node.computed_on) != 2:
                raise ValueError(
                    f"`{node.address}` was computed on {len(node.computed_on)} sources: leakage compares exactly two."
                )
            found = _spanning(node)
            for kind, members in found:
                counts[kind] += len(members)
            if found:
                blocks.extend(_group_blocks(*_pair(node), found))
        shared: list[dict[str, Cell]] = []
        for node in factors:
            a, b = _pair(node)
            for factor, values in node.value.data()["factors"].items():
                shared.extend(
                    {"pair": f"{a} vs {b}", "factor": factor, "value": value, "a": n[0], "b": n[1]}
                    for value, n in values.items()
                    if n[0] and n[1]
                )
        if shared:
            columns = [
                Column(key="pair", header="Pair", align="left"),
                Column(key="factor", header="Factor", align="left"),
                Column(key="value", header="Value", align="left"),
                Column(key="a", header="First"),
                Column(key="b", header="Second"),
            ]
            limits = table_limits()
            blocks.append(Table(columns=columns, rows=shared[: limits.rows], preview=limits.preview))
            if limits.rows is not None and len(shared) > limits.rows:
                text = f"{len(shared):,} shared values; the first {limits.rows:,} are listed."
                blocks.append(Paragraph(text=text))
        judged = [limit for limit in (config.exact, config.near, config.groups) if limit is not None]
        leaked = (
            exceeds(counts["exact"], config.exact)
            or exceeds(counts["near"], config.near)
            or exceeds(len(shared), config.groups)
        )
        severity: Severity = ("warning" if leaked else "ok") if judged else "info"
        parts = [f"{counts[kind]} {kind}" for kind in ("exact", "near") if counts[kind]]
        brief = " + ".join(parts) + " cross-split duplicates" if parts else "No cross-split duplicates"
        if shared:
            brief += f", {len(shared)} shared group value" + ("s" if len(shared) != 1 else "")
        description = (
            f"{brief} (data leakage)." if parts or shared else "No cross-split duplicates or shared group values."
        )
        gaps = _gaps(inputs.get("duplicates"), "duplicates") + _gaps(inputs.get("factors"), "factors")
        description = " ".join([description, *gaps])
        return [Finding(severity=severity, title=self.title, brief=brief, description=description, blocks=blocks)]
