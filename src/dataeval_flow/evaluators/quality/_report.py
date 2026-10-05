"""The quality evaluators' report tables: flagged items with their flags, each metric's limits, and duplicate groups.

data-cleaning reports with them too.
"""

__all__ = [
    "content_digest_section",
    "OutlierIssueRecord",
    "OutlierIssuesDict",
    "duplicate_section",
    "flag_of",
    "flagged_table",
    "groups_table",
    "label_health_section",
    "limits_sentence",
    "limits_table",
    "outlier_section",
    "warn_if_unrecorded",
]

import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, NotRequired

from typing_extensions import TypedDict

from dataeval_flow._blocks import Block, Cell, Column, Fields, Flag, ItemRef, Paragraph, Scalar, Section, Table
from dataeval_flow._blocks._table import fair_shares
from dataeval_flow._tables import group_cells, table_limits
from dataeval_flow.workflows._base import render_label_source

_logger = logging.getLogger(__name__)

# DataEval's threshold methods: the multiplier each uses by default, and how its limits read.
_METHODS: dict[str, tuple[float, str]] = {
    "zscore": (3.0, "Limits: the mean ± {t} standard deviations (z-score)."),
    "modzscore": (3.5, "Limits: a modified z-score of {t}, measured from the median by the MAD."),
    "iqr": (1.5, "Limits: {t} × the IQR beyond the quartiles."),
    "adaptive": (3.5, "Limits: {t} × the MAD on each side of the median, widened where a tail is heavy (adaptive)."),
}


class OutlierIssueRecord(TypedDict):
    """One flagged value from DataEval's ``Outliers``: which item and metric, the value, and its population.

    ``target_index`` is present for target-level outliers (object detection datasets) and absent or
    ``None`` for image-level ones. The context columns (``direction``, ``bound``, ``percentile``,
    ``population_mean``, ``population_std``) describe the comparison that flagged the value; each is
    absent or ``None`` where it wasn't recorded, as in a frame merged with one that lacks them.
    """

    item_index: int
    metric_name: str
    metric_value: float
    target_index: NotRequired[int | None]
    direction: NotRequired[Literal["upper", "lower"] | None]
    bound: NotRequired[float | None]
    percentile: NotRequired[float | None]
    population_mean: NotRequired[float | None]
    population_std: NotRequired[float | None]


class OutlierIssuesDict(TypedDict):
    """Serialized outlier issues (image or target level)."""

    issues: list[OutlierIssueRecord]
    count: int


def _number(issue: Mapping[str, Any], name: str) -> float:
    value = issue.get(name)
    return math.nan if value is None else float(value)


def flag_of(issue: Mapping[str, Any]) -> Flag:
    """The issue as a flag: its metric and value against the population it was judged in.

    An issue written by a DataEval that predates the context columns keeps its value, and its limit,
    percentile and population read as unknown (NaN), so its tag shows the value alone.
    """
    return Flag(
        name=str(issue["metric_name"]),
        value=float(issue["metric_value"]),
        direction="lower" if issue.get("direction") == "lower" else "upper",
        bound=_number(issue, "bound"),
        percentile=_number(issue, "percentile"),
        mean=_number(issue, "population_mean"),
        std=_number(issue, "population_std"),
    )


def warn_if_unrecorded(issues: Iterable[Mapping[str, Any]]) -> None:
    """Warn, once for all of *issues*, where any came without the limit it crossed: an older DataEval's record."""
    unrecorded = sum(not math.isfinite(_number(issue, "bound")) for issue in issues)
    if unrecorded:
        _logger.warning(
            "%d outlier flag(s) came without the limits they crossed, so the report shows their values alone. "
            "Upgrade DataEval to one that records each flag's limit, percentile and population.",
            unrecorded,
        )


def flagged_table(
    issues: Sequence[Mapping[str, Any]],
    *,
    key: Callable[[Mapping[str, Any]], tuple[Cell, ...]],
    key_columns: Sequence[Column],
    classes: Mapping[tuple[Cell, ...], str] | None,
    noun: str,
    groups: Sequence[str] | None = None,
    ref: Callable[[tuple[Cell, ...]], ItemRef] | None = None,
    listed_in: str = "output.raw",
) -> list[Block]:
    """A row per flagged subject, with every flag it raised, in the order of the subjects' keys.

    *key* names the subject an issue is about, such as its item, or its item and box; *key_columns*
    show those values, in the key's order, and read row keys ``key_columns[i].key``. *classes*, where
    given, names each subject's class, and *ref* its item, whose thumbnail leads the row. The table
    lists at most ``result: max_rows``, 500 by default, and a paragraph after it says how many it left out.

    *groups*, where given, are the values the key's first element takes, such as splits, in the order
    the rows run. Each group lists an equal share of the 500, a small group's spare going to the rest,
    so no group crowds out another, and the paragraph says what each left out.
    """
    flags: dict[tuple[Cell, ...], list[Flag]] = {}
    for issue in issues:
        flags.setdefault(key(issue), []).append(flag_of(issue))
    if not flags:
        return []

    # No subject ranks above another: how far past its limit a value lies doesn't say it's worse.
    subjects = sorted(flags)
    parts = [subjects] if groups is None else [[s for s in subjects if s[0] == group] for group in groups]
    limits = table_limits()
    cap = len(subjects) if limits.rows is None else limits.rows
    shares = fair_shares([len(part) for part in parts], cap)
    rows: list[dict[str, Cell]] = []
    for subject in (subject for part, share in zip(parts, shares, strict=True) for subject in part[:share]):
        row: dict[str, Cell] = {column.key: value for column, value in zip(key_columns, subject, strict=True)}
        if ref is not None:
            row["image"] = ref(subject)
        if classes is not None:
            row["class"] = classes.get(subject)
        row["flags"] = len(flags[subject])
        row["by"] = list(flags[subject])
        rows.append(row)
    columns = [
        *([Column(key="image", kind="image")] if ref is not None else []),
        *key_columns,
        *([Column(key="class", header="Class", align="left")] if classes is not None else []),
        Column(key="flags", header="Flags"),
        Column(key="by", header="Flagged by", kind="flags"),
    ]
    blocks: list[Block] = [Table(columns=columns, rows=rows, preview=limits.preview)]
    if len(rows) < len(subjects):
        if groups is None:
            text = f"{len(subjects):,} {noun} flagged; the first {len(rows):,} are listed"
        else:
            left = ", ".join(
                f"{len(part) - share:,} of {len(part):,} in {group}"
                for group, part, share in zip(groups, parts, shares, strict=True)
                if share < len(part)
            )
            text = f"{len(subjects):,} {noun} flagged; {len(rows):,} are listed, leaving out {left}"
        blocks.append(Paragraph(text=f"{text}, and every one is in `{listed_in}`."))
    return blocks


def _one(values: set[float]) -> Cell:
    """A figure every flag shares; "varies" where the flags differ, and nothing where none has one."""
    known = {round(value, 9) for value in values if math.isfinite(value)}
    if not known:
        return None
    return next(iter(known)) if len(known) == 1 else "varies"


def limits_table(issues: Sequence[Mapping[str, Any]], *, key: Callable[[Mapping[str, Any]], tuple[Cell, ...]]) -> Table:
    """A row per metric: how many subjects it flagged, the limits they crossed, and the population's mean and std.

    DataEval records the limit each flag crossed, not both, so a limit shows where some flag crossed
    it. A figure that differs between flags, as ``cluster_distance``'s does from one cluster to the
    next, reads "varies"; the flags' own cards carry each one.
    """
    subjects: dict[str, set[tuple[Cell, ...]]] = {}
    figures: dict[str, dict[str, set[float]]] = {}
    for issue in issues:
        metric = str(issue["metric_name"])
        subjects.setdefault(metric, set()).add(key(issue))
        seen = figures.setdefault(metric, {"lower": set(), "upper": set(), "mean": set(), "std": set()})
        seen["lower" if issue.get("direction") == "lower" else "upper"].add(_number(issue, "bound"))
        seen["mean"].add(_number(issue, "population_mean"))
        seen["std"].add(_number(issue, "population_std"))
    rows: list[dict[str, Cell]] = [
        {
            "metric": metric,
            "count": len(subjects[metric]),
            **{name: _one(figures[metric][name]) for name in ("lower", "upper", "mean", "std")},
        }
        for metric in sorted(subjects, key=lambda metric: (-len(subjects[metric]), metric))
    ]
    number = "{:.4g}"
    return Table(
        columns=[
            Column(key="metric", header="Metric"),
            Column(key="count", header="Count"),
            Column(key="lower", header="Lower", format=number),
            Column(key="upper", header="Upper", format=number),
            Column(key="mean", header="Mean", format=number),
            Column(key="std", header="Std", format=number),
        ],
        rows=rows,
    )


def limits_sentence(method: str | None, threshold: float | None) -> str | None:
    """How the method's limits read, with its multiplier or DataEval's default; ``None`` for an unknown method."""
    if method not in _METHODS:
        return None
    default, sentence = _METHODS[method]
    return sentence.format(t=f"{default if threshold is None else threshold:g}")


def groups_table(
    groups: Sequence[tuple[str, int, Sequence[ItemRef]]], noun: str, *, listed_in: str = "output.raw"
) -> list[Block]:
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
                f"`{listed_in}`."
            )
        )
    return blocks


def outlier_section(output: Mapping[str, Any], sources: Sequence[str]) -> list[Block]:
    """An Outliers Output's report: each flagged image with every flag it raised, then each metric's limits; flagged
    boxes likewise, under their own heading. Each item is named by the source it belongs to, for its thumbnail."""
    issues = list(output.get("rows") or [])
    if not issues:
        return [Paragraph(text="No image or box was flagged.")]
    images = [issue for issue in issues if issue.get("target_index") is None]
    boxes = [issue for issue in issues if issue.get("target_index") is not None]
    blocks = _flagged(images, sources, boxes=False)
    if boxes:
        blocks.append(Section(title="Flagged boxes", blocks=_flagged(boxes, sources, boxes=True)))
    return blocks


def _flagged(issues: Sequence[Mapping[str, Any]], sources: Sequence[str], *, boxes: bool) -> list[Block]:
    """The flagged table, then the limits table, for `issues`; keyed by source first where the run read several."""
    if not issues:
        return []
    several = len(sources) > 1

    def key(issue: Mapping[str, Any]) -> tuple[Cell, ...]:
        where: tuple[Cell, ...] = (sources[int(issue.get("dataset_index") or 0)],) if several else ()
        what: tuple[Cell, ...] = (issue["item_index"], issue["target_index"]) if boxes else (issue["item_index"],)
        return (*where, *what)

    def ref(subject: tuple[Cell, ...]) -> ItemRef:
        source, rest = (str(subject[0]), subject[1:]) if several else (sources[0], subject)
        return ItemRef.model_validate({"source": source, "index": rest[0], "target": rest[1] if boxes else None})

    columns = [
        *([Column(key="source", header="Source", align="left")] if several else []),
        Column(key="item", header="Item"),
        *([Column(key="box", header="Box")] if boxes else []),
    ]
    table = flagged_table(
        issues,
        key=key,
        key_columns=columns,
        classes=None,
        noun="boxes" if boxes else "images",
        groups=list(sources) if several else None,
        ref=ref,
        listed_in="output.rows",
    )
    return [*table, limits_table(issues, key=key)]


def duplicate_section(output: Mapping[str, Any], sources: Sequence[str], *, detailed: bool) -> list[Block]:
    """A Duplicates Output's report: each group of images, largest first, then each group at every other level, such
    as boxes, under its own heading; then any extras. Each item is named by the source it belongs to."""
    from dataeval_flow.evaluators._report import extras_blocks

    rows = list(output.get("rows") or [])
    present = list(dict.fromkeys(str(row["level"]) for row in rows))
    levels = [level for level in ("item",) if level in present] + [level for level in present if level != "item"]
    blocks: list[Block] = []
    for level in levels:
        groups = [
            (str(row["dup_type"]), int(row["group_id"]), _members(row, sources))
            for row in rows
            if row["level"] == level
        ]
        if level == "item":
            blocks.extend(groups_table(groups, "images", listed_in="output.rows"))
        else:
            noun = "boxes" if level == "target" else f"{level}s"
            blocks.append(
                Section(
                    title=f"Duplicate {noun}",
                    brief=f"{len(groups)} groups",
                    blocks=groups_table(groups, noun, listed_in="output.rows"),
                )
            )
    if not rows:
        blocks.append(Paragraph(text="No duplicates found."))
    return [*blocks, *extras_blocks(output, detailed=detailed)]


def _members(row: Mapping[str, Any], sources: Sequence[str]) -> list[ItemRef]:
    """A group's members as item references: each item, or box, in the source it belongs to."""
    items = [int(index) for index in row["item_indices"]]
    targets = row.get("target_indices") or [None] * len(items)
    datasets = row.get("dataset_indices") or [0] * len(items)
    several = len(sources) > 1
    return [
        ItemRef.model_validate(
            {"source": sources[int(dataset)] if several else sources[0], "index": item, "target": target}
        )
        for item, target, dataset in zip(items, targets, datasets, strict=True)
    ]


def content_digest_section(output: Mapping[str, Any]) -> list[Block]:
    """A ``content-digest`` Output's report: the item count and both digests, in full."""
    data = output.get("data") or {}
    fields: list[tuple[str, Scalar]] = [
        ("Items", data.get("items")),
        ("Content digest", data.get("content")),
        ("Metadata digest", data.get("metadata")),
    ]
    return [Fields(items=fields)]


def label_health_section(output: Mapping[str, Any]) -> list[Block]:
    """A ``label-health`` Output's report: its counts as fields, then each class's labels and images."""
    data = output.get("data") or {}
    fields: list[tuple[str, Scalar]] = [
        ("Items", data.get("item_count")),
        ("Classes", data.get("class_count")),
        ("Labels", data.get("label_count")),
        ("Empty images", data.get("empty_image_count")),
    ]
    if data.get("label_source"):
        fields.append(("Label source", render_label_source(data["label_source"])))
    fields_block = Fields(items=fields)
    blocks: list[Block] = [fields_block]
    labels, images = data.get("label_counts_per_class") or {}, data.get("image_counts_per_class") or {}
    if labels:
        columns = [
            Column(key="class", header="Class"),
            Column(key="labels", header="Labels"),
            Column(key="images", header="Images"),
        ]
        rows: list[dict[str, Cell]] = [
            {"class": str(name), "labels": count, "images": images.get(name, 0)} for name, count in labels.items()
        ]
        blocks.append(Table(columns=columns, rows=rows))
    return blocks
