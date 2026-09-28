"""Outlier evidence as report blocks: a row per flagged item with its flags, and a row per metric with its limits.

Data cleaning and data analysis both find outliers with DataEval's ``Outliers``, whose issues carry,
beside each flagged value, the population it was judged in: which limit it crossed, the limit, its
percentile, and the population's mean and standard deviation. These builders turn those issues into
the tables both workflows' findings show, so the two read the same way.
"""

__all__ = [
    "OutlierIssueRecord",
    "OutlierIssuesDict",
    "flag_of",
    "flagged_table",
    "limits_sentence",
    "limits_table",
    "warn_if_unrecorded",
]

import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, NotRequired

from typing_extensions import TypedDict

from dataeval_flow._blocks import Block, Cell, Column, Flag, ItemRef, Paragraph, Table
from dataeval_flow._blocks._table import fair_shares
from dataeval_flow.workflows._tables import table_limits

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
        blocks.append(Paragraph(text=f"{text}, and every one is in `output.raw`."))
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
