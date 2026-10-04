"""The `stratification` check: how far each part's class shares stray from the whole's (data-splitting spec §6.1)."""

__all__ = ["StratificationCheck", "StratificationConfig", "StratificationThresholds", "stratification_severity"]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dataeval_flow._blocks import Cell, Column, Fields, Scalar, Table
from dataeval_flow._input_spec import SourceCount
from dataeval_flow.evaluators.quality._result import LabelHealthOutput
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.workflows._base import Finding

Severity = Literal["ok", "info", "warning"]

# Over this many classes, the table keeps the most and least common, as legacy's did.
_MAX_ROWS = 20
_TOP = 10
_BOTTOM = 5


class StratificationThresholds(BaseModel):
    """When a stratification finding is `info` or warns: the largest deviation of a class's share, in percentage
    points."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    info: float | None = Field(
        default=2.0,
        ge=0.0,
        le=100.0,
        description=(
            "The largest deviation, in percentage points, above which the finding is `info`; at or below it, `ok`. "
            "`null` has no `info` band: deviations up to `warning` are `ok`. data-splitting's 2."
        ),
    )
    warning: float | None = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description=(
            "The largest deviation above which the finding warns; `null` never warns. With both `null`, the finding "
            "is `info` and judges nothing. data-splitting's 10."
        ),
    )


def stratification_severity(deviation: float, thresholds: StratificationThresholds) -> Severity:
    """The severity `deviation` earns: `warning` above `warning`, `info` above `info`, else `ok`; with both `null`,
    `info`, which judges nothing. "Above" is `>`, legacy's comparison."""
    if thresholds.info is None and thresholds.warning is None:
        return "info"
    if thresholds.warning is not None and deviation > thresholds.warning:
        return "warning"
    if thresholds.info is not None and deviation > thresholds.info:
        return "info"
    return "ok"


class StratificationConfig(CheckConfig, StratificationThresholds):
    """A `stratification` step's inputs and thresholds."""

    input: str = Field(description="A `label-health` Output over the whole Dataset the parts were split from.")
    parts: str | list[str] = Field(description="The parts' `label-health` Outputs, each judged against `input`.")
    shown: str | list[str] | None = Field(
        default=None,
        description=(
            "`label-health` Outputs shown in the table beside the parts but not judged, such as a rebalanced train."
        ),
    )


@dataclass(frozen=True)
class _Column:
    """One Dataset's label counts, headed by the address of the Dataset they were computed on."""

    name: str
    counts: dict[str, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def percent(self, name: str) -> int:
        return round(self.counts.get(name, 0) / self.total * 100) if self.total else 0

    @classmethod
    def of(cls, node: Any) -> "_Column":
        on = getattr(node, "computed_on", ())
        counts = node.value.data()["label_counts_per_class"]
        return cls(on[0].address if on else node.address, {str(key): int(value) for key, value in counts.items()})


def _nodes(value: Any) -> list[Any]:
    """A port's nodes, however the engine hands them: none, one, or several."""
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


class StratificationCheck(Check[StratificationConfig]):
    """``stratification``: the largest gap between a class's share of a part's labels and its share of the whole's,
    in percentage points, with the counts across the parts as its evidence. Makes no finding where the whole holds
    no labels."""

    name: ClassVar[str] = "stratification"
    description: ClassVar[str] = "Judges how far each part's class shares stray from the whole's."
    title: ClassVar[str] = "Stratification"
    inputs: ClassVar[tuple[Port, ...]] = (
        Port("input", DataType.OUTPUT, classes=(LabelHealthOutput,)),
        Port("parts", DataType.OUTPUT, classes=(LabelHealthOutput,), count=SourceCount.ONE_OR_MORE),
        Port("shown", DataType.OUTPUT, classes=(LabelHealthOutput,)),
    )

    def run(self, config: StratificationConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The largest deviation, the worst class and part, and the table of counts."""
        whole = _Column.of(inputs["input"])
        if not whole.total:
            return []
        parts = [_Column.of(node) for node in _nodes(inputs["parts"])]
        shown = [_Column.of(node) for node in _nodes(inputs.get("shown"))]
        deviation, worst, where = _worst(whole, [part for part in parts if part.total])
        fields: list[tuple[str, Scalar]] = [("Max deviation", f"{deviation}pp")]
        if deviation:
            share = round(whole.counts[worst] / whole.total * 100, 1)
            fields.append(("Worst", f"class '{worst}' in {where}, {share}% of {whole.name}"))
        fields.append(("Classes checked", len(whole.counts)))
        left_out = [part.name for part in parts if not part.total]
        if left_out:
            fields.append(("Left out, no labels", ", ".join(left_out)))
        return [
            Finding(
                severity=stratification_severity(deviation, config),
                title=self.title,
                brief=f"max deviation {deviation}pp" + (f" (class '{worst}' in {where})" if deviation else ""),
                description=(
                    f"Each part's class proportions against `{whole.name}`'s, in percentage points. Counts are labels: "
                    "one per box on detection data."
                ),
                blocks=[_table(whole, parts, shown), Fields(items=fields)],
            )
        ]


def _worst(whole: _Column, parts: Sequence[_Column]) -> tuple[float, str, str]:
    """The largest deviation of a class's share of a part from its share of the whole, rounded to one place as legacy
    rounded it, the class and the part; ``(0.0, "", "")`` where no part strays."""
    found = (0.0, "", "")
    for name, count in whole.counts.items():
        share = count / whole.total * 100
        for part in parts:
            deviation = abs(part.counts.get(name, 0) / part.total * 100 - share)
            if deviation > found[0]:
                found = (deviation, name, part.name)
    return round(found[0], 1), found[1], found[2]


def _table(whole: _Column, parts: Sequence[_Column], shown: Sequence[_Column]) -> Table:
    """Legacy's cross-split table: a row per class, ordered by the whole's count, and a column per part, shown Output
    and the whole. A part's cell is its bare count where every part holds the class at the same whole percent, else
    "count (pct%)"; a shown column's and the whole's always carry the percent."""
    columns = [*parts, *shown, whole]
    keys = [f"c{index}" for index in range(len(columns))]
    names = sorted(whole.counts, key=lambda name: whole.counts[name], reverse=True)
    top, bottom = (names[:_TOP], names[-_BOTTOM:]) if len(names) > _MAX_ROWS else (names, [])
    rows = [_row(name, parts, columns, keys) for name in top]
    if bottom:
        rows.append({"class": f"... {len(names) - _TOP - _BOTTOM} more ...", **dict.fromkeys(keys, "")})
        rows.extend(_row(name, parts, columns, keys) for name in bottom)
    headers = [Column(key=key, header=column.name) for key, column in zip(keys, columns, strict=True)]
    return Table(columns=[Column(key="class", header="Class"), *headers], rows=rows)


def _row(name: str, parts: Sequence[_Column], columns: Sequence[_Column], keys: Sequence[str]) -> dict[str, Cell]:
    bare = len({part.percent(name) for part in parts}) == 1
    row: dict[str, Cell] = {"class": name}
    for key, column in zip(keys, columns, strict=True):
        count = column.counts.get(name, 0)
        judged = any(column is part for part in parts)
        row[key] = count if bare and judged else f"{count} ({column.percent(name)}%)"
    return row
