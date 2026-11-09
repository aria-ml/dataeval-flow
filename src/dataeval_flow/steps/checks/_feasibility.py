"""The `performance-feasibility` check: whether the data demonstrates each performance target reachable, drawn as one
row per target, with the bounds behind it."""

__all__ = ["PerformanceFeasibilityCheck", "PerformanceFeasibilityConfig"]

import json
import math
from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from dataeval.performance import FeasibilityOutput
from pydantic import Field

from dataeval_flow._blocks import Cell, Column, Fields, Scalar, Table
from dataeval_flow.evaluators.performance._feasibility_result import PopulatedFeasibilityOutput
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.checks._limits import Severity
from dataeval_flow.workflows._base import Finding

# DataEval's words for each determination, as the banner counts them. "Ruled out" always carries its condition.
_WORDS = {
    "demonstrated_feasible": "demonstrated",
    "infeasible_under_recorded_assumptions": "ruled out under the recorded assumptions",
    "not_ruled_out": "not ruled out",
    "unresolved": "insufficient evidence",
    "evidence_conflict": "evidence conflict",
}
# The mark each answer gets in the table's first column. DataEval's printout marks "not ruled out" with a ``?`` as it
# does "insufficient evidence"; a column of marks alone could not tell the two rows apart, so this one is ``~``.
_SYMBOLS = {
    "demonstrated_feasible": "✓",
    "infeasible_under_recorded_assumptions": "✗",
    "unresolved": "?",
    "not_ruled_out": "~",
    "evidence_conflict": "!",
}
_WHOLE = 1000  # the bar's scale: tenths of a percent, the resolution the bounds are rounded to


class PerformanceFeasibilityConfig(CheckConfig):
    """A `performance-feasibility` step's input. The targets live on the `feasibility` evaluator it judges."""

    input: str = Field(description="A `feasibility` Output.")


class PerformanceFeasibilityCheck(Check[PerformanceFeasibilityConfig]):
    """``performance-feasibility``: whether every performance target is demonstrated reachable, with the bounds that
    say so."""

    name: ClassVar[str] = "performance-feasibility"
    description: ClassVar[str] = (
        "Judges `feasibility`'s output: warns for each performance target that is not demonstrated reachable, and "
        "draws the bounds behind every answer."
    )
    experimental: ClassVar[bool] = True  # judges `feasibility`, which is experimental
    title: ClassVar[str] = "Performance Feasibility"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(FeasibilityOutput,)),)

    def run(
        self,
        config: PerformanceFeasibilityConfig,  # noqa: ARG002
        inputs: Mapping[str, Any],
        context: CheckContext,  # noqa: ARG002
    ) -> list[Finding]:
        """A table of a row per target, in the order the evaluator was given them, then what the answers rest on.

        ``ok`` only where every target is demonstrated feasible; ``warning`` for any other answer, including one that
        rules a target out, since that holds only under the assumptions recorded with it. Each row's bounds are
        DataEval's, a lower bound rounded down and an upper bound up, as its printout rounds them; the unrounded
        bounds, the evidence and its assumptions stay in the evaluator's output.
        """
        output: FeasibilityOutput = inputs["input"].value
        rows = list(output.determinations.iter_rows(named=True))
        demonstrated = sum(row["determination"] == "demonstrated_feasible" for row in rows)
        others = [
            f"{count} {word}"
            for kind, word in _WORDS.items()
            if kind != "demonstrated_feasible" and (count := sum(row["determination"] == kind for row in rows))
        ]
        severity: Severity = "ok" if demonstrated == len(rows) else "warning"
        return [
            Finding(
                severity=severity,
                title=self.title,
                brief=", ".join([f"{demonstrated} of {len(rows)} targets demonstrated", *others]),
                blocks=[_table(rows), _basis(output, rows)],
            )
        ]


def _tenths(value: float, direction: Literal["lower", "upper"]) -> int:
    """*value*, a fraction, in whole tenths of a percent: a lower bound rounded down and an upper bound up, as
    DataEval's printout rounds, so that rounding never overstates a bound. The ``1e-9`` absorbs float error only, such
    as ``0.957 * 1000 = 956.9999999999999``."""
    tenths = value * _WHOLE
    return math.floor(tenths + 1e-9) if direction == "lower" else math.ceil(tenths - 1e-9)


def _pct(value: float, direction: Literal["lower", "upper"]) -> str:
    """*value* as DataEval's printout writes a bound: to 0.1%, rounded down for a lower bound and up for an upper."""
    return f"{_tenths(value, direction) / 10:.1f}%"


def _label(row: Mapping[str, Any]) -> str:
    """The target as the evaluator's `target:` key writes it, and its threshold in percent: ``accuracy ≥ 90%``."""
    spec = json.loads(row["target_spec"])
    name = f"{row['metric']}:{spec['class_name']}" if spec["class_name"] else str(row["metric"])
    scope = "" if row["judged_scope"] == "full_input" else f"/{row['judged_scope']}"
    aggregation = f"|{spec['aggregation']}" if spec["aggregation"] else ""
    macro = " (macro)" if row["metric"] == "ap" and not spec["class_name"] else ""
    return f"{name}{scope}{aggregation}{macro} ≥ {row['threshold'] * 100:g}%"


def _interval(row: Mapping[str, Any]) -> list[float]:
    """The whole 0 to 1 scale as shown, open and ruled out, the first rounded down and the last up so that neither
    claims more than the bounds do. Where a lower bound is above an upper bound, neither is trusted: all of it is
    open."""
    lower, ceiling = row["witness"], row["ceiling"]
    shown = 0 if lower is None else _tenths(lower, "lower")
    ruled_out = 0 if ceiling is None else _WHOLE - _tenths(ceiling, "upper")
    if shown + ruled_out > _WHOLE or row["determination"] == "evidence_conflict":
        shown = ruled_out = 0
    return [part / _WHOLE for part in (shown, _WHOLE - shown - ruled_out, ruled_out)]


def _bounds(row: Mapping[str, Any]) -> str:
    """What the evidence bounds the target's accuracy by, as DataEval rounds it: ``≥42.4%`` where only a lower bound
    was found, ``41.6–75.1%`` between two, a ``*`` on an upper bound that rests on allowances. A space after the
    ``≥`` would push a scoped target's row past 80 columns."""
    lower, ceiling = row["witness"], row["ceiling"]
    if ceiling is None:
        return "—" if lower is None else f"≥{_pct(lower, 'lower')}"
    upper = _pct(ceiling, "upper") + ("*" if row["determination"] == "infeasible_under_recorded_assumptions" else "")
    return f"≤{upper}" if lower is None else f"{_pct(lower, 'lower')[:-1]}–{upper}"


def _table(rows: list[dict[str, Any]]) -> Table:
    """A row per target: a mark for its answer, the target, the interval its bounds leave, and the bounds. The key to
    the marks is under the table."""
    columns = [
        Column(key="answer", header="☐"),  # the mark of each answer, keyed under the table
        Column(key="target", header="Target", align="left"),
        Column(key="interval", kind="stacked", series=["shown", "open", "ruled out"]),
        Column(key="bounds", header="Bounds"),
    ]
    cells: list[dict[str, Cell]] = [
        {
            "answer": _SYMBOLS[row["determination"]],
            "target": _label(row),
            "interval": _interval(row),
            "bounds": _bounds(row),
        }
        for row in rows
    ]
    return Table(columns=columns, rows=cells)


def _key(rows: list[dict[str, Any]]) -> str:
    """The marks the table uses and the answer each stands for, in DataEval's words, as ``✓ demonstrated``."""
    kinds = {row["determination"] for row in rows}
    return "  ".join(
        f"{symbol} {_WORDS[kind].split(' under ')[0]}" for kind, symbol in _SYMBOLS.items() if kind in kinds
    )


def _confidence(output: FeasibilityOutput) -> str | None:
    """The confidence every bound holds at together, as DataEval's printout words it, rounded down; ``None`` where the
    call recorded none."""
    alpha = next((ev.details["call_alpha"] for ev in output.evidence.values() if "call_alpha" in ev.details), None)
    return None if alpha is None else _pct(1 - alpha, "lower").replace(".0%", "%")


def _allowances(output: FeasibilityOutput) -> str:
    """The allowances any evidence of the call records, such as ``beta=0.0, gamma=0.0``; empty where none was given."""
    return next(
        (
            ", ".join(f"{name}={value}" for name, value in allowed.items())
            for evidence in output.evidence.values()
            if (allowed := evidence.details.get("allowances"))
        ),
        "",
    )


def _basis(output: FeasibilityOutput, rows: list[dict[str, Any]]) -> Fields:
    """What the table rests on, a line each: the key to its marks, the confidence, the population where it is not simply
    every item the source holds, how the items were grouped into independent units, and the allowances if any. What
    each answer means, the assumption every answer rests on, and what would settle a target are in the documentation,
    and in the evaluator's output."""
    confidence = _confidence(output)
    plain = not isinstance(output, PopulatedFeasibilityOutput) or (
        output.labeled == output.items and not output.absent_classes
    )
    same = (
        f"like these {output.labeled:,}"
        if isinstance(output, PopulatedFeasibilityOutput) and plain
        else "from this population"
    )
    items: list[tuple[str, Scalar]] = [
        ("Key", _key(rows)),
        (
            "Confidence",
            f"{confidence}, all bounds together, for new images {same}" if confidence else f"for new images {same}",
        ),
    ]
    if isinstance(output, PopulatedFeasibilityOutput) and not plain:
        items.append(("Population", output.population()))
    if isinstance(output, PopulatedFeasibilityOutput) and output.groups:  # zero where the grouping was not recorded
        items.append(("Grouping", output.grouping()))
    if allowances := _allowances(output):
        starred = any(row["determination"] == "infeasible_under_recorded_assumptions" for row in rows)
        items.append(("Allowances", f"{allowances}, asserted by you" + ("; * rests on them" if starred else "")))
    return Fields(items=items)
