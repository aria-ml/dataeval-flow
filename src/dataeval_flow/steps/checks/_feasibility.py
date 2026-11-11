"""The `performance-feasibility` check: whether the data demonstrates each performance target reachable, drawn as one
row per target, with the bounds that decide it."""

__all__ = ["PerformanceFeasibilityCheck", "PerformanceFeasibilityConfig"]

import math
from collections.abc import Mapping
from typing import Any, ClassVar, Literal

from dataeval.performance import FeasibilityOutput
from pydantic import Field

from dataeval_flow._blocks import Cell, Column, Fields, Scalar, Table
from dataeval_flow.evaluators.performance._feasibility_result import PopulatedFeasibilityOutput, requirement_label
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.checks._limits import Severity
from dataeval_flow.workflows._base import Finding

# The mark each determination gets in the table's first column. DataEval's printout marks nothing; "not ruled out" is
# ``~`` and "unresolved" ``?`` so that a column of marks alone tells the two rows apart. An `unresolved` determination
# whose limitation is `conflict` is ``!``: its bounds contradict each other and neither is trusted.
_SYMBOLS = {
    "demonstrated_feasible": "✓",
    "infeasible_under_recorded_assumptions": "✗",
    "not_ruled_out": "~",
    "unresolved": "?",
}
_CONFLICT = "!"
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
        """A table of a row per target, in the order the evaluator was given them, what the answers rest on, and the
        bounds that decide them.

        ``ok`` only where every target is ``demonstrated_feasible``; ``warning`` for any other determination,
        including one that rules a target out, since that holds only under the assumptions recorded with it, and
        ``not_ruled_out`` or ``unresolved``, which show nothing either way. Each row's bounds are DataEval's, a lower
        bound rounded down and an upper bound up, as its printout rounds them; the unrounded bounds and their
        assumptions stay in the evaluator's output.
        """
        output: FeasibilityOutput = inputs["input"].value
        rows = list(output.determinations.iter_rows(named=True))
        demonstrated = sum(row["determination"] == "demonstrated_feasible" for row in rows)
        others = [
            f"{count} {kind}"
            for kind in _SYMBOLS
            if kind != "demonstrated_feasible" and (count := sum(row["determination"] == kind for row in rows))
        ]
        severity: Severity = "ok" if demonstrated == len(rows) else "warning"
        return [
            Finding(
                severity=severity,
                title=self.title,
                brief=", ".join([f"{demonstrated} of {len(rows)} targets demonstrated_feasible", *others]),
                blocks=[_table(rows), _basis(output, rows), _deciding(output, rows)],
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
    """The requirement as DataEval's printout names it: ``accuracy ≥0.9``, with its class, aggregation and scope."""
    return requirement_label(
        row["metric"], row["threshold"], row["class_name"], row["aggregation"], row["judged_scope"]
    )


def _mark(row: Mapping[str, Any]) -> str:
    return _CONFLICT if row["limitation"] == "conflict" else _SYMBOLS[row["determination"]]


def _interval(row: Mapping[str, Any]) -> list[float]:
    """The whole 0 to 1 scale as shown, open and ruled out, the first rounded down and the last up so that neither
    claims more than the bounds do. Where a lower bound is above an upper bound, neither is trusted: all of it is
    open."""
    lower, envelope = row["witness"], row["envelope"]
    shown = 0 if lower is None else _tenths(lower, "lower")
    ruled_out = 0 if envelope is None else _WHOLE - _tenths(envelope, "upper")
    if shown + ruled_out > _WHOLE or row["limitation"] == "conflict":
        shown = ruled_out = 0
    return [part / _WHOLE for part in (shown, _WHOLE - shown - ruled_out, ruled_out)]


def _bounds(row: Mapping[str, Any]) -> str:
    """What the bounds put the target's metric between, as DataEval rounds it: ``≥42.4%`` where only a witness's lower
    bound was found, ``41.6–75.1%`` between that and an envelope's upper bound, a ``*`` on an upper bound that rests
    on allowances. A space after the ``≥`` would push a scoped target's row past 80 columns."""
    lower, envelope = row["witness"], row["envelope"]
    if envelope is None:
        return "—" if lower is None else f"≥{_pct(lower, 'lower')}"
    upper = _pct(envelope, "upper") + ("*" if row["determination"] == "infeasible_under_recorded_assumptions" else "")
    return f"≤{upper}" if lower is None else f"{_pct(lower, 'lower')[:-1]}–{upper}"


def _table(rows: list[dict[str, Any]]) -> Table:
    """A row per target: a mark for its determination, its index, the target, the interval its bounds leave, and the
    bounds. The key to the marks is under the table."""
    columns = [
        Column(key="answer", header="☐"),  # the mark of each determination, keyed under the table
        Column(key="requirement", header="#"),
        Column(key="target", header="Target", align="left"),
        Column(key="interval", kind="stacked", series=["shown", "open", "ruled out"]),
        Column(key="bounds", header="Bounds"),
    ]
    cells: list[dict[str, Cell]] = [
        {
            "answer": _mark(row),
            "requirement": row["requirement"],
            "target": _label(row),
            "interval": _interval(row),
            "bounds": _bounds(row),
        }
        for row in rows
    ]
    return Table(columns=columns, rows=cells)


def _key(rows: list[dict[str, Any]]) -> str:
    """The marks the table uses and the determination each stands for, in DataEval's names, as ``✓
    demonstrated_feasible``; ``!`` is a conflict, which DataEval records as ``unresolved`` with the limitation
    ``conflict``."""
    marks = {_mark(row): row["determination"] for row in rows}
    names = {**{symbol: kind for kind, symbol in _SYMBOLS.items()}, _CONFLICT: "unresolved, conflict"}
    return "  ".join(f"{mark} {names[mark]}" for mark in [*_SYMBOLS.values(), _CONFLICT] if mark in marks)


def _confidence(output: FeasibilityOutput) -> str:
    """The confidence every bound holds at together, as DataEval's printout words it, rounded down.

    Raises
    ------
    ValueError
        When the output records no ``alpha``: a confidence is not guessed, and one that is left out would read as no
        claim of confidence at all.
    """
    alpha = output.meta().state.get("alpha")
    if alpha is None:
        raise ValueError(
            "This FeasibilityOutput records no `alpha`, the error budget its bounds hold at, so no confidence can be "
            "stated for them. Judge an output that `Feasibility.evaluate` or the `feasibility` evaluator returned."
        )
    return _pct(1 - alpha, "lower").replace(".0%", "%")


def _allowances(output: FeasibilityOutput) -> str:
    """The allowances the call was given, such as ``beta=0.0, gamma=0.0``; empty where none was given."""
    allowed = output.allowances
    if allowed is None:
        return ""
    return f"beta={allowed.beta}" + ("" if allowed.gamma is None else f", gamma={allowed.gamma}")


def _basis(output: FeasibilityOutput, rows: list[dict[str, Any]]) -> Fields:
    """What the table rests on, a line each: the key to its marks, the confidence, the population where it is not simply
    every item the source holds, how the items were grouped into independent units, the allowances if any, and for each
    undecided target its limitation and the suggested minimum where DataEval gives one. What each limitation means, and
    the assumption every bound rests on, are in the documentation, and in the evaluator's output."""
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
        ("Confidence", f"{confidence}, all bounds together, for new images {same}"),
    ]
    if isinstance(output, PopulatedFeasibilityOutput) and not plain:
        items.append(("Population", output.population()))
    if isinstance(output, PopulatedFeasibilityOutput) and output.groups:  # zero where the grouping was not recorded
        items.append(("Grouping", output.grouping()))
    if allowances := _allowances(output):
        starred = any(row["determination"] == "infeasible_under_recorded_assumptions" for row in rows)
        items.append(("Allowances", f"{allowances}, asserted by you" + ("; * rests on them" if starred else "")))
    if limited := [f"{_label(row)}: {row['limitation']}" for row in rows if row["limitation"] is not None]:
        items.append(("Limitation", "\n".join(limited)))
    if minimum := [
        f"{_label(row)}: {row['suggested_minimum']:,} independent units, from {row['minimum_source']} "
        f"{row['minimum_quantity']}"
        for row in rows
        if row["suggested_minimum"] is not None
    ]:
        items.append(("Suggested minimum", "\n".join(minimum)))
    return Fields(items=items)


def _deciding(output: FeasibilityOutput, rows: list[dict[str, Any]]) -> Table:
    """The bounds each determination rests on, and the informative envelopes that look like they could rule a target
    out but cannot, because they hold at a narrower scope than the target is judged at: ``Decides`` says which."""
    judged = {row["requirement"]: row["judged_scope"] for row in rows}
    columns = [
        Column(key="requirement", header="#"),
        Column(key="role", header="Role", align="left"),
        Column(key="source", header="Source", align="left"),
        Column(key="bound", header="Bound"),
        Column(key="scope", header="Scope", align="left"),
        Column(key="decides", header="Decides", align="left"),
    ]
    cells: list[dict[str, Cell]] = [
        {
            "requirement": bound["requirement"],
            "role": bound["role"],
            "source": bound["source"],
            "bound": (
                f"≥{_pct(bound['bound'], 'lower')}"
                if bound["bound_type"] == "lower"
                else f"≤{_pct(bound['bound'], 'upper')}"
            ),
            "scope": bound["information_scope"],
            "decides": "yes" if bound["is_determining"] else "no",
        }
        for bound in output.bounds.iter_rows(named=True)
        if bound["bound"] is not None
        and (
            bound["is_determining"]
            or (
                bound["role"] == "envelope"
                and bound["state"] == "informative"
                and bound["information_scope"] != judged[bound["requirement"]]
            )
        )
    ]
    return Table(columns=columns, rows=cells)
