"""The bias checks: metadata factors that tell much about the class (`shortcut-risk`, audit spec §10.2, §19 I7), and
factors significantly associated with it, with the factor-level combinations to collect (`factor-parity`)."""

__all__ = ["FactorParityCheck", "FactorParityConfig", "ShortcutRiskCheck", "ShortcutRiskConfig"]

import math
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

from dataeval.bias import BalanceOutput, ParityOutput
from pydantic import Field

from dataeval_flow._blocks import Block, Cell, Column, Paragraph, Table
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import StepSkipped
from dataeval_flow.steps.checks._limits import Severity, exceeds
from dataeval_flow.workflows._base import Finding


class ShortcutRiskConfig(CheckConfig):
    """A `shortcut-risk` step's input, and how much a factor may tell about the class."""

    input: str = Field(description="A `balance` Output.")
    warning: float | None = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "The mutual information with the class, from 0 to 1, past which a factor warns; `null` judges nothing."
        ),
    )


class ShortcutRiskCheck(Check[ShortcutRiskConfig]):
    """``shortcut-risk``: the metadata factors whose mutual information with the class is past a limit."""

    name: ClassVar[str] = "shortcut-risk"
    description: ClassVar[str] = "Judges `balance`'s output: warns when a metadata factor tells much about the class."
    title: ClassVar[str] = "Shortcut Risk"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(BalanceOutput,)),)

    def run(self, config: ShortcutRiskConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The factors past the limit, most informative first, and every factor's mutual information.

        Raises
        ------
        StepSkipped
            When no factor is left to score once `class_label` is dropped: recorded as not assessed.
        """
        from dataeval_flow.evaluators.bias._report import ranked_table
        from dataeval_flow.steps.combines._gaps import mi_from_balance

        balance = inputs["input"].value
        # Balance scores exactly the policy's factors plus `class_label`, which is 1.0 by definition (spec §19 I7).
        names = [str(name) for name in balance.balance["factor_name"].to_list() if name != "class_label"]
        mi = mi_from_balance(balance, names)
        if not mi:
            raise StepSkipped("no factor to score")
        over = [
            (name, value)
            for name, value in sorted(mi.items(), key=lambda item: -item[1])
            if exceeds(value, config.warning)
        ]
        severity: Severity = "info" if config.warning is None else ("warning" if over else "ok")
        if over:
            listed = ", ".join(f"{name} (MI={value:.2f})" for name, value in over[:3])
            brief = f"{len(over)} of {len(mi)} factors tied to the class: {listed}"
        else:
            brief = f"{len(mi)} factors checked"
        return [
            Finding(
                severity=severity,
                title=self.title,
                brief=brief,
                description=(
                    "A factor that tells much about the class is a shortcut a model can learn instead of the task. "
                    "Mutual information is the share of the class's entropy the factor accounts for."
                ),
                blocks=[ranked_table(mi, headers=("Factor", "MI with the class"))],
            )
        ]


class FactorParityConfig(CheckConfig):
    """A `factor-parity` step's input, and how strong and how significant an association warns."""

    input: str = Field(description="A `parity` Output.")
    warning: float | None = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "The bias-corrected Cramér's V with the class, from 0 to 1, past which a significant factor warns; `null` "
            "judges nothing."
        ),
    )
    p_value: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description=(
            "The adjusted p-value (Benjamini-Hochberg, across the factors) at or under which a factor's association "
            "counts as significant."
        ),
    )


# Why a factor of each status was not assessed. Neither reads as passing, since neither says the factor is clear.
_NOT_ASSESSED = {
    "untestable": "too few levels, classes or groups to test",
    "inconclusive": "not significant, but on too little data to call it a negative",
}
_TESTS = {"pearson": "chi-square", "rao_scott_f": "Rao-Scott F"}


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _listed(unassessed: Mapping[str, list[str]]) -> str:
    """The factors not assessed by status, each with why: ``untestable: a, b (reason); inconclusive: c (reason)``."""
    return "; ".join(f"{status}: {', '.join(names)} ({_NOT_ASSESSED[status]})" for status, names in unassessed.items())


def _by_deficit(cells: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """`cells`, the largest deficit first."""
    return sorted(cells, key=lambda cell: -cell["deficit"])


def _thin_short(cells: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The cells too thin to judge that hold fewer items than expected, the largest deficit first.

    A thin cell holding its expected share or more cannot fall short, so there is nothing to collect for it.
    """
    return _by_deficit(cell for cell in cells if cell["status"] == "too_thin" and cell["deficit"] > 0)


def _thin_summary(thin: list[dict[str, Any]]) -> str:
    """How many thin cells fall short and the three furthest, for a reason the findings cannot carry; empty for none."""
    if not thin:
        return ""
    top = ", ".join(
        f"{cell['factor_name']}: {cell['level']} × {cell['class_name']} (deficit {cell['deficit']:.1f})"
        for cell in thin[:3]
    )
    verb = "holds" if len(thin) == 1 else "hold"
    return f". {_count(len(thin), 'thin cell')} {verb} fewer items than expected; the largest deficits: {top}"


def _cell_table(cells: list[dict[str, Any]], *, judged: bool) -> Table:
    """`cells` as a table; a cell too thin to judge has no lift or p-value shown, since neither can be trusted."""
    columns = [
        Column(key="factor", header="Factor"),
        Column(key="level", header="Level"),
        Column(key="class", header="Class"),
        Column(key="observed", header="Observed"),
        Column(key="expected", header="Expected", format="{:.1f}"),
        Column(key="deficit", header="Deficit", format="{:.1f}"),
    ]
    if judged:
        columns += [
            Column(key="lift", header="Lift", format="{:.2f}"),
            Column(key="p_value", header="Adj. p-value", format="{:.3g}"),
        ]
    rows: list[dict[str, Cell]] = [
        {
            "factor": cell["factor_name"],
            "level": cell["level"],
            "class": cell["class_name"],
            "observed": cell["observed"],
            "expected": cell["expected"],
            "deficit": cell["deficit"],
            **({"lift": cell["lift"], "p_value": cell["p_value_adjusted"]} if judged else {}),
        }
        for cell in cells
    ]
    return Table(columns=columns, rows=rows, preview=10)


def _number(value: float) -> float | None:
    """A score as a cell: ``None`` where there is none, as for a factor that could not be tested."""
    return None if math.isnan(value) else value


class FactorParityCheck(Check[FactorParityConfig]):
    """``factor-parity``: the metadata factors whose association with the class is both strong and significant, and
    the factor-level combinations to collect."""

    name: ClassVar[str] = "factor-parity"
    description: ClassVar[str] = (
        "Judges `parity`'s output: warns when a metadata factor is significantly associated with the class, and lists "
        "the cells to collect."
    )
    experimental: ClassVar[bool] = True  # judges `parity`, which is experimental
    title: ClassVar[str] = "Factor Parity"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(ParityOutput,)),)

    def run(self, config: FactorParityConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The factors past both limits, strongest first, every factor's score as a table, and the cells to collect.

        A factor that is untestable or inconclusive is never read as clear: it is listed as not assessed, and the
        finding is at best information. Only a skip moves the verdict, so where no factor was assessed the check is
        skipped, with the thin cells short of their expected count summarized in the reason, since a skip replaces
        every finding; where some were, a finding is all it can make, and the verdict stays ready if nothing warns.

        Raises
        ------
        StepSkipped
            When the Output scores no factor, or every factor is untestable or inconclusive: recorded as not assessed.
        """
        output = inputs["input"].value
        rows = sorted(
            output.factors.to_dicts(), key=lambda row: math.inf if row["status"] == "untestable" else -row["score"]
        )
        if not rows:
            raise StepSkipped("no factor to score")
        unassessed = {
            status: [row["factor_name"] for row in rows if row["status"] == status] for status in _NOT_ASSESSED
        }
        unassessed = {status: names for status, names in unassessed.items() if names}
        skipped = sum(len(names) for names in unassessed.values())
        if skipped == len(rows):  # a finding would leave the verdict ready; only a skip makes it ready-with-caveats
            # The skip replaces every finding, the cells' too, so what they would list rides in the reason.
            thin = _thin_summary(_thin_short(output.cells.to_dicts()))
            raise StepSkipped(f"no factor could be assessed, {_listed(unassessed)}{thin}")
        over = [
            row for row in rows if exceeds(row["score"], config.warning) and row["p_value_adjusted"] <= config.p_value
        ]
        sparse = [
            row["factor_name"] for row in rows if row["has_insufficient_data"] and row["status"] not in _NOT_ASSESSED
        ]
        severity: Severity
        if config.warning is None:
            severity = "info"
        elif over:
            severity = "warning"
        else:
            severity = "info" if skipped else "ok"
        if over:
            listed = ", ".join(f"{row['factor_name']} (V={row['score']:.2f})" for row in over[:3])
            brief = f"{len(over)} of {len(rows)} factors associated with the class: {listed}"
        else:
            brief = f"{len(rows)} factors checked"
        if skipped:
            brief += f", {skipped} not assessed"
        test = _TESTS.get(output.test or "", "factor")
        description = (
            "A factor whose values differ by class is a shortcut a model can learn instead of the task. The score is "
            f"the bias-corrected Cramér's V between the factor and the class; the p-value is the {test} test's, "
            "adjusted across the factors."
        )
        if unassessed:
            description += f" Not assessed, so not passing: {_listed(unassessed)}."
        if sparse:
            description += (
                f" {len(sparse)} factors have tables too thin to trust the reference distribution (expected counts "
                f"under 5, or too few groups), so their p-values are unreliable: {', '.join(sparse)}."
            )
        table_rows: list[dict[str, Cell]] = [
            {
                "factor": row["factor_name"],
                "status": row["status"],
                "score": _number(row["score"]),
                "p_value": _number(row["p_value_adjusted"]),
                "sparse": "yes" if row["has_insufficient_data"] else "",
            }
            for row in rows
        ]
        columns = [
            Column(key="factor", header="Factor"),
            Column(key="status", header="Status"),
            Column(key="score", header="Cramér's V", format="{:.3f}"),
            Column(key="p_value", header="Adj. p-value", format="{:.3g}"),
            Column(key="sparse", header="Sparse"),
        ]
        findings = [
            Finding(
                severity=severity,
                title=self.title,
                brief=brief,
                description=description,
                blocks=[Table(columns=columns, rows=table_rows)],
            )
        ]
        if collect := self._collect(output):
            findings.append(collect)
        return findings

    def _collect(self, output: ParityOutput) -> Finding | None:
        """The under-represented cells, and the thin cells short of their expected count, largest deficit first.

        Information, never a warning: a shortfall is a place to collect, and nothing here judges how much is too much.
        ``None`` where there is neither.
        """
        cells = output.cells.to_dicts()
        short = _by_deficit(cell for cell in cells if cell["status"] == "underrepresented")
        thin = _thin_short(cells)
        if not short and not thin:
            return None
        parts: list[str] = []
        blocks: list[Block] = []
        if short:
            parts.append(_count(len(short), "under-represented cell"))
            blocks.append(_cell_table(short, judged=True))
        if thin:
            parts.append(f"{_count(len(thin), 'cell')} too thin to judge")
            blocks.append(
                Paragraph(
                    text=(
                        "Too thin to judge, and holding fewer items than expected. Collect more of these to find out "
                        "whether they fall short, the largest deficit first:"
                    )
                )
            )
            blocks.append(_cell_table(thin, judged=False))
        return Finding(
            severity="info",
            title="Parity Collection Gaps",
            brief=", ".join(parts),
            description=(
                "A combination of a factor's level and a class that holds well fewer items than the class's share "
                "across the factor predicts is one to collect more of. The deficit is the items expected less the "
                "items observed, and the lift is observed over expected."
            ),
            blocks=blocks,
        )
