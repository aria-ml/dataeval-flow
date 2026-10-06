"""The bias checks: metadata factors that tell much about the class (`shortcut-risk`, audit spec §10.2, §19 I7), and
factors significantly associated with it (`factor-parity`)."""

__all__ = ["FactorParityCheck", "FactorParityConfig", "ShortcutRiskCheck", "ShortcutRiskConfig"]

from collections.abc import Mapping
from typing import Any, ClassVar

from dataeval.bias import BalanceOutput, ParityOutput
from pydantic import Field

from dataeval_flow._blocks import Cell, Column, Table
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
    description: ClassVar[str] = "Warns when a metadata factor tells much about the class."
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
        description="The chi-square p-value at or under which a factor's association counts as significant.",
    )


class FactorParityCheck(Check[FactorParityConfig]):
    """``factor-parity``: the metadata factors whose association with the class is both strong and significant."""

    name: ClassVar[str] = "factor-parity"
    description: ClassVar[str] = "Warns when a metadata factor is significantly associated with the class."
    title: ClassVar[str] = "Factor Parity"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(ParityOutput,)),)

    def run(self, config: FactorParityConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The factors past both limits, strongest first, and every factor's score as a table.

        Raises
        ------
        StepSkipped
            When the Output scores no factor: recorded as not assessed.
        """
        rows = sorted(inputs["input"].value.factors.to_dicts(), key=lambda row: -row["score"])
        if not rows:
            raise StepSkipped("no factor to score")
        over = [row for row in rows if exceeds(row["score"], config.warning) and row["p_value"] <= config.p_value]
        sparse = [row["factor_name"] for row in rows if row["has_insufficient_data"]]
        severity: Severity = "info" if config.warning is None else ("warning" if over else "ok")
        if over:
            listed = ", ".join(f"{row['factor_name']} (V={row['score']:.2f})" for row in over[:3])
            brief = f"{len(over)} of {len(rows)} factors associated with the class: {listed}"
        else:
            brief = f"{len(rows)} factors checked"
        description = (
            "A factor whose values differ by class is a shortcut a model can learn instead of the task. The score is "
            "the bias-corrected Cramér's V between the factor and the class; the p-value is the chi-square test's."
        )
        if sparse:
            description += (
                f" {len(sparse)} factors have contingency cells expected to hold fewer than 5 items, so their p-values "
                f"are unreliable: {', '.join(sparse)}."
            )
        table_rows: list[dict[str, Cell]] = [
            {
                "factor": row["factor_name"],
                "score": row["score"],
                "p_value": row["p_value"],
                "sparse": "yes" if row["has_insufficient_data"] else "",
            }
            for row in rows
        ]
        columns = [
            Column(key="factor", header="Factor"),
            Column(key="score", header="Cramér's V", format="{:.3f}"),
            Column(key="p_value", header="p-value", format="{:.3g}"),
            Column(key="sparse", header="Sparse"),
        ]
        return [
            Finding(
                severity=severity,
                title=self.title,
                brief=brief,
                description=description,
                blocks=[Table(columns=columns, rows=table_rows)],
            )
        ]
