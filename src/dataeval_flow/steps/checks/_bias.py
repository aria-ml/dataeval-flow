"""The `shortcut-risk` check: metadata factors that tell much about the class (audit spec §10.2, §19 I7)."""

__all__ = ["ShortcutRiskCheck", "ShortcutRiskConfig"]

from collections.abc import Mapping
from typing import Any, ClassVar

from dataeval.bias import BalanceOutput
from pydantic import Field

from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.checks._limits import Severity, exceeds
from dataeval_flow.workflows._base import Finding


class ShortcutRiskConfig(CheckConfig):
    """A `shortcut-risk` step's input, and how much a factor may tell about the class."""

    input: str = Field(description="A `balance` Output.")
    mutual_information: float | None = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "The mutual information with the class, from 0 to 1, past which a factor warns; `null` judges nothing. "
            "Legacy data-analysis's 0.1."
        ),
    )


class ShortcutRiskCheck(Check[ShortcutRiskConfig]):
    """``shortcut-risk``: the metadata factors whose mutual information with the class is past a limit."""

    name: ClassVar[str] = "shortcut-risk"
    description: ClassVar[str] = "Warns when a metadata factor tells much about the class."
    title: ClassVar[str] = "Shortcut Risk"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(BalanceOutput,)),)

    def run(self, config: ShortcutRiskConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The factors past the limit, most informative first, and every factor's mutual information."""
        from dataeval_flow.evaluators.bias._report import ranked_table
        from dataeval_flow.steps.combines._gaps import mi_from_balance

        balance = inputs["input"].value
        # Balance scores exactly the policy's factors plus `class_label`, which is 1.0 by definition (spec §19 I7).
        names = [str(name) for name in balance.balance["factor_name"].to_list() if name != "class_label"]
        mi = mi_from_balance(balance, names)
        over = [
            (name, value)
            for name, value in sorted(mi.items(), key=lambda item: -item[1])
            if exceeds(value, config.mutual_information)
        ]
        severity: Severity = "info" if config.mutual_information is None else ("warning" if over else "ok")
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
                blocks=[ranked_table(mi, headers=("Factor", "MI with the class"))] if mi else [],
            )
        ]
