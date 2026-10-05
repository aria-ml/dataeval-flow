"""The `factor-coverage-gaps` check: legacy data-coverage's Metadata Coverage Gaps finding (coverage spec §6.2)."""

__all__ = ["FactorCoverageGapsCheck", "FactorCoverageGapsConfig"]

from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import Field

from dataeval_flow._blocks import Block, Cell, Column, Table
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.checks._limits import Severity
from dataeval_flow.steps.combines._gaps import FactorGapsOutput
from dataeval_flow.workflows._base import Finding


class FactorCoverageGapsConfig(CheckConfig):
    """A `factor-coverage-gaps` step's input, and how many gaps make a warning."""

    input: str = Field(description="A `factor-gaps` Output.")
    warning: int | None = Field(
        default=2,
        ge=0,
        description=(
            "The most under-represented class-factor-value combinations before the finding warns; `null` never warns."
        ),
    )


class FactorCoverageGapsCheck(Check[FactorCoverageGapsConfig]):
    """``factor-coverage-gaps``: warns past `warning` gaps, informs with up to that many, and is ok with none."""

    name: ClassVar[str] = "factor-coverage-gaps"
    description: ClassVar[str] = "Warns when enough class-factor-value combinations are under-represented."
    title: ClassVar[str] = "Factor Coverage Gaps"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(FactorGapsOutput,)),)

    def run(self, config: FactorCoverageGapsConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The gaps' count against `warning`, with the gaps as a table."""
        gaps = inputs["input"].value.gaps
        if not gaps:
            return [
                Finding(
                    severity="ok",
                    title=self.title,
                    brief="No significant gaps detected",
                    description="No class-factor-value combinations are significantly under-represented.",
                )
            ]
        severity: Severity = "warning" if config.warning is not None and len(gaps) > config.warning else "info"
        rows: list[dict[str, Cell]] = [
            {
                "class": gap.class_name,
                "factor": gap.factor_name,
                "value": gap.factor_value,
                "count": gap.class_count,
                "expected": round(gap.expected_count, 1),
                "deficit": round(gap.deficit * 100, 1),
            }
            for gap in gaps
        ]
        columns = [
            Column(key="class", header="Class"),
            Column(key="factor", header="Factor"),
            Column(key="value", header="Value"),
            Column(key="count", header="Count"),
            Column(key="expected", header="Expected"),
            Column(key="deficit", header="Deficit", format="{:.1f}%"),
        ]
        blocks: list[Block] = [Table(columns=columns, rows=rows)]
        return [
            Finding(
                severity=severity,
                title=self.title,
                brief=f"{len(gaps)} gaps identified",
                description=(
                    f"{len(gaps)} class-factor-value combinations are under-represented. "
                    "These represent gaps in data collection that may affect model performance."
                ),
                blocks=blocks,
            )
        ]
