"""The `uncovered-rate` check: how much of a Dataset coverage left uncovered (data-splitting spec §6.2)."""

__all__ = ["UncoveredRateCheck", "UncoveredRateConfig"]

from collections.abc import Mapping
from typing import Any, ClassVar

from dataeval.scope import CoverageOutput
from pydantic import Field

from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.checks._limits import exceeds
from dataeval_flow.workflows._base import Finding


class UncoveredRateConfig(CheckConfig):
    """An `uncovered-rate` step's input, and the share uncovered that may hold."""

    input: str = Field(description="A `coverage` Output.")
    rate: float | None = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description=(
            "The percent of the Dataset's items uncovered past which the finding warns; `null` judges nothing. Judge "
            "only `naive` coverage: adaptive coverage marks `percent` of the items uncovered by construction. "
            "data-coverage's `health_thresholds.uncovered_rate`."
        ),
    )


class UncoveredRateCheck(Check[UncoveredRateConfig]):
    """``uncovered-rate``: warns when more than ``rate`` percent of a Dataset's items are uncovered."""

    name: ClassVar[str] = "uncovered-rate"
    description: ClassVar[str] = "Warns when more than `rate` percent of a Dataset's items are uncovered."
    title: ClassVar[str] = "Uncovered Rate"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(CoverageOutput,)),)

    def run(self, config: UncoveredRateConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """How many items were uncovered, of how many."""
        node = inputs["input"]
        total = node.items or 0
        uncovered = len(node.value.uncovered_indices)
        percent = uncovered / total * 100 if total else 0.0
        return [
            Finding(
                severity="warning" if exceeds(percent, config.rate) else "info",
                title=self.title,
                brief=f"{uncovered} of {total} uncovered ({round(percent, 1)}%)",
            )
        ]
