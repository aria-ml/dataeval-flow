"""The `uncovered-rate` check: how much of a Dataset coverage left uncovered (data-splitting spec §6.2)."""

__all__ = ["CompletenessScoreCheck", "CompletenessScoreConfig", "UncoveredRateCheck", "UncoveredRateConfig"]

from collections.abc import Mapping
from typing import Any, ClassVar, Self

from dataeval.scope import CoverageOutput
from pydantic import Field, model_validator

from dataeval_flow._blocks import Fields
from dataeval_flow.evaluators.scope import CompletenessOutput
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.checks._limits import Severity, exceeds
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


class CompletenessScoreConfig(CheckConfig):
    """A `completeness-score` step's input, and the bands of its score."""

    input: str = Field(description="A `completeness` Output.")
    warning: float | None = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="The score under which the finding warns; `null` turns it off. Legacy `completeness_score`.",
    )
    info: float | None = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="The score under which the finding informs; `null` turns it off. Legacy's hard-coded 0.8.",
    )

    @model_validator(mode="after")
    def _warning_under_info(self) -> Self:
        if self.warning is not None and self.info is not None and self.warning > self.info:
            raise ValueError(f"`warning` ({self.warning}) must not exceed `info` ({self.info}).")
        return self


class CompletenessScoreCheck(Check[CompletenessScoreConfig]):
    """``completeness-score``: legacy data-coverage's Dimensional Completeness finding (coverage spec §6.2)."""

    name: ClassVar[str] = "completeness-score"
    description: ClassVar[str] = "Warns when the embeddings fill too little of their space's dimensions."
    title: ClassVar[str] = "Dimensional Completeness"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(CompletenessOutput,)),)

    def run(self, config: CompletenessScoreConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The score, rounded to three places, against the two bands."""
        data = inputs["input"].value.data()
        score = round(float(data["completeness"]), 3)
        if config.warning is None and config.info is None:
            severity: Severity = "info"
        elif config.warning is not None and score < config.warning:
            severity = "warning"
        elif config.info is not None and score < config.info:
            severity = "info"
        else:
            severity = "ok"
        threshold = f" (threshold: {config.warning})" if config.warning is not None else ""
        return [
            Finding(
                severity=severity,
                title=self.title,
                brief=f"Completeness: {score}",
                description=f"Dimensional completeness score is {score}{threshold}.",
                blocks=[
                    Fields(
                        items=[
                            ("Completeness Score", score),
                            ("Nearest Neighbor Pairs", len(data["nearest_neighbor_pairs"])),
                        ]
                    )
                ],
            )
        ]
