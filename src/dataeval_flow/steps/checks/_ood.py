"""The `ood` check: how much of a test source an OOD detector flagged (ood-detection spec §5.1)."""

__all__ = ["OODCheck", "OODCheckConfig", "OODThresholds", "assessed_images", "ood_severity"]

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import numpy as np
from dataeval.shift import OODOutput
from pydantic import BaseModel, ConfigDict, Field

from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.checks._drift import evaluator_heading
from dataeval_flow.workflows import Finding

Severity = Literal["ok", "info", "warning"]


class OODThresholds(BaseModel):
    """When an OOD finding is `info` or warns: the percent of the assessed test images flagged."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: float | None = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description=(
            "The percent of assessed test images flagged at which the finding warns; `null` never warns. "
            "ood-detection's `health_thresholds.ood_pct_warning` before its port."
        ),
    )
    info: float | None = Field(
        default=1.0,
        ge=0.0,
        le=100.0,
        description=(
            "The percent at which the finding is `info`, below which it is `ok`; `null` is never `info`. With both "
            "`null`, the finding is `info` and judges nothing. ood-detection's `health_thresholds.ood_pct_info` before "
            "its port."
        ),
    )


def ood_severity(percent: float, thresholds: OODThresholds) -> Severity:
    """The severity `percent` earns: `warning` from `warning`, `info` from `info`, else `ok`. With both `null`,
    `info`, which judges nothing, as `_limits.unjudged` rules for one threshold. "From" is `>=`, legacy's
    comparison."""
    if thresholds.warning is None and thresholds.info is None:
        return "info"
    if thresholds.warning is not None and percent >= thresholds.warning:
        return "warning"
    if thresholds.info is not None and percent >= thresholds.info:
        return "info"
    return "ok"


def assessed_images(output: OODOutput) -> int:
    """How many test images `output` assessed: every one, less those with no detection on detection rows."""
    rows = getattr(output, "rows", None)
    return len(output.is_ood) - (len(rows["unassessed"]) if rows else 0)


class OODCheckConfig(CheckConfig, OODThresholds):
    """An `ood` step's input, its thresholds, and what its finding is titled."""

    input: str = Field(description="An OOD evaluator's Output.")
    subject: str | None = Field(
        default=None,
        description=(
            "What the finding is titled. Unset: the evaluator's title, followed by its entry's name where that "
            "differs from its type, such as `OOD (K-Neighbors) · uncertain`."
        ),
    )


class OODCheck(Check[OODCheckConfig]):
    """``ood``: how many of a test source's images an OOD detector flagged, `info` and warning past its thresholds."""

    name: ClassVar[str] = "ood"
    description: ClassVar[str] = "Judges the share of a test source's images an OOD detector flagged."
    title: ClassVar[str] = "OOD"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(OODOutput,)),)

    def run(self, config: OODCheckConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """One finding: the images flagged of those assessed, and on detection rows the detections flagged."""
        node = inputs["input"]
        output = node.value
        title = config.subject or (evaluator_heading(node.config) if node.config is not None else self.title)
        flagged, assessed = int(np.sum(output.is_ood)), assessed_images(output)
        percent = 100.0 * flagged / assessed if assessed else 0.0
        brief = f"{flagged}/{assessed} images OOD ({percent:.1f}%"
        description = f"{flagged} of {assessed} test images score above the detector's threshold."
        rows = getattr(output, "rows", None)
        if rows:
            detections = rows["detections"]
            hits = sum(1 for row in detections if row["is_ood"])
            brief += f"; {hits:,}/{len(detections):,} detections at confidence ≥ {rows['confidence']}"
            description += " An image is out of distribution when any of its detections is."
        return [
            Finding(severity=ood_severity(percent, config), title=title, brief=brief + ")", description=description)
        ]
