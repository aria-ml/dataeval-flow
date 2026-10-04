"""The `ood` check: how much of a test source an OOD detector flagged (ood-detection spec §5.1)."""

__all__ = [
    "EvalCoverageCheck",
    "EvalCoverageConfig",
    "OODAgreementCheck",
    "OODAgreementConfig",
    "OODCheck",
    "OODCheckConfig",
    "OODThresholds",
    "assessed_images",
    "ood_severity",
]

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import numpy as np
from dataeval.shift import OODOutput
from pydantic import BaseModel, ConfigDict, Field

from dataeval_flow._blocks import Paragraph
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.checks._drift import evaluator_heading
from dataeval_flow.steps.combines._ood import OODUnion
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


class OODAgreementConfig(CheckConfig, OODThresholds):
    """An `ood-agreement` step's input and thresholds."""

    input: str = Field(description="An `ood-union` Output.")


class OODAgreementCheck(Check[OODAgreementConfig]):
    """``ood-agreement``: how many images every OOD detector flagged, judged as `ood` judges, and the images one
    detector alone flagged."""

    name: ClassVar[str] = "ood-agreement"
    description: ClassVar[str] = (
        "Judges the share of a test source's images every OOD detector flagged, and counts those one alone flagged."
    )
    title: ClassVar[str] = "OOD Agreement"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(OODUnion,)),)

    def run(self, config: OODAgreementConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The aggregate finding, and the unique one where any image is unique."""
        union = inputs["input"].value
        mutual = len(union.mutual)
        percent = 100.0 * mutual / union.assessed if union.assessed else 0.0
        findings = [
            Finding(
                severity=ood_severity(percent, config),
                title="Aggregate OOD (all detectors agree)",
                brief=f"{mutual}/{len(union.union)} OOD images agreed by all detectors ({percent:.1f}%)",
                description=(
                    "Ranked most out of distribution first. A score is a multiple of the detector's threshold, "
                    "averaged over the detectors."
                ),
            )
        ]
        unique = sum(len(indices) for indices in union.unique.values())
        if unique:
            partial = "; images some but not all flagged are listed as partial" if union.partial else ""
            findings.append(
                Finding(
                    severity="info",
                    title="Unique OOD Samples (single-detector only)",
                    brief=f"{unique} image(s) flagged by only one detector",
                    description=f"Images one detector flagged and the others did not{partial}.",
                )
            )
        return findings


class EvalCoverageConfig(CheckConfig, OODThresholds):
    """An `eval-coverage` step's input, and the share of an evaluation split that may lie beyond train."""

    input: str = Field(description="An `ood-kneighbors` Output fitted on train and run on one evaluation split.")
    info: float | None = Field(
        default=2.0,
        ge=0.0,
        le=100.0,
        description=(
            "The percent flagged at which the finding is `info`, below which it is `ok`; `null` is never `info`. A "
            "split drawn like train has about 100 - `threshold_perc` percent flagged by construction, so `2.0` "
            "suits `threshold_perc: 99`."
        ),
    )


class EvalCoverageCheck(Check[EvalCoverageConfig]):
    """``eval-coverage``: how much of an evaluation split lies farther from train than most of train does."""

    name: ClassVar[str] = "eval-coverage"
    description: ClassVar[str] = "Warns when much of an evaluation split lies beyond what train covers."
    title: ClassVar[str] = "Evaluation Coverage"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(OODOutput,)),)

    def run(self, config: EvalCoverageConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The share flagged, against the percentile the evaluator flagged at."""
        node = inputs["input"]
        output = node.value
        on = getattr(node, "computed_on", ())
        reference, split = (on[0].address, on[-1].address) if on else ("train", "the evaluation split")
        perc = getattr(node.config, "threshold_perc", None)
        perc = 95.0 if perc is None else float(perc)  # DataEval's default, which OODOutput doesn't record
        flagged, assessed = int(np.sum(output.is_ood)), assessed_images(output)
        percent = 100.0 * flagged / assessed if assessed else 0.0
        return [
            Finding(
                severity=ood_severity(percent, config),
                title=self.title,
                brief=f"{flagged}/{assessed} farther from {reference} than {perc:g}% of it ({percent:.1f}%)",
                description=(
                    f"{flagged} of {assessed} items in `{split}` lie farther from `{reference}` than {perc:g}% of "
                    f"`{reference}` lies from itself."
                ),
                blocks=[
                    Paragraph(
                        text=f"A split drawn like `{reference}` has about {100 - perc:g}% flagged by construction."
                    )
                ],
            )
        ]
