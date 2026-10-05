"""The outlier checks: data-cleaning's Image, Target and Classwise Outliers findings, as steps (spec §9.2)."""

__all__ = [
    "ClasswiseOutliersCheck",
    "ClasswiseOutliersConfig",
    "ImageOutliersCheck",
    "ImageOutliersConfig",
    "TargetOutliersCheck",
    "TargetOutliersConfig",
]

from collections.abc import Mapping
from typing import Any, ClassVar

from dataeval.quality import OutliersOutput
from pydantic import Field

from dataeval_flow._blocks import Cell, Column, Table
from dataeval_flow._classwise import split_outlier_issues
from dataeval_flow.evaluators.quality._result import LabelHealthOutput
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.checks._limits import exceeds, unjudged
from dataeval_flow.steps.combines._classwise import OutliersByClassOutput
from dataeval_flow.workflows._base import Finding


class ImageOutliersConfig(CheckConfig):
    """An `image-outliers` step's input, and the share of images that may be outliers."""

    input: str = Field(description="An `outliers` Output.")
    warning: float | None = Field(
        default=3.0,
        ge=0.0,
        le=100.0,
        description=(
            "Most images, as a percentage of the Dataset, that may be flagged before the finding warns; `null` "
            "judges nothing."
        ),
    )


class ImageOutliersCheck(Check[ImageOutliersConfig]):
    """``image-outliers``: warns when more than ``warning`` percent of a Dataset's images are outliers."""

    name: ClassVar[str] = "image-outliers"
    description: ClassVar[str] = "Warns when more than `warning` percent of a Dataset's images are outliers."
    title: ClassVar[str] = "Image Outliers"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(OutliersOutput,)),)

    def run(self, config: ImageOutliersConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The share of images with at least one image-level flag."""
        node = inputs["input"]
        image_issues, _ = split_outlier_issues(node.value.data())
        count = image_issues["item_index"].n_unique() if len(image_issues) else 0
        size = node.items or 0
        pct = count / size * 100 if size else 0
        brief = f"{count} images ({round(pct, 1)}%)"
        if count == 0:
            severity = unjudged(config.warning, "ok")
            return [Finding(severity=severity, title=self.title, brief=brief)]
        severity = "warning" if exceeds(pct, config.warning) else "info"
        return [Finding(severity=severity, title=self.title, brief=brief)]


class TargetOutliersConfig(CheckConfig):
    """A `target-outliers` step's inputs, and the share of targets that may be outliers."""

    input: str = Field(description="An `outliers` Output computed per target (`per_target: true`).")
    labels: str = Field(
        description="A `label-health` Output on the same Dataset: its label count is the number of targets."
    )
    warning: float | None = Field(
        default=3.0,
        ge=0.0,
        le=100.0,
        description=(
            "Most targets, as a percentage of all, that may be flagged before the finding warns; `null` judges nothing."
        ),
    )


class TargetOutliersCheck(Check[TargetOutliersConfig]):
    """``target-outliers``: warns when more than ``warning`` percent of the boxes are outliers.

    Makes no finding where nothing was flagged per target, as a classification Dataset never is.
    """

    name: ClassVar[str] = "target-outliers"
    description: ClassVar[str] = "Warns when more than `warning` percent of the boxes are outliers."
    title: ClassVar[str] = "Target Outliers"
    inputs: ClassVar[tuple[Port, ...]] = (
        Port("input", DataType.OUTPUT, classes=(OutliersOutput,)),
        Port("labels", DataType.OUTPUT, classes=(LabelHealthOutput,)),
    )

    def run(self, config: TargetOutliersConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The share of labelled targets with at least one flag."""
        _, target_issues = split_outlier_issues(inputs["input"].value.data())
        if target_issues is None or len(target_issues) == 0:
            return []
        count = target_issues.select("item_index", "target_index").n_unique()
        total = int(inputs["labels"].value.data()["label_count"])
        pct = round(count / total * 100, 1) if total > 0 else 0.0
        severity = "warning" if exceeds(pct, config.warning) else "info"
        return [
            Finding(
                severity=severity,
                title=self.title,
                brief=f"{count} targets ({pct}%)",
            )
        ]


class ClasswiseOutliersConfig(CheckConfig):
    """A `classwise-outliers` step's input, and the share of a class that may be outliers."""

    input: str = Field(description="An `outliers-by-class` Output.")
    warning: float | None = Field(
        default=3.0,
        ge=0.0,
        le=100.0,
        description=(
            "Most items or boxes, as a percentage of all, the outliers may take up before the finding warns; each "
            "class is counted against it too. `null` judges nothing."
        ),
    )


class ClasswiseOutliersCheck(Check[ClasswiseOutliersConfig]):
    """``classwise-outliers``: warns when outliers take up more than ``warning`` percent across classes, naming
    the worst class and how many classes pass the limit."""

    name: ClassVar[str] = "classwise-outliers"
    description: ClassVar[str] = "Warns when outliers pass `warning` percent across classes; names the worst class."
    title: ClassVar[str] = "Classwise Outliers"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(OutliersByClassOutput,)),)

    def run(
        self,
        config: ClasswiseOutliersConfig,
        inputs: Mapping[str, Any],
        context: CheckContext,  # noqa: ARG002
    ) -> list[Finding]:
        """The worst class, how many classes pass the limit, and whether the total does."""
        pivot: OutliersByClassOutput = inputs["input"].value
        limit = config.warning
        if not pivot.rows or pivot.total is None:
            return [
                Finding(
                    severity=unjudged(limit, "ok"),
                    title=self.title,
                    brief="no outliers detected",
                )
            ]
        worst = max(pivot.rows, key=lambda row: row.pct)
        named = f"worst: {worst.class_name} ({worst.pct}%)"
        if limit is None:
            brief = named
        else:
            over = sum(row.pct > limit for row in pivot.rows)
            within = f"{over}/{len(pivot.rows)} classes over {limit}%" if over else f"all classes within {limit}%"
            brief = f"{named}, {within}"
        rows: list[dict[str, Cell]] = [
            {"class_name": row.class_name, "count": row.count, "pct": row.pct} for row in [*pivot.rows, pivot.total]
        ]
        columns = [
            Column(key="class_name", header="Class Name"),
            Column(key="count", header="Count"),
            Column(key="pct", header="%", format="{:.1f}%"),
        ]
        return [
            Finding(
                severity="warning" if exceeds(pivot.total.pct, limit) else "info",
                title=self.title,
                brief=brief,
                blocks=[Table(columns=columns, rows=rows)],
            )
        ]
