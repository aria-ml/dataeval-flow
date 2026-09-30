"""The label check: data-cleaning's Label Distribution finding, as a step (spec §9.2)."""

__all__ = ["ClassImbalanceCheck", "ClassImbalanceConfig"]

from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import Field

from dataeval_flow._blocks import Block, Paragraph
from dataeval_flow.evaluators.bias._report import ranked_table
from dataeval_flow.evaluators.quality._result import LabelHealthOutput
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.checks._limits import exceeds
from dataeval_flow.workflows._base import Finding, render_label_source


class ClassImbalanceConfig(CheckConfig):
    """A `class-imbalance` step's input, and how uneven its classes may be."""

    input: str = Field(description="A `label-health` Output.")
    ratio: float | None = Field(
        default=5.0,
        ge=1.0,
        description=(
            "Largest class count over smallest that may hold before the finding warns; `null` judges nothing but "
            "an empty class, which always warns. data-cleaning's `health_thresholds.class_label_imbalance`."
        ),
    )


class ClassImbalanceCheck(Check[ClassImbalanceConfig]):
    """``class-imbalance``: warns when the largest class outnumbers the smallest by more than ``ratio``, or when a
    class has no labels. Makes no finding where no item has a label, or the Dataset declares no class."""

    name: ClassVar[str] = "class-imbalance"
    description: ClassVar[str] = "Warns when the largest class outnumbers the smallest by more than `ratio`."
    title: ClassVar[str] = "Label Distribution"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(LabelHealthOutput,)),)

    def run(self, config: ClassImbalanceConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The ratio of the largest class count to the smallest, with the counts ranked as its evidence."""
        data = inputs["input"].value.data()
        classes, items = int(data["class_count"]), int(data["item_count"])
        if not data["label_count"] or not classes:
            return []
        counts: dict[str, int] = dict(data["label_counts_per_class"])
        empty = bool(counts) and min(counts.values()) == 0
        ratio = round(max(counts.values()) / min(counts.values()), 1) if counts and not empty else 0.0
        source = data.get("label_source")
        notes = [f"Labels {render_label_source(source)}"] if source else []
        if empty:
            notes.append("Warning: one or more classes have zero items")
        elif ratio == 1.0:
            notes.append("Balanced: all classes have equal counts")
        elif ratio != 0.0:
            notes.append(f"Imbalance ratio: {ratio} (max/min)")
        blocks: list[Block] = []
        if counts:
            blocks = [ranked_table(counts, headers=("Class", "Count")), *(Paragraph(text=note) for note in notes)]
        return [
            Finding(
                severity="warning" if empty or exceeds(ratio, config.ratio) else "info",
                title="Label/Directory_Name Distribution" if source == "filepath" else self.title,
                brief=f"{classes} classes, {items} items, imbalance {ratio}:1",
                description=f"{classes} classes, {items} items.",
                blocks=blocks,
            )
        ]
