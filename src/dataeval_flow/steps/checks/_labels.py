"""The label check: data-cleaning's Label Distribution finding, as a step (spec §9.2)."""

__all__ = ["ClassImbalanceCheck", "ClassImbalanceConfig"]

from collections.abc import Mapping
from typing import Any, ClassVar, Self

from pydantic import Field, model_validator

from dataeval_flow._blocks import Block, Cell, Column, Paragraph, Table
from dataeval_flow.evaluators.quality._result import LabelHealthOutput
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.checks._limits import Severity, exceeds
from dataeval_flow.workflows._base import Finding, render_label_source
from dataeval_flow.workflows._tables import unlabelled_blocks


class ClassImbalanceConfig(CheckConfig):
    """A `class-imbalance` step's input, how uneven its classes may be, and the band under which they are ok."""

    input: str = Field(description="A `label-health` Output.")
    ratio: float | None = Field(
        default=5.0,
        ge=1.0,
        description=(
            "Largest class count over smallest, among the classes with labels, past which the finding warns; `null` "
            "judges nothing but an empty class, which always warns. data-cleaning's "
            "`health_thresholds.class_label_imbalance`."
        ),
    )
    info: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "A ratio at or under which the finding is ok, between which and `ratio` it informs; `null` makes every "
            "ratio under `ratio` information. Must not exceed `ratio`."
        ),
    )

    @model_validator(mode="after")
    def _info_under_ratio(self) -> Self:
        if self.info is not None and self.ratio is not None and self.info > self.ratio:
            raise ValueError(f"`info` ({self.info}) must not exceed `ratio` ({self.ratio}).")
        return self


class ClassImbalanceCheck(Check[ClassImbalanceConfig]):
    """``class-imbalance``: warns when the largest class outnumbers the smallest by more than ``ratio``, among the
    classes with labels, or when a class has none. Makes a finding whenever the Dataset has classes, declared or
    observed (coverage spec §5.3)."""

    name: ClassVar[str] = "class-imbalance"
    description: ClassVar[str] = "Warns when the largest class outnumbers the smallest by more than `ratio`."
    title: ClassVar[str] = "Label Distribution"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(LabelHealthOutput,)),)

    def run(self, config: ClassImbalanceConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The ratio over the classes with labels, the empty classes named, the counts and shares ranked, and the
        images with no labels."""
        node = inputs["input"]
        data = node.value.data()
        counts: dict[str, int] = dict(data["label_counts_per_class"])
        items, labels = int(data["item_count"]), int(data["label_count"])
        classes = int(data["class_count"]) or len(counts)
        if not classes:
            return []
        present = [count for count in counts.values() if count > 0]
        empty = [name for name, count in counts.items() if count == 0]
        ratio = round(max(present) / min(present), 1) if present else 0.0
        source = data.get("label_source")
        notes = [f"Labels {render_label_source(source)}"] if source else []
        if empty:
            notes.append(f"Classes with no labels: {', '.join(empty)}")
        elif ratio == 1.0:
            notes.append("Balanced: all classes have equal counts")
        elif ratio != 0.0:
            notes.append(f"Imbalance ratio: {ratio} (max/min)")
        if labels != items:
            notes.append(f"Percentages are shares of {labels} labels across {items} images.")
        blocks: list[Block] = [*(Paragraph(text=note) for note in notes)]
        if counts:
            blocks.insert(0, _shares(counts))
        on = getattr(node, "computed_on", ())
        empties = list(data.get("empty_image_indices") or [])
        if on and empties:
            blocks += unlabelled_blocks({on[0].address: empties}, header="Source")
        severity: Severity
        if empty or exceeds(ratio, config.ratio):
            severity = "warning"
        elif config.info is not None and not exceeds(ratio, config.info):
            severity = "ok"
        else:
            severity = "info"
        return [
            Finding(
                severity=severity,
                title="Label/Directory_Name Distribution" if source == "filepath" else self.title,
                brief=f"{classes} classes, {items} items, imbalance {ratio}:1",
                blocks=blocks,
            )
        ]


def _shares(counts: Mapping[str, int]) -> Table:
    """Each class's count and share of the labels, largest first, with a bar."""
    total = sum(counts.values())
    rows: list[dict[str, Cell]] = [
        {"name": name, "value": count, "share": count / total * 100 if total else 0.0}
        for name, count in sorted(counts.items(), key=lambda item: -item[1])
    ]
    return Table(
        columns=[
            Column(key="name", header="Class"),
            Column(key="value", header="Count"),
            Column(key="share", header="Share", format="{:.1f}%"),
            Column(key="value", kind="bar"),
        ],
        rows=rows,
    )
