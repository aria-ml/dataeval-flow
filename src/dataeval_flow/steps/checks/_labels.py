"""The label check: data-cleaning's Label Distribution finding, as a step (spec §9.2)."""

__all__ = [
    "ClassImbalanceCheck",
    "ClassImbalanceConfig",
    "ClassSufficiencyCheck",
    "ClassSufficiencyConfig",
    "UntrainedClassesCheck",
    "UntrainedClassesConfig",
]

from collections import defaultdict
from collections.abc import Mapping
from typing import Any, ClassVar, Self

from pydantic import Field, model_validator

from dataeval_flow._blocks import Block, Cell, Column, Paragraph, Table
from dataeval_flow.evaluators.quality._result import LabelHealthOutput
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._step import StepSkipped
from dataeval_flow.steps.checks._leakage import _nodes
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
            "judges nothing but an empty class, which warns unless `empty` is false. data-cleaning's "
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

    empty: bool = Field(
        default=True,
        description=(
            "Whether a declared class with no labels warns. `false` judges the ratio over the classes with labels "
            "alone, and leaves a class with none to `untrained-classes` and `class-sufficiency`; the class is still "
            "named in the evidence."
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
        empties = [name for name, count in counts.items() if count == 0]
        ratio = round(max(present) / min(present), 1) if present else 0.0
        source = data.get("label_source")
        notes = [f"Labels {render_label_source(source)}"] if source else []
        if empties:
            notes.append(f"Classes with no labels: {', '.join(empties)}")
        if not (empties and config.empty) and ratio != 0.0:
            notes.append(
                "Balanced: all classes have equal counts" if ratio == 1.0 else f"Imbalance ratio: {ratio} (max/min)"
            )
        if labels != items:
            notes.append(f"Percentages are shares of {labels} labels across {items} images.")
        blocks: list[Block] = [*(Paragraph(text=note) for note in notes)]
        if counts:
            blocks.insert(0, _shares(counts))
        on = getattr(node, "computed_on", ())
        unlabelled = list(data.get("empty_image_indices") or [])
        if on and unlabelled:
            blocks += unlabelled_blocks({on[0].address: unlabelled}, header="Source")
        severity: Severity
        if (empties and config.empty) or exceeds(ratio, config.ratio):
            severity = "warning"
        elif not present:
            severity = "info"
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


def _counts(node: Any) -> tuple[str, dict[str, int]]:
    """A `label-health` node's Dataset address and its labels per class, by name."""
    on = getattr(node, "computed_on", ())
    counts = node.value.data()["label_counts_per_class"]
    return (on[0].address if on else node.address), {str(key): int(value) for key, value in counts.items()}


class ClassSufficiencyConfig(CheckConfig):
    """A `class-sufficiency` step's inputs, and the fewest labels a class needs to learn and to evaluate."""

    input: str = Field(description="A `label-health` Output over train.")
    evals: str | None = Field(
        default=None,
        description=(
            "The evaluation splits' `label-health` Outputs, a list that may be empty; unset judges train alone."
        ),
    )
    train: int | None = Field(
        default=20,
        ge=0,
        description="The fewest labels each class train holds needs in train; `null` judges no minimum there.",
    )
    eval: int | None = Field(
        default=30,
        ge=0,
        description=(
            "The fewest labels each class train holds needs in each evaluation split, a class the split lacks "
            "included. At 30, a per-class metric's 95% interval is about ±18 points."
        ),
    )


class ClassSufficiencyCheck(Check[ClassSufficiencyConfig]):
    """``class-sufficiency``: warns when a class train holds has too few labels in train, or in an evaluation split."""

    name: ClassVar[str] = "class-sufficiency"
    description: ClassVar[str] = "Warns when a class has too few labels to learn or to evaluate."
    title: ClassVar[str] = "Class Sufficiency"
    inputs: ClassVar[tuple[Port, ...]] = (
        Port("input", DataType.OUTPUT, classes=(LabelHealthOutput,)),
        Port("evals", DataType.OUTPUT, classes=(LabelHealthOutput,), is_list=True, may_be_empty=True),
    )

    def run(self, config: ClassSufficiencyConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """Each class train holds, counted in train and in each evaluation split, against the two minimums.

        Raises
        ------
        StepSkipped
            When train holds no labelled class: recorded as not assessed.
        """
        train_name, train = _counts(inputs["input"])
        held = {name: count for name, count in train.items() if count > 0}
        if not held:
            raise StepSkipped("train holds no labelled class")
        splits = [_counts(node) for node in _nodes(inputs.get("evals"))]
        thin_train = sorted(name for name, count in held.items() if config.train is not None and count < config.train)
        parts = [f"{len(thin_train)} under {config.train} in {train_name}"] if thin_train else []
        warn = bool(thin_train)
        for split_name, counts in splits:
            thin = [name for name in held if config.eval is not None and counts.get(name, 0) < config.eval]
            if thin:
                warn = True
                parts.append(f"{len(thin)} under {config.eval} in {split_name}")
        sentences = []
        if config.train is not None:
            sentences.append(f"Each class `{train_name}` holds needs at least {config.train} labels there.")
        if config.eval is not None:
            sentences.append(
                f"It needs at least {config.eval} in each evaluation split, where a per-class metric is otherwise "
                "mostly noise."
            )
        sentences.append("Counts are labels: one per box on detection data.")
        columns = [Column(key="name", header="Class", align="left"), Column(key="train", header=train_name)]
        columns += [Column(key=f"s{i}", header=name) for i, (name, _) in enumerate(splits)]
        rows: list[dict[str, Cell]] = [
            {"name": name, "train": count, **{f"s{i}": counts.get(name, 0) for i, (_, counts) in enumerate(splits)}}
            for name, count in sorted(held.items(), key=lambda item: item[1])
        ]
        severity: Severity = "info" if config.train is None and config.eval is None else "warning" if warn else "ok"
        return [
            Finding(
                severity=severity,
                title=self.title,
                brief=", ".join(parts) or f"{len(held)} classes sufficient",
                description=" ".join(sentences),
                blocks=[Table(columns=columns, rows=rows)],
            )
        ]


class UntrainedClassesConfig(CheckConfig):
    """An `untrained-classes` step's inputs, and whether a declared class in no split also warns."""

    input: str = Field(description="A `label-health` Output over train.")
    evals: str | None = Field(
        default=None,
        description="The evaluation splits' `label-health` Outputs, a list that may be empty.",
    )
    declared: bool = Field(
        default=False,
        description=(
            "Whether a declared class with no labels in train also warns. Off, it is listed: a vocabulary conformed "
            "to an ontology carries abstract concepts with no items."
        ),
    )


class UntrainedClassesCheck(Check[UntrainedClassesConfig]):
    """``untrained-classes``: warns when an evaluation split holds a class train has no labels for."""

    name: ClassVar[str] = "untrained-classes"
    description: ClassVar[str] = "Warns when an evaluation split holds a class train lacks."
    title: ClassVar[str] = "Untrained Classes"
    inputs: ClassVar[tuple[Port, ...]] = ClassSufficiencyCheck.inputs

    def run(self, config: UntrainedClassesConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The classes with labels in an evaluation split and none in train, and the declared classes in no split.

        Raises
        ------
        StepSkipped
            When evaluation splits are present but none holds a labelled class: recorded as not assessed.
        """
        train_name, train = _counts(inputs["input"])
        lacking = [name for name, count in train.items() if count == 0]
        splits = [_counts(node) for node in _nodes(inputs.get("evals"))]
        unseen: dict[str, list[str]] = defaultdict(list)
        for split_name, counts in splits:
            for name, count in counts.items():
                if count > 0 and train.get(name, 0) == 0:
                    unseen[name].append(split_name)
        absent = [name for name in lacking if name not in unseen]
        if splits and not any(count > 0 for _, counts in splits for count in counts.values()):
            raise StepSkipped("no evaluation split holds a labelled class")
        severity: Severity
        if not splits and not config.declared:
            return [
                Finding(
                    severity="info",
                    title=self.title,
                    brief="no evaluation split to compare",
                    description="No evaluation split holds labels to compare with train's.",
                )
            ]
        severity = "warning" if unseen or (config.declared and lacking) else "ok"
        n = len(unseen)
        brief = (
            f"{n} class{'es' if n != 1 else ''} in evaluation but not in train"
            if unseen
            else "every evaluated class is in train"
            if splits
            else "no evaluation split"
        )
        if config.declared and absent:
            brief += f", {len(absent)} declared class{'es' if len(absent) != 1 else ''} in no split"
        blocks: list[Block] = []
        if unseen:
            blocks.append(
                Table(
                    columns=[Column(key="name", header="Class", align="left"), Column(key="held", header="Held by")],
                    rows=[{"name": name, "held": ", ".join(held)} for name, held in sorted(unseen.items())],
                )
            )
        if absent:
            text = f"Declared, with no labels in any split: {', '.join(absent)}."
            blocks.append(
                Paragraph(
                    text=text if config.declared else f"{text} They do not warn; set `declared: true` to make them."
                )
            )
        return [
            Finding(
                severity=severity,
                title=self.title,
                brief=brief,
                description=(
                    f"A model never sees the classes `{train_name}` lacks, so an evaluation split holding one "
                    "measures nothing it learned."
                ),
                blocks=blocks,
            )
        ]
