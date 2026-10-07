"""The `uncovered-items` check: how much of a Dataset coverage left uncovered (data-splitting spec §6.2)."""

__all__ = [
    "ClassCoverageCheck",
    "ClassCoverageConfig",
    "DimensionalCompletenessCheck",
    "DimensionalCompletenessConfig",
    "UncoveredItemsCheck",
    "UncoveredItemsConfig",
]

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self

from dataeval.scope import CoverageOutput
from pydantic import Field, model_validator

from dataeval_flow._blocks import Fields, Paragraph
from dataeval_flow.evaluators.scope import CompletenessOutput
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.checks._limits import Severity, exceeds
from dataeval_flow.workflows._base import Finding


class UncoveredItemsConfig(CheckConfig):
    """An `uncovered-items` step's input, and the share uncovered that may hold."""

    input: str = Field(description="A `coverage` Output.")
    warning: float | None = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description=(
            "The percent of the Dataset's items uncovered past which the finding warns; `null` judges nothing. Judge "
            "only `naive` coverage: adaptive coverage marks `percent` of the items uncovered by construction."
        ),
    )


class UncoveredItemsCheck(Check[UncoveredItemsConfig]):
    """``uncovered-items``: warns when more than ``warning`` percent of a Dataset's items are uncovered."""

    name: ClassVar[str] = "uncovered-items"
    description: ClassVar[str] = (
        "Judges `coverage`'s output: warns when more than `warning` percent of a Dataset's items are uncovered."
    )
    title: ClassVar[str] = "Uncovered Items"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(CoverageOutput,)),)

    def run(self, config: UncoveredItemsConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """How many items were uncovered, of how many."""
        node = inputs["input"]
        total = node.items or 0
        uncovered = len(node.value.uncovered_indices)
        percent = uncovered / total * 100 if total else 0.0
        return [
            Finding(
                severity="warning" if exceeds(percent, config.warning) else "info",
                title=self.title,
                brief=f"{uncovered} of {total} uncovered ({round(percent, 1)}%)",
            )
        ]


class DimensionalCompletenessConfig(CheckConfig):
    """A `dimensional-completeness` step's input, and the bands of its score."""

    input: str = Field(description="A `completeness` Output.")
    warning: float | None = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="The score under which the finding warns; `null` turns it off.",
    )
    info: float | None = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="The score under which the finding informs; `null` turns it off.",
    )

    @model_validator(mode="after")
    def _warning_under_info(self) -> Self:
        if self.warning is not None and self.info is not None and self.warning > self.info:
            raise ValueError(f"`warning` ({self.warning}) must not exceed `info` ({self.info}).")
        return self


class DimensionalCompletenessCheck(Check[DimensionalCompletenessConfig]):
    """``dimensional-completeness``: legacy data-coverage's Dimensional Completeness finding (coverage spec §6.2)."""

    name: ClassVar[str] = "dimensional-completeness"
    description: ClassVar[str] = (
        "Judges `completeness`'s output: warns when the embeddings fill too little of their space's dimensions."
    )
    title: ClassVar[str] = "Dimensional Completeness"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(CompletenessOutput,)),)

    def run(
        self,
        config: DimensionalCompletenessConfig,
        inputs: Mapping[str, Any],
        context: CheckContext,  # noqa: ARG002
    ) -> list[Finding]:
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


class ClassCoverageConfig(CheckConfig):
    """A `class-coverage` step's input, and how clustered, flat or padded an assessable class may be."""

    input: str = Field(description="A `coverage` Output.")
    dispersion: float | None = Field(
        default=0.5,
        ge=0.0,
        description="A class's dispersion under which it is clustered; `null` turns it off.",
    )
    isotropy: float | None = Field(
        default=0.5,
        ge=0.0,
        description="A class's isotropy under which it is one-dimensional; `null` turns it off.",
    )
    near_duplicates: float | None = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=("A class's near-duplicate share over which it is duplicate-padded; `null` turns it off."),
    )


class ClassCoverageCheck(Check[ClassCoverageConfig]):
    """``class-coverage``: legacy data-coverage's Embedding Coverage finding. Warns where an assessable class is
    clustered, one-dimensional or duplicate-padded; informs where any item is uncovered. The uncovered rate itself is
    `uncovered-items`'s (coverage spec §6.2)."""

    name: ClassVar[str] = "class-coverage"
    description: ClassVar[str] = (
        "Judges `coverage`'s output: warns when a class is clustered, one-dimensional or padded with near-duplicates."
    )
    title: ClassVar[str] = "Class Coverage"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(CoverageOutput,)),)

    def run(self, config: ClassCoverageConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The uncovered share, each flagged class, and, on detection crops, legacy's notes on them."""
        from dataeval.data import DetectionCrops

        node = inputs["input"]
        rows = [row for row in node.value.data().to_dicts() if row["assessable"]]
        clustered = _flagged(rows, "dispersion", config.dispersion, under=True)
        flat = _flagged(rows, "isotropy", config.isotropy, under=True)
        padded = _flagged(rows, "near_duplicate_fraction", config.near_duplicates, under=False)
        uncovered = len(node.value.uncovered_indices)
        total = node.items or 0
        pct = round(uncovered / total * 100, 1) if total else 0.0
        crops = [on.value for on in getattr(node, "computed_on", ()) if isinstance(on.value, DetectionCrops)]
        units = "detection crops" if crops else "images"
        if clustered or flat or padded:
            severity: Severity = "warning"
        elif uncovered:
            severity = "info"
        else:
            severity = "ok"
        flags = [
            f"{len(names)} {word}"
            for names, word in ((clustered, "clustered"), (flat, "one-dimensional"), (padded, "duplicate-padded"))
            if names
        ]
        brief = f"{uncovered} uncovered ({pct}%)" + (" · " + ", ".join(flags) if flags else "")
        notes: list[str] = []
        if crops:
            notes.append(
                f"The embedding assessments run on {units} — one per ground-truth box — because "
                "coverage assumes one embedding per label."
            )
            dropped = sum(int(crop.n_dropped) for crop in crops)
            if dropped:
                notes.append(
                    f"{dropped} detection(s) were too small or degenerate to embed and "
                    "are not covered by these numbers."
                )
        if clustered:
            notes.append(f"Clustered (low dispersion): {', '.join(clustered)}.")
        if flat:
            notes.append(f"One-dimensional (low isotropy): {', '.join(flat)}.")
        if padded:
            notes.append(f"Duplicate-padded: {', '.join(padded)}.")
        return [
            Finding(
                severity=severity,
                title=self.title,
                brief=brief,
                description=f"{uncovered} of {total} {units} uncovered in embedding space.",
                blocks=[Paragraph(text=note) for note in notes],
            )
        ]


def _flagged(rows: Sequence[Mapping[str, Any]], column: str, limit: float | None, *, under: bool) -> list[str]:
    """The classes whose `column` is under (or over) `limit`; none where the limit is ``None``."""
    if limit is None:
        return []
    return [
        str(row["class"])
        for row in rows
        if row[column] is not None and (row[column] < limit if under else row[column] > limit)
    ]
