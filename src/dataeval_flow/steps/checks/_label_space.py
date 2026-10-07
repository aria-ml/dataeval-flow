"""The taxonomy checks: legacy data-coverage's Label Space Coverage (now Leaf Coverage), Label Conformance and
Ontology Structure findings, as steps (coverage spec §3.4)."""

__all__ = [
    "ClassShortfallCheck",
    "ClassShortfallConfig",
    "LabelConformanceCheck",
    "LabelConformanceConfig",
    "LeafCoverageCheck",
    "LeafCoverageConfig",
    "OntologyStructureCheck",
    "OntologyStructureConfig",
    "shortfall_notes",
    "worklist_table",
]

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from dataeval.scope import RepresentationOutput
from pydantic import Field

from dataeval_flow._blocks import Block, Cell, Column, Fields, Paragraph, Table
from dataeval_flow.evaluators.scope import LabelReconciliationOutput, OntologyValidationOutput
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps.checks._limits import Severity
from dataeval_flow.workflows._base import Finding


def worklist_table(rows: Sequence[Mapping[str, Any]]) -> list[Block]:
    """The concepts to acquire or augment, and by how much; nothing when none fall short."""
    if not rows:
        return []
    columns = [
        Column(key="concept", header="Concept"),
        Column(key="action", header="Action"),
        Column(key="count", header="Count"),
        Column(key="target", header="Target"),
        Column(key="deficit", header="Deficit"),
    ]
    cells: list[dict[str, Cell]] = [
        {
            "concept": row["label"],
            "action": row["action"],
            "count": row["count"],
            "target": row["target"],
            "deficit": row["deficit"],
        }
        for row in rows
    ]
    return [Table(columns=columns, rows=cells)]


def shortfall_notes(
    violations: Sequence[Mapping[str, Any]],
    ignored: Sequence[str],
    *,
    why: str = "they resolve to zero or several concepts",
) -> list[str]:
    """The unmet minimum shares and the ignored `expected` entries, with *why* they were ignored, as legacy noted
    them."""
    notes: list[str] = []
    if violations:
        names = ", ".join(f"{v['label']} ({v['actual']:.1%} < {v['floor']:.1%})" for v in violations)
        notes.append(f"Asserted minimum shares not met: {names}.")
    if ignored:
        notes.append(f"Ignored `expected` entries ({why}): {', '.join(ignored)}.")
    return notes


class LeafCoverageConfig(CheckConfig):
    """A `leaf-coverage` step's input, and how much of the ontology's leaves must have examples."""

    input: str = Field(description="A `representation` Output, computed against a declared ontology.")
    coverage: float | None = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description=(
            "The least share of the ontology's leaves that must have examples; under it the finding warns. `null` "
            "turns this criterion off."
        ),
    )
    empty_branches: int | None = Field(
        default=0,
        ge=0,
        description=("Wholly empty branches tolerated; more warn. `null` turns this criterion off."),
    )


class LeafCoverageCheck(Check[LeafCoverageConfig]):
    """``leaf-coverage``: warns when too few of an ontology's leaves have examples, a branch is wholly empty, or an
    asserted minimum share is not met."""

    name: ClassVar[str] = "leaf-coverage"
    description: ClassVar[str] = "Warns when too few of an ontology's leaves have examples, or a branch is empty."
    title: ClassVar[str] = "Leaf Coverage"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(RepresentationOutput,)),)

    def run(self, config: LeafCoverageConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The leaves with examples, the worklist, and the empty branches and unmet shares."""
        value = inputs["input"].value
        worklist = value.data().to_dicts()
        violations = value.violations.to_dicts()
        dark = value.dark_branches.to_dicts()
        ignored = list(getattr(value, "ignored_expected", []))
        coverage = float(value.leaf_coverage)
        deficit = int(value.total_deficit)
        pct = round(coverage * 100, 1)
        acquire = sum(1 for row in worklist if row["action"] == "acquire")
        severity: Severity = "ok"
        if (
            violations
            or (config.coverage is not None and coverage < config.coverage)
            or (config.empty_branches is not None and len(dark) > config.empty_branches)
        ):
            severity = "warning"
        elif worklist:
            severity = "info"
        notes: list[str] = []
        if dark:
            names = ", ".join(f"{b['label']} ({b['leaves']} leaves)" for b in dark)
            notes.append(f"Wholly-empty branches: {names}.")
        notes += shortfall_notes(violations, ignored)
        return [
            Finding(
                severity=severity,
                title=self.title,
                brief=f"leaf coverage {pct}% · {acquire} to acquire · deficit {deficit}",
                description=(
                    f"{pct}% of the ontology's sanctioned leaf species have examples. "
                    f"The dataset is {deficit} labels short of an even spread across them."
                ),
                blocks=[
                    *(Paragraph(text=note) for note in notes),
                    *worklist_table(worklist),
                    Fields(items=[("Ontology source", getattr(value, "ontology_source", None))]),
                ],
            )
        ]


class LabelConformanceConfig(CheckConfig):
    """A `label-conformance` step's input, and how many class names may fail to resolve."""

    input: str = Field(description="A `label-reconciliation` Output.")
    warning: int | None = Field(
        default=0,
        ge=0,
        description=(
            "Class names that may resolve to no concept; more warn. An ambiguous name always warns. `null` turns "
            "the unmatched criterion off."
        ),
    )


class LabelConformanceCheck(Check[LabelConformanceConfig]):
    """``label-conformance``: warns when class names resolve to no ontology concept, or to several."""

    name: ClassVar[str] = "label-conformance"
    description: ClassVar[str] = "Warns when class names resolve to no ontology concept, or to several."
    title: ClassVar[str] = "Label Conformance"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(LabelReconciliationOutput,)),)

    def run(self, config: LabelConformanceConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The matched, unmatched and ambiguous names."""
        data = inputs["input"].value.data()
        matched, unmatched, ambiguous = data["matched"], data["unmatched"], data["ambiguous"]
        warns = (config.warning is not None and len(unmatched) > config.warning) or bool(ambiguous)
        notes: list[str] = []
        if data["conforms"]:
            brief = "conforms"
            description = "Every class name resolves to exactly one ontology concept."
        else:
            brief = f"{len(unmatched)} unmatched, {len(ambiguous)} ambiguous"
            description = (
                f"{len(matched)} class name(s) resolved to a concept. An unmatched name is out of vocabulary — a "
                "typo, or a class the ontology does not sanction."
            )
            if unmatched:
                notes.append(f"Unmatched: {', '.join(unmatched)}.")
            if ambiguous:
                names = ", ".join(f"{name} -> {len(ids)} concepts" for name, ids in ambiguous.items())
                notes.append(f"Ambiguous: {names}. Disambiguate upstream by passing a concept id.")
        return [
            Finding(
                severity="warning" if warns else "ok",
                title=self.title,
                brief=brief,
                description=description,
                blocks=[
                    *(Paragraph(text=note) for note in notes),
                    Fields(
                        items=[("Matched", len(matched)), ("Unmatched", len(unmatched)), ("Ambiguous", len(ambiguous))]
                    ),
                ],
            )
        ]


class OntologyStructureConfig(CheckConfig):
    """An `ontology-structure` step's input. It has no thresholds: only a label collision is a defect."""

    input: str = Field(description="An `ontology-validation` Output.")


def _smells(data: Mapping[str, Any]) -> list[str]:
    """Human-readable counts of the structural observations that are not collisions."""
    smells: list[str] = []
    for key, text in (
        ("isolated", "isolated"),
        ("redundant_edges", "redundant edges"),
        ("ancestor_siblings", "ancestor-sibling pairs"),
        ("unary_parents", "single-child links"),
        ("external_ancestors", "truncated ancestries"),
        ("nonconforming_labels", "nonconforming labels"),
    ):
        if data[key]:
            smells.append(f"{len(data[key])} {text}")
    return smells


class OntologyStructureCheck(Check[OntologyStructureConfig]):
    """``ontology-structure``: reports an ontology's structural facts, and warns on a label several concepts share."""

    name: ClassVar[str] = "ontology-structure"
    description: ClassVar[str] = "Reports an ontology's structure, and warns on a label several concepts share."
    title: ClassVar[str] = "Ontology Structure"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(OntologyValidationOutput,)),)

    def run(self, config: OntologyStructureConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The ontology's size and depth, its smells, and its collisions."""
        data = inputs["input"].value.data()
        collisions = data["label_collisions"]
        smells = _smells(data)
        brief = f"{data['concept_count']} concepts, {data['leaf_count']} leaves, depth {data['max_depth']}"
        if smells:
            brief += " · " + ", ".join(smells)
        notes: list[str] = []
        if collisions:
            notes.append(
                f"{len(collisions)} name(s) resolve to more than one concept ({', '.join(collisions)}); this is what "
                "makes reconciliation ambiguous and should be fixed in the ontology."
            )
        if smells and not collisions:
            notes.append(
                "The remaining observations are facts, not defects — a truncated ancestry is expected in a "
                "deliberately distributed ontology subset."
            )
        return [
            Finding(
                severity="warning" if collisions else "info",
                title=self.title,
                brief=brief,
                description=(
                    f"The ontology has {data['concept_count']} concepts, {data['leaf_count']} of them leaves, reaching "
                    f"depth {data['max_depth']}."
                ),
                blocks=[
                    *(Paragraph(text=note) for note in notes),
                    Fields(
                        items=[
                            ("Concepts", data["concept_count"]),
                            ("Leaves", data["leaf_count"]),
                            ("Max Depth", data["max_depth"]),
                            ("Roots", len(data["roots"])),
                            ("Label Collisions", len(collisions)),
                        ]
                    ),
                ],
            )
        ]


class ClassShortfallConfig(CheckConfig):
    """A `class-shortfall` step's input. It has no thresholds: an unmet minimum share warns, a worklist informs."""

    input: str = Field(description="A `representation` Output, computed with no ontology.")


class ClassShortfallCheck(Check[ClassShortfallConfig]):
    """``class-shortfall``: legacy data-coverage's Class Balance Worklist, the classes short of an even spread over
    the classes the dataset declares (coverage spec §6.2)."""

    name: ClassVar[str] = "class-shortfall"
    description: ClassVar[str] = "Lists the classes short of an even spread, and warns on an unmet minimum share."
    title: ClassVar[str] = "Class Shortfall"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(RepresentationOutput,)),)

    def run(self, config: ClassShortfallConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """The worklist, the unmet shares and the ignored `expected` names."""
        value = inputs["input"].value
        worklist = value.data().to_dicts()
        violations = value.violations.to_dicts()
        deficit = int(value.total_deficit)
        severity: Severity = "warning" if violations else "info" if worklist else "ok"
        notes = shortfall_notes(violations, list(getattr(value, "ignored_expected", [])), why="no class has that name")
        return [
            Finding(
                severity=severity,
                title=self.title,
                brief=f"{len(worklist)} classes short · deficit {deficit}",
                description=(
                    f"{len(worklist)} class(es) fall short of an even spread, by {deficit} labels in total. Targets "
                    "come from a uniform expectation over the classes the dataset itself declares — run a "
                    "`taxonomy` entry with a declared `ontology` to measure coverage of a sanctioned label space "
                    "instead, which is what reveals classes that were never collected at all."
                ),
                blocks=[*(Paragraph(text=note) for note in notes), *worklist_table(worklist)],
            )
        ]
