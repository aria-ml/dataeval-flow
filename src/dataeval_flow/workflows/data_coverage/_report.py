"""Data coverage workflow report/finding builders."""

import json
from typing import Any, Literal

import yaml

from dataeval_flow._blocks import Block, Cell, Code, Column, Fields, ItemRef, Paragraph, Table
from dataeval_flow.workflows._base import Finding
from dataeval_flow.workflows._tables import uncovered_blocks, unlabelled_blocks
from dataeval_flow.workflows.data_coverage._config import DataCoverageHealthThresholds
from dataeval_flow.workflows.data_coverage._outputs import DataCoverageRawOutput, LabelSpaceCoverage, UncoveredItem

__all__ = ["build_findings"]


def _table(columns: list[Column], rows: list[dict[str, Cell]]) -> list[Block]:
    """*rows* under *columns*, or no block at all when there are no rows to show."""
    return [Table(columns=columns, rows=rows)] if rows else []


def _uncovered_blocks(uncovered: list[UncoveredItem], units: str, source: str) -> list[Block]:
    """The uncovered observations, farthest first, each named in *source* as its item or the box it was cut from."""
    return uncovered_blocks(
        [(ItemRef(source=source, index=row.index, target=row.target), row.class_name, row.radius) for row in uncovered],
        units,
    )


def _finding_coverage(  # noqa: C901
    raw: DataCoverageRawOutput,
    thresholds: DataCoverageHealthThresholds,
    source: str,
) -> Finding | None:
    """Embedding-space coverage finding, broken down by class."""
    cov = raw.coverage
    if cov is None:
        if raw.coverage_skipped_reason is None:
            return None
        # An extractor was configured but the assessment could not run — say so
        # rather than letting the section silently vanish from the report.
        return Finding(
            severity="info",
            title="Embedding Coverage",
            brief="skipped",
            description=f"Embedding coverage was skipped: {raw.coverage_skipped_reason}.",
        )

    pct = round(cov.uncovered_rate * 100, 1)
    # coverage_adaptive selects a fixed fraction of observations, so its rate carries
    # no information about sparsity — only threshold a naive rate.
    rate_is_data_driven = cov.method.startswith("naive")

    assessable = [row for row in cov.per_class if row.assessable]
    clustered = [r for r in assessable if r.dispersion is not None and r.dispersion < thresholds.min_dispersion]
    flat = [r for r in assessable if r.isotropy is not None and r.isotropy < thresholds.min_isotropy]
    padded = [
        r
        for r in assessable
        if r.near_duplicate_fraction is not None and r.near_duplicate_fraction > thresholds.max_near_duplicate_fraction
    ]

    severity: Literal["ok", "info", "warning"] = "ok"
    if clustered or flat or padded or rate_is_data_driven and pct > thresholds.uncovered_rate:
        severity = "warning"
    elif cov.uncovered_count > 0:
        severity = "info"

    rows: list[dict[str, Cell]] = [
        {
            "class_name": row.class_name,
            "count": row.count,
            "dispersion": "-" if row.dispersion is None else round(row.dispersion, 2),
            "isotropy": "-" if row.isotropy is None else round(row.isotropy, 2),
            "near_dup": "-" if row.near_duplicate_fraction is None else round(row.near_duplicate_fraction, 2),
        }
        for row in cov.per_class
    ]
    columns = [
        Column(key="class_name", header="Class Name"),
        Column(key="count", header="Count"),
        Column(key="dispersion", header="Dispersion"),
        Column(key="isotropy", header="Isotropy"),
        Column(key="near_dup", header="NearDup"),
    ]

    flags: list[str] = []
    if clustered:
        flags.append(f"{len(clustered)} clustered")
    if flat:
        flags.append(f"{len(flat)} one-dimensional")
    if padded:
        flags.append(f"{len(padded)} duplicate-padded")
    brief = f"{cov.uncovered_count} uncovered ({pct}%)"
    if flags:
        brief += " · " + ", ".join(flags)

    units = f"{cov.observation_unit}s"
    # Results written before observation_count existed carry 0 — fall back to the
    # image count, which is what those runs assessed.
    observed = cov.observation_count or raw.dataset_size
    notes: list[str] = []
    if cov.observation_unit != "image":
        notes.append(
            f"The embedding assessments run on {units} — one per ground-truth box — because "
            "coverage assumes one embedding per label."
        )
    if cov.dropped_detections:
        notes.append(
            f"{cov.dropped_detections} detection(s) were too small or degenerate to embed and "
            "are not covered by these numbers."
        )
    if clustered:
        notes.append(f"Clustered (low dispersion): {', '.join(r.class_name for r in clustered)}.")
    if flat:
        notes.append(f"One-dimensional (low isotropy): {', '.join(r.class_name for r in flat)}.")
    if padded:
        notes.append(f"Duplicate-padded: {', '.join(r.class_name for r in padded)}.")
    if not rate_is_data_driven:
        notes.append(
            "The uncovered rate is not health-checked: coverage_method='adaptive' flags a "
            "fixed coverage_percent of observations by construction. The per-class columns "
            "are the data-driven signal."
        )

    return Finding(
        severity=severity,
        title="Embedding Coverage",
        brief=brief,
        description=f"{cov.uncovered_count} of {observed} {units} uncovered in embedding space.",
        blocks=[
            *(Paragraph(text=note) for note in notes),
            *_table(columns, rows),
            Fields(
                items=[
                    ("Method", cov.method),
                    ("Radius", round(cov.coverage_radius, 4)),
                    ("Observations", f"{observed} {units}"),
                ]
            ),
            *_uncovered_blocks(cov.uncovered, units, source),
        ],
    )


def _finding_completeness(
    raw: DataCoverageRawOutput,
    thresholds: DataCoverageHealthThresholds,
) -> Finding | None:
    """Dimensional completeness finding."""
    comp = raw.completeness
    if comp is None:
        return None

    score = round(comp.completeness_score, 3)
    severity: Literal["ok", "info", "warning"] = "ok"
    if score < thresholds.completeness_score:
        severity = "warning"
    elif score < 0.8:
        severity = "info"

    brief = f"Completeness: {score}"

    return Finding(
        severity=severity,
        title="Dimensional Completeness",
        brief=brief,
        description=(f"Dimensional completeness score is {score} (threshold: {thresholds.completeness_score})."),
        blocks=[
            Fields(
                items=[
                    ("Completeness Score", score),
                    ("Nearest Neighbor Pairs", len(comp.nearest_neighbor_pairs)),
                ]
            )
        ],
    )


def _finding_label_distribution(
    raw: DataCoverageRawOutput,
    thresholds: DataCoverageHealthThresholds,
    source: str,
) -> Finding:
    """Label distribution finding."""
    ld = raw.label_distribution
    # Imbalance is only meaningful across classes that have samples; a class with
    # zero samples is reported separately as a missing class, not as a ratio.
    present = [c for c in ld.class_distribution.values() if c > 0]
    ratio = round(max(present) / min(present), 1) if present else 0.0

    severity: Literal["ok", "info", "warning"] = "ok"
    if ld.missing_classes or (ratio > thresholds.class_imbalance_ratio):
        severity = "warning"
    elif ratio > 2.0:
        severity = "info"

    # Share of the label pool, not of the images: a multi-label or object-detection
    # dataset carries more labels than images, so dividing by the image count would
    # produce column percentages summing well past 100%.
    total_labels = sum(ld.class_distribution.values())

    rows: list[dict[str, Cell]] = []
    for cls in sorted(ld.class_distribution, key=lambda c: ld.class_distribution[c], reverse=True):
        count = ld.class_distribution[cls]
        pct = round((count / max(total_labels, 1)) * 100, 1)
        rows.append({"class": cls, "count": count, "pct": pct})
    columns = [
        Column(key="class", header="Class"),
        Column(key="count", header="Count"),
        Column(key="pct", header="%", format="{:.1f}%"),
    ]

    brief = f"{ld.num_classes} classes, ratio {ratio}:1"
    if ld.missing_classes:
        brief += f", {len(ld.missing_classes)} with no samples"
    if ld.empty_images:
        brief += f", {len(ld.empty_images)} empty images"

    description = (
        f"{ld.num_classes} classes with imbalance ratio {ratio}:1. {len(ld.empty_images)} images have no labels."
    )
    notes: list[str] = []
    if total_labels != raw.dataset_size:
        notes.append(f"Percentages are shares of {total_labels} labels across {raw.dataset_size} images.")
    if ld.missing_classes:
        notes.append(
            f"{len(ld.missing_classes)} declared class(es) have zero samples: {', '.join(ld.missing_classes)}."
        )

    return Finding(
        severity=severity,
        title="Label Distribution",
        brief=brief,
        description=description,
        blocks=[
            *(Paragraph(text=note) for note in notes),
            *_table(columns, rows),
            *unlabelled_blocks({source: ld.empty_images}, header="Source"),
        ],
    )


def _finding_metadata_distribution(
    raw: DataCoverageRawOutput,
) -> Finding:
    """Metadata distribution finding."""
    md = raw.metadata_distribution
    if not md.metadata_factors:
        return Finding(
            severity="info",
            title="Metadata Distribution",
            brief="No metadata factors available",
            description="No metadata factors were extracted from the dataset.",
        )

    rows: list[dict[str, Cell]] = []
    for factor in md.metadata_factors:
        info = md.metadata_summary.get(factor, {})
        row: dict[str, Cell] = {
            "factor": factor,
            "type": info.get("type", "unknown"),
        }
        if "unique_values" in info:
            row["unique"] = info["unique_values"]
        elif info.get("mean") is not None:
            # An all-null column (e.g. a target-level factor read off image-level rows)
            # summarizes to a null mean — show it as absent rather than raising.
            row["unique"] = f"μ={round(info['mean'], 2)}"
        else:
            row["unique"] = "-"
        row["nulls"] = info.get("null_count", 0)
        rows.append(row)
    columns = [
        Column(key="factor", header="Factor"),
        Column(key="type", header="Type"),
        Column(key="unique", header="Unique"),
        Column(key="nulls", header="Nulls"),
    ]

    brief = f"{len(md.metadata_factors)} factors"
    if md.balance_summary:
        brief += ", balance computed"
    if md.diversity_summary:
        brief += ", diversity computed"

    return Finding(
        severity="info",
        title="Metadata Distribution",
        brief=brief,
        description=f"{len(md.metadata_factors)} metadata factors analyzed.",
        blocks=_table(columns, rows),
    )


def _finding_metadata_gaps(
    raw: DataCoverageRawOutput,
    thresholds: DataCoverageHealthThresholds,
) -> Finding | None:
    """Metadata coverage gap finding."""
    gaps = raw.metadata_gaps
    if gaps is None:
        return None

    if not gaps.gaps:
        return Finding(
            severity="ok",
            title="Metadata Coverage Gaps",
            brief="No significant gaps detected",
            description="No class-factor-value combinations are significantly under-represented.",
        )

    severity: Literal["ok", "info", "warning"] = "warning" if len(gaps.gaps) >= thresholds.gap_count else "info"

    rows: list[dict[str, Cell]] = [
        {
            "class": gap.class_name,
            "factor": gap.factor_name,
            "value": gap.factor_value,
            "count": gap.class_count,
            "expected": round(gap.expected_count, 1),
            "deficit": round(gap.deficit * 100, 1),
        }
        for gap in gaps.gaps
    ]
    columns = [
        Column(key="class", header="Class"),
        Column(key="factor", header="Factor"),
        Column(key="value", header="Value"),
        Column(key="count", header="Count"),
        Column(key="expected", header="Expected"),
        Column(key="deficit", header="Deficit", format="{:.1f}%"),
    ]

    brief = f"{len(gaps.gaps)} gaps identified"

    return Finding(
        severity=severity,
        title="Metadata Coverage Gaps",
        brief=brief,
        description=(
            f"{len(gaps.gaps)} class-factor-value combinations are under-represented. "
            "These represent gaps in data collection that may affect model performance."
        ),
        blocks=_table(columns, rows),
    )


def _worklist_table(rep: LabelSpaceCoverage) -> list[Block]:
    """The concepts to acquire or augment, and by how much; nothing when none fall short."""
    columns = [
        Column(key="concept", header="Concept"),
        Column(key="action", header="Action"),
        Column(key="count", header="Count"),
        Column(key="target", header="Target"),
        Column(key="deficit", header="Deficit"),
    ]
    rows: list[dict[str, Cell]] = [
        {"concept": row.label, "action": row.action, "count": row.count, "target": row.target, "deficit": row.deficit}
        for row in rep.worklist
    ]
    return _table(columns, rows)


# Mergeability to the severity it reports at.  A collapse is usually deliberate, so `lossy`
# informs rather than warns; `partial` warns because Relabel will drop a class.
_MERGEABILITY_SEVERITY: dict[str, Literal["ok", "info", "warning"]] = {
    "lossless": "ok",
    "lossy": "info",
    "partial": "warning",
}


def _finding_label_space(
    raw: DataCoverageRawOutput,
    thresholds: DataCoverageHealthThresholds,
) -> Finding | None:
    """Label-space coverage against a configured ontology."""
    onto = raw.ontology
    if onto is None or onto.synthesized:
        return None

    rep = onto.representation
    pct = round(rep.leaf_coverage * 100, 1)
    acquire = sum(1 for row in rep.worklist if row.action == "acquire")

    severity: Literal["ok", "info", "warning"] = "ok"
    if (
        rep.violations
        or rep.leaf_coverage < thresholds.leaf_coverage
        or len(rep.dark_branches) > thresholds.dark_branch_count
    ):
        severity = "warning"
    elif rep.worklist:
        severity = "info"

    brief = f"leaf coverage {pct}% · {acquire} to acquire · deficit {rep.total_deficit}"

    description = (
        f"{pct}% of the ontology's sanctioned leaf species have examples. "
        f"The dataset is {rep.total_deficit} labels short of an even spread across them."
    )
    notes: list[str] = []
    if rep.dark_branches:
        names = ", ".join(f"{b.label} ({b.leaves} leaves)" for b in rep.dark_branches)
        notes.append(f"Wholly-empty branches: {names}.")
    if rep.violations:
        names = ", ".join(f"{v.label} ({v.actual:.1%} < {v.floor:.1%})" for v in rep.violations)
        notes.append(f"Asserted minimum shares not met: {names}.")
    if rep.ignored_expected:
        notes.append(
            f"Ignored ontology_expected entries (they resolve to zero or several concepts): "
            f"{', '.join(rep.ignored_expected)}."
        )

    return Finding(
        severity=severity,
        title="Label Space Coverage",
        brief=brief,
        description=description,
        blocks=[
            *(Paragraph(text=note) for note in notes),
            *_worklist_table(rep),
            Fields(items=[("Ontology source", onto.source)]),
        ],
    )


def _finding_class_balance(
    raw: DataCoverageRawOutput,
    thresholds: DataCoverageHealthThresholds,  # noqa: ARG001 - kept for signature symmetry with the loop in build_findings
) -> Finding | None:
    """Balance worklist against an ontology synthesized from index2label.

    Deliberately titled differently from Label Space Coverage: a synthesized
    ontology can only name classes the dataset already declares, so this measures
    balance, not coverage.
    """
    onto = raw.ontology
    if onto is None or not onto.synthesized:
        return None

    rep = onto.representation
    severity: Literal["ok", "info", "warning"] = "ok"
    if rep.violations:
        severity = "warning"
    elif rep.worklist:
        severity = "info"

    brief = f"{len(rep.worklist)} classes short · deficit {rep.total_deficit}"

    description = (
        f"{len(rep.worklist)} class(es) fall short of an even spread, by {rep.total_deficit} "
        "labels in total. Targets come from a uniform expectation over the classes the dataset "
        "itself declares — configure an `ontology` to measure coverage of a sanctioned label "
        "space instead, which is what reveals classes that were never collected at all."
    )
    notes: list[str] = []
    if rep.violations:
        names = ", ".join(f"{v.label} ({v.actual:.1%} < {v.floor:.1%})" for v in rep.violations)
        notes.append(f"Asserted minimum shares not met: {names}.")
    if rep.ignored_expected:
        notes.append(f"Ignored ontology_expected entries: {', '.join(rep.ignored_expected)}.")

    return Finding(
        severity=severity,
        title="Class Balance Worklist",
        brief=brief,
        description=description,
        blocks=[*(Paragraph(text=note) for note in notes), *_worklist_table(rep)],
    )


def _finding_conformance(
    raw: DataCoverageRawOutput,
    thresholds: DataCoverageHealthThresholds,
) -> Finding | None:
    """Do the dataset's class names resolve to ontology concepts?"""
    onto = raw.ontology
    if onto is None or onto.conformance is None:
        return None

    conf = onto.conformance
    severity: Literal["ok", "info", "warning"] = "ok"
    if len(conf.unmatched) > thresholds.unmatched_class_count or conf.ambiguous:
        severity = "warning"

    notes: list[str] = []
    if conf.conforms:
        brief = "conforms"
        description = "Every class name resolves to exactly one ontology concept."
    else:
        brief = f"{len(conf.unmatched)} unmatched, {len(conf.ambiguous)} ambiguous"
        description = (
            f"{len(conf.matched)} class name(s) resolved to a concept. "
            "An unmatched name is out of vocabulary — a typo, or a class the ontology "
            "does not sanction."
        )
        if conf.unmatched:
            notes.append(f"Unmatched: {', '.join(conf.unmatched)}.")
        if conf.ambiguous:
            names = ", ".join(f"{name} -> {len(ids)} concepts" for name, ids in conf.ambiguous.items())
            notes.append(f"Ambiguous: {names}. Disambiguate upstream by passing a concept id.")

    return Finding(
        severity=severity,
        title="Label Conformance",
        brief=brief,
        description=description,
        blocks=[
            *(Paragraph(text=note) for note in notes),
            Fields(
                items=[
                    ("Matched", len(conf.matched)),
                    ("Unmatched", len(conf.unmatched)),
                    ("Ambiguous", len(conf.ambiguous)),
                ]
            ),
        ],
    )


def _yaml_scalar(value: str) -> str:
    """A YAML scalar for *value*, quoted only where a plain scalar would not round-trip.

    Checked by round-tripping rather than by matching a character set. A label may contain
    a metacharacter, but it may also be a plain word that YAML resolves to a non-string:
    ``0``, ``on``, ``null``, or a date. A config that parses back to an int key never
    matches the class it names, and ``Relabel`` then drops that class silently.
    """
    try:
        safe = yaml.safe_load(f"[{value}]") == [value]
    except yaml.YAMLError:
        safe = False
    return value if safe else json.dumps(value)


def _relabel_stanza(paste_remap: dict[str, str], target_vocabulary: list[str]) -> str:
    """The alignment as a view operation that can be pasted into a config."""
    lines = [
        "      - type: Relabel",
        "        params:",
        "          class_remap:",
    ]
    lines.extend(
        f"            {_yaml_scalar(source)}: {_yaml_scalar(target)}" for source, target in sorted(paste_remap.items())
    )
    targets = ", ".join(_yaml_scalar(t) for t in target_vocabulary)
    lines.append(f"          target: [{targets}]")
    return "\n".join(lines)


def _finding_alignment(
    raw: DataCoverageRawOutput,
    thresholds: DataCoverageHealthThresholds,
) -> Finding | None:
    """What each class name maps to in the reference vocabulary, and what is lost."""
    del thresholds  # severity comes from mergeability, which is not configurable
    onto = raw.ontology
    if onto is None or onto.alignment is None:
        return None

    al = onto.alignment
    severity: Literal["ok", "info", "warning"] = _MERGEABILITY_SEVERITY.get(al.mergeability, "info")
    if al.ambiguous_labels:
        severity = "warning"

    described = {
        "lossless": "Every class carries over one-to-one.",
        "lossy": "Every class carries over, but two or more collapse into a single concept.",
        "partial": "At least one class cannot carry over and is dropped by Relabel.",
    }
    description = f"Mergeability: {al.mergeability}. {described.get(al.mergeability, '')}"

    blocks: list[Block] = []
    if al.unaligned_source:
        blocks.append(Paragraph(text=f"Dropped: {', '.join(al.unaligned_source)}."))
    if al.unaligned_target:
        blocks.append(Paragraph(text=f"Concepts this dataset does not cover: {', '.join(al.unaligned_target)}."))
    if al.ambiguous_labels:
        blocks.append(
            Paragraph(
                text=(
                    f"{len(al.ambiguous_labels)} target label(s) name more than one concept "
                    f"({', '.join(al.ambiguous_labels)}). The stanza below cannot be used until the "
                    "ontology is fixed, because the index such a label takes is undetermined."
                )
            )
        )

    columns = [
        Column(key="source", header="Source"),
        Column(key="relation", header="Relation"),
        Column(key="target", header="Target"),
        Column(key="confidence", header="Confidence"),
        Column(key="matcher", header="Matcher"),
    ]
    rows: list[dict[str, Cell]] = [
        {
            "source": c.source,
            "relation": c.relation,
            "target": c.target_label,
            "confidence": round(c.confidence, 3),
            "matcher": c.matcher,
        }
        for c in al.correspondences
    ]
    blocks.extend(_table(columns, rows))

    if al.paste_remap:
        blocks.extend(
            [
                Paragraph(text="To conform a dataset to this vocabulary, add to its view:"),
                # Printed exactly as given: the leading spaces nest it under a view's `operations:`.
                Code(text=_relabel_stanza(al.paste_remap, al.target_vocabulary), language="yaml"),
                Paragraph(
                    text=(
                        "Datasets merged together must pass the identical `target`, or their integer "
                        "labels denote different classes."
                    )
                ),
            ]
        )
    if al.label_space_digest:
        blocks.append(Fields(items=[("Label space", al.label_space_digest)]))

    return Finding(
        severity=severity,
        title="Label Alignment",
        description=description,
        blocks=blocks,
    )


def _finding_ontology_skipped(raw: DataCoverageRawOutput) -> Finding | None:
    """Say why the ontology sections are absent.

    A bad ontology path, or a failure inside the analysis, degrades to a skip reason
    rather than aborting the run — without this the whole ontology half of the report
    would vanish with no explanation.
    """
    if raw.ontology is not None or raw.ontology_skipped_reason is None:
        return None

    return Finding(
        severity="info",
        title="Ontology Analysis",
        brief="skipped",
        description=f"Ontology analysis was skipped: {raw.ontology_skipped_reason}.",
    )


def _structure_smells(st: Any) -> list[str]:
    """Human-readable counts of non-collision structural observations."""
    smells: list[str] = []
    if st.isolated:
        smells.append(f"{len(st.isolated)} isolated")
    if st.redundant_edges:
        smells.append(f"{len(st.redundant_edges)} redundant edges")
    if st.ancestor_siblings:
        smells.append(f"{len(st.ancestor_siblings)} ancestor-sibling pairs")
    if st.unary_parents:
        smells.append(f"{len(st.unary_parents)} single-child links")
    if st.external_ancestors:
        smells.append(f"{len(st.external_ancestors)} truncated ancestries")
    if st.nonconforming_labels:
        smells.append(f"{len(st.nonconforming_labels)} nonconforming labels")
    return smells


def _finding_ontology_structure(raw: DataCoverageRawOutput) -> Finding | None:
    """Structural facts about the ontology artifact.

    Reports ingredients, not a verdict — whether a finding is a defect is
    contextual. The one exception is a label collision, which is the artifact-side
    cause of reconciliation ambiguity and therefore a genuine defect.
    """
    onto = raw.ontology
    if onto is None or onto.structure is None:
        return None

    st = onto.structure
    severity: Literal["ok", "info", "warning"] = "warning" if st.label_collisions else "info"

    smells = _structure_smells(st)

    brief = f"{st.concept_count} concepts, {st.leaf_count} leaves, depth {st.max_depth}"
    if smells:
        brief += " · " + ", ".join(smells)

    description = (
        f"The ontology has {st.concept_count} concepts, {st.leaf_count} of them leaves, reaching depth {st.max_depth}."
    )
    notes: list[str] = []
    if st.label_collisions:
        names = ", ".join(st.label_collisions)
        notes.append(
            f"{len(st.label_collisions)} name(s) resolve to more than one concept ({names}); "
            "this is what makes reconciliation ambiguous and should be fixed in the ontology."
        )
    if smells and not st.label_collisions:
        notes.append(
            "The remaining observations are facts, not defects — a truncated ancestry is "
            "expected in a deliberately distributed ontology subset."
        )

    return Finding(
        severity=severity,
        title="Ontology Structure",
        brief=brief,
        description=description,
        blocks=[
            *(Paragraph(text=note) for note in notes),
            Fields(
                items=[
                    ("Concepts", st.concept_count),
                    ("Leaves", st.leaf_count),
                    ("Max Depth", st.max_depth),
                    ("Roots", len(st.roots)),
                    ("Label Collisions", len(st.label_collisions)),
                ]
            ),
        ],
    )


def build_findings(
    raw: DataCoverageRawOutput,
    thresholds: DataCoverageHealthThresholds,
    *,
    source: str,
) -> list[Finding]:
    """Build all findings for the data coverage report, naming each item it pictures in *source*."""
    findings: list[Finding] = []

    # Embedding-based (conditional)
    cov_finding = _finding_coverage(raw, thresholds, source)
    if cov_finding is not None:
        findings.append(cov_finding)

    comp_finding = _finding_completeness(raw, thresholds)
    if comp_finding is not None:
        findings.append(comp_finding)

    # Always present
    findings.append(_finding_label_distribution(raw, thresholds, source))
    findings.append(_finding_metadata_distribution(raw))

    # Gap analysis (conditional)
    gap_finding = _finding_metadata_gaps(raw, thresholds)
    if gap_finding is not None:
        findings.append(gap_finding)

    # Ontology (conditional — exactly one of the two worklist findings appears)
    for builder in (_finding_label_space, _finding_class_balance, _finding_conformance, _finding_alignment):
        finding = builder(raw, thresholds)
        if finding is not None:
            findings.append(finding)

    skipped_finding = _finding_ontology_skipped(raw)
    if skipped_finding is not None:
        findings.append(skipped_finding)

    structure_finding = _finding_ontology_structure(raw)
    if structure_finding is not None:
        findings.append(structure_finding)

    return findings
