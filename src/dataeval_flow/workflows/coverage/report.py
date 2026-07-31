"""Data coverage workflow report/finding builders."""

from typing import Any, Literal

from dataeval_flow.workflow.base import Reportable
from dataeval_flow.workflows.coverage.outputs import DataCoverageRawOutputs, LabelSpaceCoverage
from dataeval_flow.workflows.coverage.params import DataCoverageHealthThresholds

__all__ = ["build_findings"]


def _finding_coverage(  # noqa: C901
    raw: DataCoverageRawOutputs,
    thresholds: DataCoverageHealthThresholds,
) -> Reportable | None:
    """Embedding-space coverage finding, broken down by class."""
    cov = raw.coverage
    if cov is None:
        if raw.coverage_skipped_reason is None:
            return None
        # An extractor was configured but the assessment could not run — say so
        # rather than letting the section silently vanish from the report.
        return Reportable(
            report_type="key_value",
            severity="info",
            title="Embedding Coverage",
            data={"brief": "skipped"},
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

    rows: list[dict[str, Any]] = [
        {
            "class_name": row.class_name,
            "count": row.count,
            "Dispersion": "-" if row.dispersion is None else round(row.dispersion, 2),
            "Isotropy": "-" if row.isotropy is None else round(row.isotropy, 2),
            "NearDup": "-" if row.near_duplicate_fraction is None else round(row.near_duplicate_fraction, 2),
        }
        for row in cov.per_class
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
    description = f"{cov.uncovered_count} of {observed} {units} uncovered in embedding space."
    if cov.observation_unit != "image":
        description += (
            f" The embedding assessments run on {units} — one per ground-truth box — because "
            "coverage assumes one embedding per label."
        )
    if clustered:
        description += f" Clustered (low dispersion): {', '.join(r.class_name for r in clustered)}."
    if flat:
        description += f" One-dimensional (low isotropy): {', '.join(r.class_name for r in flat)}."
    if padded:
        description += f" Duplicate-padded: {', '.join(r.class_name for r in padded)}."
    if not rate_is_data_driven:
        description += (
            " The uncovered rate is not health-checked: coverage_method='adaptive' flags a "
            "fixed coverage_percent of observations by construction. The per-class columns "
            "are the data-driven signal."
        )

    return Reportable(
        report_type="pivot_table",
        severity=severity,
        title="Embedding Coverage",
        data={
            "brief": brief,
            "table_data": rows,
            "table_headers": ["Class Name", "Count", "Dispersion", "Isotropy", "NearDup"],
            "footer_lines": [
                f"method={cov.method}  radius={round(cov.coverage_radius, 4)}  observations={observed} {units}"
            ],
        },
        description=description,
    )


def _finding_completeness(
    raw: DataCoverageRawOutputs,
    thresholds: DataCoverageHealthThresholds,
) -> Reportable | None:
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

    return Reportable(
        report_type="key_value",
        severity=severity,
        title="Dimensional Completeness",
        data={
            "brief": brief,
            "Completeness Score": score,
            "Nearest Neighbor Pairs": len(comp.nearest_neighbor_pairs),
        },
        description=(f"Dimensional completeness score is {score} (threshold: {thresholds.completeness_score})."),
    )


def _finding_label_distribution(
    raw: DataCoverageRawOutputs,
    thresholds: DataCoverageHealthThresholds,
) -> Reportable:
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

    rows: list[dict[str, Any]] = []
    for cls in sorted(ld.class_distribution, key=lambda c: ld.class_distribution[c], reverse=True):
        count = ld.class_distribution[cls]
        pct = round((count / max(total_labels, 1)) * 100, 1)
        # "Count" and "%" headers are aliased by _render_pivot_table to the
        # lowercase row keys "count"/"pct" — match them or the columns render blank.
        rows.append({"Class": cls, "count": count, "pct": pct})

    brief = f"{ld.num_classes} classes, ratio {ratio}:1"
    if ld.missing_classes:
        brief += f", {len(ld.missing_classes)} with no samples"
    if ld.empty_images:
        brief += f", {len(ld.empty_images)} empty images"

    description = (
        f"{ld.num_classes} classes with imbalance ratio {ratio}:1. {len(ld.empty_images)} images have no labels."
    )
    if total_labels != raw.dataset_size:
        description += f" Percentages are shares of {total_labels} labels across {raw.dataset_size} images."
    if ld.missing_classes:
        description += (
            f" {len(ld.missing_classes)} declared class(es) have zero samples: {', '.join(ld.missing_classes)}."
        )

    return Reportable(
        report_type="pivot_table",
        severity=severity,
        title="Label Distribution",
        data={
            "brief": brief,
            "table_data": rows,
            "table_headers": ["Class", "Count", "%"],
        },
        description=description,
    )


def _finding_metadata_distribution(
    raw: DataCoverageRawOutputs,
) -> Reportable:
    """Metadata distribution finding."""
    md = raw.metadata_distribution
    if not md.metadata_factors:
        return Reportable(
            report_type="key_value",
            severity="info",
            title="Metadata Distribution",
            data={"brief": "No metadata factors available"},
            description="No metadata factors were extracted from the dataset.",
        )

    rows: list[dict[str, Any]] = []
    for factor in md.metadata_factors:
        info = md.metadata_summary.get(factor, {})
        row: dict[str, Any] = {
            "Factor": factor,
            "Type": info.get("type", "unknown"),
        }
        if "unique_values" in info:
            row["Unique"] = info["unique_values"]
        elif info.get("mean") is not None:
            # An all-null column (e.g. a target-level factor read off image-level rows)
            # summarizes to a null mean — show it as absent rather than raising.
            row["Unique"] = f"μ={round(info['mean'], 2)}"
        else:
            row["Unique"] = "-"
        row["Nulls"] = info.get("null_count", 0)
        rows.append(row)

    brief = f"{len(md.metadata_factors)} factors"
    if md.balance_summary:
        brief += ", balance computed"
    if md.diversity_summary:
        brief += ", diversity computed"

    return Reportable(
        report_type="pivot_table",
        severity="info",
        title="Metadata Distribution",
        data={
            "brief": brief,
            "table_data": rows,
            "table_headers": ["Factor", "Type", "Unique", "Nulls"],
        },
        description=f"{len(md.metadata_factors)} metadata factors analyzed.",
    )


def _finding_metadata_gaps(
    raw: DataCoverageRawOutputs,
    thresholds: DataCoverageHealthThresholds,
) -> Reportable | None:
    """Metadata coverage gap finding."""
    gaps = raw.metadata_gaps
    if gaps is None:
        return None

    if not gaps.gaps:
        return Reportable(
            report_type="key_value",
            severity="ok",
            title="Metadata Coverage Gaps",
            data={"brief": "No significant gaps detected"},
            description="No class-factor-value combinations are significantly under-represented.",
        )

    severity: Literal["ok", "info", "warning"] = "warning" if len(gaps.gaps) >= thresholds.gap_count else "info"

    # "Count" is aliased by _render_pivot_table to the lowercase row key "count".
    rows: list[dict[str, Any]] = [
        {
            "Class": gap.class_name,
            "Factor": gap.factor_name,
            "Value": gap.factor_value,
            "count": gap.class_count,
            "Expected": round(gap.expected_count, 1),
            "Deficit": f"{round(gap.deficit * 100, 1)}%",
        }
        for gap in gaps.gaps
    ]

    brief = f"{len(gaps.gaps)} gaps identified"

    return Reportable(
        report_type="pivot_table",
        severity=severity,
        title="Metadata Coverage Gaps",
        data={
            "brief": brief,
            "table_data": rows,
            "table_headers": ["Class", "Factor", "Value", "Count", "Expected", "Deficit"],
        },
        description=(
            f"{len(gaps.gaps)} class-factor-value combinations are under-represented. "
            "These represent gaps in data collection that may affect model performance."
        ),
    )


def _worklist_rows(rep: LabelSpaceCoverage) -> list[dict[str, Any]]:
    """Worklist rows in pivot-table form. 'Count' aliases the lowercase 'count' key."""
    return [
        {
            "Concept": row.label,
            "Action": row.action,
            "count": row.count,
            "Target": row.target,
            "Deficit": row.deficit,
        }
        for row in rep.worklist
    ]


_WORKLIST_HEADERS = ["Concept", "Action", "Count", "Target", "Deficit"]


def _finding_label_space(
    raw: DataCoverageRawOutputs,
    thresholds: DataCoverageHealthThresholds,
) -> Reportable | None:
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
    if rep.dark_branches:
        names = ", ".join(f"{b.label} ({b.leaves} leaves)" for b in rep.dark_branches)
        description += f" Wholly-empty branches: {names}."
    if rep.violations:
        names = ", ".join(f"{v.label} ({v.actual:.1%} < {v.floor:.1%})" for v in rep.violations)
        description += f" Asserted minimum shares not met: {names}."
    if rep.ignored_expected:
        description += (
            f" Ignored ontology_expected entries (they resolve to zero or several concepts): "
            f"{', '.join(rep.ignored_expected)}."
        )

    return Reportable(
        report_type="pivot_table",
        severity=severity,
        title="Label Space Coverage",
        data={
            "brief": brief,
            "table_data": _worklist_rows(rep),
            "table_headers": _WORKLIST_HEADERS,
            "footer_lines": [f"ontology source: {onto.source}"],
        },
        description=description,
    )


def _finding_class_balance(
    raw: DataCoverageRawOutputs,
    thresholds: DataCoverageHealthThresholds,  # noqa: ARG001 - kept for signature symmetry with the loop in build_findings
) -> Reportable | None:
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
    if rep.violations:
        names = ", ".join(f"{v.label} ({v.actual:.1%} < {v.floor:.1%})" for v in rep.violations)
        description += f" Asserted minimum shares not met: {names}."
    if rep.ignored_expected:
        description += f" Ignored ontology_expected entries: {', '.join(rep.ignored_expected)}."

    return Reportable(
        report_type="pivot_table",
        severity=severity,
        title="Class Balance Worklist",
        data={
            "brief": brief,
            "table_data": _worklist_rows(rep),
            "table_headers": _WORKLIST_HEADERS,
        },
        description=description,
    )


def _finding_conformance(
    raw: DataCoverageRawOutputs,
    thresholds: DataCoverageHealthThresholds,
) -> Reportable | None:
    """Do the dataset's class names resolve to ontology concepts?"""
    onto = raw.ontology
    if onto is None or onto.conformance is None:
        return None

    conf = onto.conformance
    severity: Literal["ok", "info", "warning"] = "ok"
    if len(conf.unmatched) > thresholds.unmatched_class_count or conf.ambiguous:
        severity = "warning"

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
            description += f" Unmatched: {', '.join(conf.unmatched)}."
        if conf.ambiguous:
            names = ", ".join(f"{name} -> {len(ids)} concepts" for name, ids in conf.ambiguous.items())
            description += f" Ambiguous: {names}. Disambiguate upstream by passing a concept id."

    return Reportable(
        report_type="key_value",
        severity=severity,
        title="Label Conformance",
        data={
            "brief": brief,
            "Matched": len(conf.matched),
            "Unmatched": len(conf.unmatched),
            "Ambiguous": len(conf.ambiguous),
        },
        description=description,
    )


def _finding_ontology_skipped(raw: DataCoverageRawOutputs) -> Reportable | None:
    """Say why the ontology sections are absent instead of dropping them silently.

    A bad ontology path, or a failure inside the analysis, degrades to a skip reason
    rather than aborting the run — without this the whole ontology half of the report
    would simply vanish with no explanation.
    """
    if raw.ontology is not None or raw.ontology_skipped_reason is None:
        return None

    return Reportable(
        report_type="key_value",
        severity="info",
        title="Ontology Analysis",
        data={"brief": "skipped"},
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


def _finding_ontology_structure(raw: DataCoverageRawOutputs) -> Reportable | None:
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
    if st.label_collisions:
        names = ", ".join(st.label_collisions)
        description += (
            f" {len(st.label_collisions)} name(s) resolve to more than one concept ({names}); "
            "this is what makes reconciliation ambiguous and should be fixed in the ontology."
        )
    if smells and not st.label_collisions:
        description += (
            " The remaining observations are facts, not defects — a truncated ancestry is "
            "expected in a deliberately distributed ontology subset."
        )

    return Reportable(
        report_type="key_value",
        severity=severity,
        title="Ontology Structure",
        data={
            "brief": brief,
            "Concepts": st.concept_count,
            "Leaves": st.leaf_count,
            "Max Depth": st.max_depth,
            "Roots": len(st.roots),
            "Label Collisions": len(st.label_collisions),
        },
        description=description,
    )


def build_findings(
    raw: DataCoverageRawOutputs,
    thresholds: DataCoverageHealthThresholds,
) -> list[Reportable]:
    """Build all findings for the data coverage report."""
    findings: list[Reportable] = []

    # Embedding-based (conditional)
    cov_finding = _finding_coverage(raw, thresholds)
    if cov_finding is not None:
        findings.append(cov_finding)

    comp_finding = _finding_completeness(raw, thresholds)
    if comp_finding is not None:
        findings.append(comp_finding)

    # Always present
    findings.append(_finding_label_distribution(raw, thresholds))
    findings.append(_finding_metadata_distribution(raw))

    # Gap analysis (conditional)
    gap_finding = _finding_metadata_gaps(raw, thresholds)
    if gap_finding is not None:
        findings.append(gap_finding)

    # Ontology (conditional — exactly one of the two worklist findings appears)
    for builder in (_finding_label_space, _finding_class_balance, _finding_conformance):
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
