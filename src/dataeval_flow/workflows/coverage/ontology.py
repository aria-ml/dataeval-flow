"""Project a dataset's labels onto an ontology and lint the ontology itself.

Wraps four DataEval calls behind one pydantic result:

- :class:`dataeval.scope.Representation` — the collection worklist
- :func:`dataeval.core.label_reconciliation` — do the class names resolve?
- :func:`dataeval.core.label_alignment` — what does each class name map to?
- :func:`dataeval.core.ontology_validation` — is the artifact itself sound?

The last three are skipped for a synthesized ontology, where they describe the ontology's
own construction rather than the data.
"""

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from dataeval_flow.workflows.coverage.outputs import (
    AlignmentCorrespondence,
    DarkBranch,
    LabelAlignment,
    LabelConformance,
    LabelSpaceCoverage,
    OntologyAssessment,
    OntologyStructure,
    RepresentationRow,
    RepresentationViolation,
)

if TYPE_CHECKING:
    from dataeval import Ontology

__all__ = ["run_ontology_analysis"]

_logger = logging.getLogger(__name__)


def _representation(
    ontology: "Ontology",
    class_counts: Mapping[str, int],
    expected: Mapping[str, float] | None,
) -> LabelSpaceCoverage:
    """Run the Representation evaluator and flatten its frames into pydantic models."""
    from dataeval.scope import Representation

    result = Representation(ontology, expected=dict(expected) if expected else None).evaluate(dict(class_counts))

    # Representation drops expected names that do not resolve to exactly one concept,
    # with only a log warning. Recompute which ones so a silently-ignored assertion
    # stays visible to the caller.
    ignored = sorted(name for name in (expected or {}) if len(ontology.find(name)) != 1)

    return LabelSpaceCoverage(
        leaf_coverage=float(result.leaf_coverage),
        total_deficit=int(result.total_deficit),
        worklist=[RepresentationRow(**row) for row in result.data().to_dicts()],
        dark_branches=[DarkBranch(**row) for row in result.dark_branches.to_dicts()],
        violations=[RepresentationViolation(**row) for row in result.violations.to_dicts()],
        ignored_expected=ignored,
    )


def _conformance(ontology: "Ontology", class_names: "list[str]") -> LabelConformance:
    """Reconcile the dataset's class names against the ontology."""
    from dataeval.core import label_reconciliation

    result = label_reconciliation(class_names, ontology)
    unmatched = list(result["unmatched"])
    ambiguous = {name: list(ids) for name, ids in result["ambiguous"].items()}
    return LabelConformance(
        conforms=not unmatched and not ambiguous,
        matched=dict(result["matched"]),
        unmatched=unmatched,
        ambiguous=ambiguous,
    )


def _alignment(ontology: "Ontology", class_names: "list[str]") -> LabelAlignment:
    """Align the dataset's vocabulary against the ontology and render it for a config.

    Two forms of the rewrite are returned. ``class_remap`` holds the concept ids DataEval
    produced, which are IRIs for an RDF artifact. ``paste_remap`` resolves those ids to
    labels, because :class:`dataeval.data.Relabel` reads its values as ids only when
    ``target`` is an :class:`~dataeval.Ontology` object, and the list-valued ``target`` a
    YAML config can express takes labels. The two are identical for a hand-built ontology,
    where ids are labels.
    """
    from dataeval.core import label_alignment

    from dataeval_flow.label_space import label_space_digest, ontology_digest

    result = label_alignment(class_names, ontology)

    labels = {cid: ontology.concept(cid).label for cid in ontology.ids}
    vocabulary = [labels[cid] for cid in ontology.ids]

    # A label naming two concepts has no determined index in a list-valued target, so the
    # emitted stanza cannot be used until the ontology is fixed. Reported rather than
    # suppressed; `ontology_validation` reports the same collisions from the artifact side.
    counts: dict[str, int] = {}
    for label in vocabulary:
        counts[label] = counts.get(label, 0) + 1
    ambiguous = sorted(name for name, n in counts.items() if n > 1)

    class_remap = dict(result["class_remap"])
    paste_remap = {source: labels.get(target, target) for source, target in class_remap.items()}

    return LabelAlignment(
        mergeability=result["mergeability"],
        correspondences=[
            AlignmentCorrespondence(
                source=c.source,
                relation=c.relation,
                target=c.target,
                target_label=labels.get(c.target, c.target),
                confidence=c.confidence,
                matcher=c.matcher,
            )
            for c in result["correspondences"]
        ],
        unaligned_source=list(result["unaligned_source"]),
        unaligned_target=[labels.get(t, t) for t in result["unaligned_target"]],
        class_remap=class_remap,
        paste_remap=paste_remap,
        target_vocabulary=vocabulary,
        ambiguous_labels=ambiguous,
        label_space_digest=label_space_digest(
            ontology=ontology_digest(ontology.ids),
            class_remap=paste_remap,
            target=vocabulary,
        ),
    )


def _structure(ontology: "Ontology", label_pattern: str | None) -> OntologyStructure:
    """Lint the ontology artifact and flatten the result into JSON-safe types."""
    from dataeval.core import ontology_validation

    result = ontology_validation(ontology, label_pattern=label_pattern)
    depths = result["depth"]
    return OntologyStructure(
        concept_count=len(ontology.ids),
        leaf_count=len(result["leaves"]),
        max_depth=max(depths.values()) if depths else 0,
        roots=list(result["roots"]),
        isolated=list(result["isolated"]),
        external_ancestors={cid: list(ids) for cid, ids in result["external_ancestors"].items()},
        # DataEval returns tuples; JSON has no tuple, so store 2-element lists.
        redundant_edges=[list(edge) for edge in result["redundant_edges"]],
        ancestor_siblings=[list(pair) for pair in result["ancestor_siblings"]],
        unary_parents=list(result["unary_parents"]),
        label_collisions={name: list(ids) for name, ids in result["label_collisions"].items()},
        nonconforming_labels=dict(result["nonconforming_labels"]),
    )


def run_ontology_analysis(
    ontology: "Ontology",
    *,
    source: str,
    synthesized: bool,
    class_counts: Mapping[str, int],
    expected: Mapping[str, float] | None = None,
    label_pattern: str | None = None,
) -> OntologyAssessment:
    """Assess a dataset's labels against an ontology.

    Parameters
    ----------
    ontology : Ontology
        The sanctioned label space.
    source : str
        Where the ontology came from — ``"inline"``, a path, or ``"index2label"``.
    synthesized : bool
        True when the ontology was built from the dataset's own ``index2label``.
        Conformance, alignment, and structure are then skipped: the class names came
        from the ontology, so they always reconcile and align trivially, and a flat
        graph has no structure to lint.
    class_counts : Mapping[str, int]
        Class name to sample count.
    expected : Mapping[str, float] or None
        Class name to its asserted minimum share of the dataset.
    label_pattern : str or None
        Regex that concept labels must match, for the naming lint.

    Returns
    -------
    OntologyAssessment
    """
    _logger.info("  Running ontology analysis (source=%s) ...", source)
    representation = _representation(ontology, class_counts, expected)

    if synthesized:
        return OntologyAssessment(
            source=source,
            synthesized=True,
            representation=representation,
        )

    return OntologyAssessment(
        source=source,
        synthesized=False,
        representation=representation,
        conformance=_conformance(ontology, list(class_counts)),
        alignment=_alignment(ontology, list(class_counts)),
        structure=_structure(ontology, label_pattern),
    )
