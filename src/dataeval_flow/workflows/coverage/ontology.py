"""Project a dataset's labels onto an ontology and lint the ontology itself.

Wraps three DataEval calls behind one pydantic result:

- :class:`dataeval.scope.Representation` — the collection worklist
- :func:`dataeval.core.label_reconciliation` — do the class names resolve?
- :func:`dataeval.core.ontology_validation` — is the artifact itself sound?

The last two are skipped for a synthesized ontology, where both answer questions
about their own construction rather than about the data.
"""

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from dataeval_flow.workflows.coverage.outputs import (
    DarkBranch,
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
        Conformance and structure are then skipped: the class names came from the
        ontology, so they always reconcile, and a flat graph has no structure to lint.
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
        structure=_structure(ontology, label_pattern),
    )
