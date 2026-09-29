"""How a dataset's vocabulary aligns to a reference ontology (spec §6.2, §6.4).

`data-coverage`'s ontology analysis and the `scope.label-alignment` evaluator both align a dataset's class names
against an ontology; this is the one place that calls DataEval's ``label_alignment`` and shapes the result, so the
two report identically. ``LabelAlignmentOutput`` sits beside ``LabelAlignment`` so the evaluator's config and
implementation modules can both import it without a cycle.
"""

__all__ = [
    "AlignmentCorrespondence",
    "LabelAlignment",
    "LabelAlignmentOutput",
    "align_labels",
    "effective_mergeability",
]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from dataeval_flow.evaluators._core import CoreOutput

if TYPE_CHECKING:
    from dataeval import Ontology


class AlignmentCorrespondence(BaseModel):
    """One typed mapping from a source class to a target concept."""

    source: str = Field(description="Source class name")
    relation: Literal["equivalent", "narrower", "broader", "related"] = Field(
        description=(
            "How the source relates to the target. 'equivalent' is a rename and 'narrower' is a "
            "coarsening up the hierarchy; both carry over into class_remap. 'broader' and "
            "'related' are reported as diagnostics and do not carry over."
        )
    )
    target: str = Field(description="Target concept id")
    target_label: str = Field(description="Human-readable label of the target concept")
    confidence: float = Field(
        description="Strength in [0, 1]. Exact and structurally entailed correspondences are 1.0."
    )
    matcher: str = Field(description="Matcher that produced it: 'exact', 'structural', or a custom name")


class LabelAlignment(BaseModel):
    """How a dataset's vocabulary maps onto the reference ontology.

    The general form of :class:`LabelConformance`. Conformance reports whether every class
    name resolves; alignment reports what each name maps to and what is lost in the mapping.
    """

    mergeability: Literal["lossless", "lossy", "partial"] = Field(
        description=(
            "How completely the vocabulary is expressible in the ontology. 'lossless': every class "
            "carries over one to one. 'lossy': every class carries over, but two or more "
            "collapse into a single concept. 'partial': at least one class cannot carry over "
            "and Relabel drops it."
        )
    )
    correspondences: list[AlignmentCorrespondence] = Field(
        default_factory=list, description="Every accepted correspondence, carryable or diagnostic"
    )
    unaligned_source: list[str] = Field(
        default_factory=list,
        description=(
            "Class names with no carryable correspondence. These are out of vocabulary for this ontology, not invalid."
        ),
    )
    unaligned_target: list[str] = Field(
        default_factory=list,
        description="Concept labels this dataset does not cover",
    )
    class_remap: dict[str, str] = Field(
        default_factory=dict,
        description="Class name to target concept id, verbatim from label_alignment",
    )
    paste_remap: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Class name to target label. This is the form Relabel takes beside a list-valued "
            "`target`, which is the only form expressible in YAML. Identical to class_remap "
            "for a hand-built ontology, where ids are labels."
        ),
    )
    target_vocabulary: list[str] = Field(
        default_factory=list,
        description=(
            "Target concept labels in ontology index order, which is the `target` a Relabel view "
            "must pass. Datasets merged together must pass the identical list, or their "
            "integer labels denote different classes and merge_datasets rejects them."
        ),
    )
    ambiguous_labels: list[str] = Field(
        default_factory=list,
        description=(
            "Target labels naming more than one concept. While any exist, paste_remap and "
            "target_vocabulary cannot be used as emitted, because the index such a label "
            "takes is undetermined. Fix the ontology."
        ),
    )
    label_space_digest: str = Field(
        default="",
        description=(
            "Identity of the vocabulary this alignment defines, computed over the ontology, the "
            "paste_remap and the target_vocabulary. A downstream result conformed under this "
            "alignment carries the same value, which is how it is matched to this audit."
        ),
    )


class LabelAlignmentOutput(CoreOutput):
    """``scope.label-alignment``'s output: the alignment, and the resolved ontology a ``conform`` step relabels onto.

    ``ontology_source`` is how the config named that ontology (an ``ontologies:`` entry's name, the resolved path,
    ``inline`` or ``concepts``), which ``conform`` records with the label space it applies.
    """

    def __init__(self, alignment: LabelAlignment, ontology: Any, meta: Any, *, ontology_source: str | None) -> None:
        super().__init__(alignment.model_dump(mode="json"), meta)
        self.alignment = alignment
        self.ontology = ontology
        self.ontology_source = ontology_source


# Defined here to avoid an import cycle, but public in the scope evaluators: the step catalog names it by its home.
LabelAlignmentOutput.__module__ = "dataeval_flow.evaluators.scope"


def align_labels(ontology: "Ontology", class_names: "Sequence[str]", *, threshold: float = 0.0) -> LabelAlignment:
    """Align `class_names` to `ontology` with DataEval's label_alignment, with Flow's paste remap and digest.

    Two forms of the rewrite are returned. ``class_remap`` holds the concept ids DataEval produced, which are IRIs
    for an RDF artifact. ``paste_remap`` resolves those ids to labels, because :class:`dataeval.data.Relabel` reads
    its values as ids only when ``target`` is an :class:`~dataeval.Ontology` object, and the list-valued ``target``
    a YAML config can express takes labels. The two are identical for a hand-built ontology, where ids are labels.
    """
    from dataeval.core import label_alignment

    from dataeval_flow._label_space import label_space_digest, ontology_digest

    result = label_alignment(class_names, ontology, threshold=threshold)

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


def effective_mergeability(class_remap: "Mapping[str, str]", unaligned: "Sequence[str]") -> str:
    """``partial`` if a source class stays unaligned, ``lossy`` if two share a target, else ``lossless``."""
    if unaligned:
        return "partial"
    targets = list(class_remap.values())
    return "lossy" if len(set(targets)) < len(targets) else "lossless"
