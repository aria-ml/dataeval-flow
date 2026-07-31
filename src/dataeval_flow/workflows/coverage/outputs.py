"""Data coverage workflow outputs."""

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from pydantic import BaseModel, Field
from typing_extensions import TypeIs

from dataeval_flow.config.schemas import ResultMetadata
from dataeval_flow.workflow.base import Reportable, WorkflowOutputsBase, WorkflowReportBase

if TYPE_CHECKING:
    from dataeval_flow.workflow import WorkflowResult

__all__ = [
    "ClassCoverageRow",
    "ClassMetadataGap",
    "CompletenessAssessment",
    "CoverageAssessment",
    "DarkBranch",
    "DataCoverageMetadata",
    "DataCoverageOutputs",
    "DataCoverageRawOutputs",
    "DataCoverageReport",
    "DataCoverageResult",
    "LabelConformance",
    "LabelSpaceCoverage",
    "MetadataDistributionResult",
    "MetadataGapResult",
    "OntologyAssessment",
    "OntologyStructure",
    "RepresentationRow",
    "RepresentationViolation",
    "is_coverage_result",
]


# ---------------------------------------------------------------------------
# Embedding-based assessment sub-models
# ---------------------------------------------------------------------------


class ClassCoverageRow(BaseModel):
    """Per-class embedding variety signals from ``dataeval.scope.Coverage``."""

    class_name: str = Field(description="Class label name")
    count: int = Field(description="Number of samples in this class")
    uncovered: int = Field(description="Samples of this class in sparse regions")
    uncovered_fraction: float = Field(description="Fraction of this class flagged uncovered")
    dispersion: float | None = Field(
        default=None,
        description=(
            "Magnitude of spread: mean distance to the class centroid relative to a typical "
            "class. Around 1.0 is typical; well below 1.0 means the class is clustered. "
            "None when the class is below min_class_samples."
        ),
    )
    isotropy: float | None = Field(
        default=None,
        description=(
            "Shape of the spread: how many independent directions the class varies in, "
            "relative to a typical class. Low means one-dimensional even when dispersion is "
            "normal. None when the class has fewer samples than embedding dimensions."
        ),
    )
    near_duplicate_fraction: float | None = Field(
        default=None,
        description=(
            "Share of the class sitting in unusually tight nearest-neighbor pairs. High means "
            "a chunk of the class is repeated frames. None when the class is unassessable."
        ),
    )
    assessable: bool = Field(description="Whether the class had enough samples for per-class signals")


class CoverageAssessment(BaseModel):
    """Embedding-space coverage analysis results."""

    method: str = Field(
        description=(
            "Coverage radius method that produced these numbers. Normally 'naive' or "
            "'adaptive'; reads 'adaptive (naive overflowed)' when the naive radius was not "
            "computable for the embedding width and adaptive was substituted."
        )
    )
    uncovered_count: int = Field(description="Number of uncovered observations")
    uncovered_rate: float = Field(description="Fraction of observations flagged as uncovered")
    coverage_radius: float = Field(description="Coverage radius threshold")
    observation_count: int = Field(
        default=0,
        description="Number of embeddings assessed — images, or detection crops for object detection",
    )
    observation_unit: str = Field(
        default="image",
        description=(
            "What one assessed observation is: 'image' for image classification, "
            "'detection crop' when an object-detection dataset was presented as its "
            "ground-truth boxes via DetectionCrops"
        ),
    )
    per_class: list[ClassCoverageRow] = Field(
        default_factory=list,
        description="Per-class variety signals, lowest dispersion first",
    )


class CompletenessAssessment(BaseModel):
    """Dimensional completeness results."""

    completeness_score: float = Field(description="Completeness score between 0 and 1")
    nearest_neighbor_pairs: list[list[int]] = Field(default_factory=list, description="Point pairs sorted by distance")


# ---------------------------------------------------------------------------
# Metadata-based assessment sub-models
# ---------------------------------------------------------------------------


class MetadataDistributionResult(BaseModel):
    """Metadata factor distribution analysis results."""

    metadata_factors: list[str] = Field(description="Names of metadata factors present")
    metadata_summary: dict[str, dict[str, Any]] = Field(description="Per-factor summary statistics")
    balance_summary: dict[str, Any] | None = Field(default=None, description="Balance (MI) analysis results")
    diversity_summary: dict[str, Any] | None = Field(default=None, description="Diversity analysis results")


class ClassMetadataGap(BaseModel):
    """A single metadata coverage gap for a class-factor-value combination."""

    class_name: str = Field(description="Class label name")
    factor_name: str = Field(description="Metadata factor name")
    factor_value: str = Field(description="Metadata factor value that is under-represented")
    class_count: int = Field(description="Actual count of this class with this factor value")
    expected_count: float = Field(description="Expected count based on overall distribution")
    deficit: float = Field(description="Relative deficit (0 to 1, where 1 means complete absence)")


class MetadataGapResult(BaseModel):
    """Metadata coverage gap analysis results."""

    mutual_info_class_to_factor: dict[str, float] = Field(
        description="MI scores between class labels and each metadata factor"
    )
    gaps: list[ClassMetadataGap] = Field(default_factory=list, description="Identified under-representations by class")


# ---------------------------------------------------------------------------
# Label distribution
# ---------------------------------------------------------------------------


class LabelDistributionResult(BaseModel):
    """Label completeness and distribution results."""

    num_classes: int = Field(description="Number of class labels actually observed in the dataset")
    class_distribution: dict[str, int] = Field(
        description="Mapping of class name to label count, including declared-but-absent classes at 0"
    )
    empty_images: list[int] = Field(default_factory=list, description="Indices of images with no labels")
    missing_classes: list[str] = Field(
        default_factory=list,
        description="Classes declared in the dataset's index2label that have zero samples",
    )


# ---------------------------------------------------------------------------
# Ontology assessment sub-models
# ---------------------------------------------------------------------------


class RepresentationRow(BaseModel):
    """One leaf species short of its expected share — a collection worklist row."""

    concept: str = Field(description="Ontology concept id")
    label: str = Field(description="Human-readable concept label")
    parent: str = Field(description="Comma-separated parent labels, empty for a root")
    action: Literal["acquire", "augment"] = Field(
        description="'acquire' when the class has no examples at all, 'augment' when it has too few"
    )
    count: int = Field(description="Observed samples for this concept")
    target: int = Field(description="Samples expected under the uniform or asserted share")
    deficit: int = Field(description="target minus count, always positive")


class DarkBranch(BaseModel):
    """A maximal internal branch of the ontology with no samples anywhere under it."""

    concept: str = Field(description="Ontology concept id at the top of the dark branch")
    label: str = Field(description="Human-readable concept label")
    leaves: int = Field(description="Number of leaf species under this branch")


class RepresentationViolation(BaseModel):
    """An asserted minimum share the dataset does not meet."""

    concept: str = Field(description="Ontology concept id")
    label: str = Field(description="Human-readable concept label")
    floor: float = Field(description="Asserted minimum share of the dataset")
    actual: float = Field(description="Observed share of the dataset")
    shortfall: int = Field(description="Samples needed to reach the floor")


class LabelSpaceCoverage(BaseModel):
    """How a dataset's label mass covers an ontology's sanctioned label space."""

    leaf_coverage: float = Field(description="Fraction of sanctioned leaf species with any examples")
    total_deficit: int = Field(description="Sum of all positive deficits — the collection budget")
    worklist: list[RepresentationRow] = Field(
        default_factory=list, description="Concepts to acquire or augment, largest deficit first"
    )
    dark_branches: list[DarkBranch] = Field(
        default_factory=list, description="Wholly-unpopulated branches, most leaves first"
    )
    violations: list[RepresentationViolation] = Field(
        default_factory=list, description="Asserted minimum shares the dataset does not meet"
    )
    ignored_expected: list[str] = Field(
        default_factory=list,
        description=(
            "Class names given in ontology_expected that resolved to zero or several concepts "
            "and were therefore ignored"
        ),
    )


class LabelConformance(BaseModel):
    """Whether a dataset's class names resolve to ontology concepts."""

    conforms: bool = Field(description="True when every class name resolves to exactly one concept")
    matched: dict[str, str] = Field(
        default_factory=dict, description="Class name to the single concept id it resolved to"
    )
    unmatched: list[str] = Field(
        default_factory=list,
        description="Class names resolving to no concept — out of vocabulary, a typo, or an unsanctioned class",
    )
    ambiguous: dict[str, list[str]] = Field(
        default_factory=dict, description="Class name to the several concept ids it resolved to"
    )


class OntologyStructure(BaseModel):
    """Structural facts about the ontology artifact itself."""

    concept_count: int = Field(description="Number of defined concepts")
    leaf_count: int = Field(description="Number of leaf concepts")
    max_depth: int = Field(description="Longest is-a path from a root")
    roots: list[str] = Field(default_factory=list, description="Concepts with no parents")
    isolated: list[str] = Field(default_factory=list, description="Concepts with neither parents nor children")
    external_ancestors: dict[str, list[str]] = Field(
        default_factory=dict, description="Concept id to parent ids not defined in this ontology"
    )
    redundant_edges: list[list[str]] = Field(
        default_factory=list, description="Direct is-a edges already implied by a longer path, as [parent, child]"
    )
    ancestor_siblings: list[list[str]] = Field(
        default_factory=list, description="Pairs where a concept is declared alongside its own ancestor"
    )
    unary_parents: list[str] = Field(
        default_factory=list, description="Concepts with exactly one child — depth without discrimination"
    )
    label_collisions: dict[str, list[str]] = Field(
        default_factory=dict, description="Name to the several concept ids it resolves to"
    )
    nonconforming_labels: dict[str, str] = Field(
        default_factory=dict, description="Concept id to the label that failed ontology_label_pattern"
    )


class OntologyAssessment(BaseModel):
    """Complete ontology analysis for one dataset."""

    source: str = Field(description="'inline', the resolved ontology file path, or 'index2label' when synthesized")
    synthesized: bool = Field(
        description=(
            "True when the ontology was built from the dataset's own index2label rather than "
            "configured. A synthesized ontology can only name classes the dataset declares, so "
            "conformance and structure are omitted and leaf_coverage / dark_branches are not "
            "health-checked."
        )
    )
    representation: LabelSpaceCoverage = Field(description="Label-space coverage and the collection worklist")
    conformance: LabelConformance | None = Field(
        default=None, description="Class-name conformance (None when the ontology was synthesized)"
    )
    structure: OntologyStructure | None = Field(
        default=None, description="Ontology artifact lint (None when the ontology was synthesized)"
    )


# ---------------------------------------------------------------------------
# Workflow output models
# ---------------------------------------------------------------------------


class DataCoverageRawOutputs(WorkflowOutputsBase):
    """Machine-readable results from data coverage workflow."""

    coverage: CoverageAssessment | None = Field(
        default=None, description="Embedding coverage results (None if no extractor)"
    )
    coverage_skipped_reason: str | None = Field(
        default=None,
        description="Why embedding coverage was skipped despite an extractor being configured",
    )
    completeness: CompletenessAssessment | None = Field(
        default=None, description="Dimensional completeness results (None if no extractor)"
    )
    metadata_distribution: MetadataDistributionResult = Field(description="Metadata factor distribution analysis")
    metadata_gaps: MetadataGapResult | None = Field(
        default=None, description="Metadata coverage gaps by class (None if gap analysis disabled)"
    )
    label_distribution: LabelDistributionResult = Field(description="Label distribution and completeness")
    ontology: OntologyAssessment | None = Field(
        default=None,
        description="Ontology label-space analysis (None when it could not run)",
    )
    ontology_skipped_reason: str | None = Field(
        default=None,
        description="Why ontology analysis was skipped",
    )


class DataCoverageReport(WorkflowReportBase):
    """Human-readable report for data coverage workflow."""

    findings: list[Reportable] = Field(default_factory=list)


class DataCoverageOutputs(BaseModel):
    """Complete data coverage workflow output."""

    raw: DataCoverageRawOutputs
    report: DataCoverageReport


class DataCoverageMetadata(ResultMetadata):
    """Metadata for the data-coverage workflow."""

    mode: Literal["advisory", "preparatory"] = "advisory"
    has_extractor: bool = Field(default=False, description="Whether an extractor was configured")


# ---------------------------------------------------------------------------
# Type alias and TypeIs guard for type narrowing
# ---------------------------------------------------------------------------

#: Fully typed result alias for the data-coverage workflow.
DataCoverageResult: TypeAlias = "WorkflowResult[DataCoverageMetadata, DataCoverageOutputs]"


def is_coverage_result(
    result: "WorkflowResult[Any, Any]",
) -> TypeIs["WorkflowResult[DataCoverageMetadata, DataCoverageOutputs]"]:
    """Narrow a generic ``WorkflowResult`` to a data-coverage result."""
    return isinstance(result.metadata, DataCoverageMetadata)
