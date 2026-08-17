"""Data coverage workflow parameters."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from dataeval_flow.workflow.base import MetadataConfigMixin, StatsConfigMixin, WorkflowParametersBase

__all__ = ["DataCoverageHealthThresholds", "DataCoverageParameters"]


class DataCoverageHealthThresholds(BaseModel):
    """Configurable warning thresholds for data coverage health status.

    Each threshold controls when the corresponding finding is elevated to
    ``severity="warning"``; otherwise it stays at ``severity="info"``.

    All ten are required numbers — ``None`` is not accepted. To stop a metric
    from ever warning, set it past the value it can reach: ``uncovered_rate=100.0``,
    ``completeness_score=0.0``, ``leaf_coverage=0.0``, or a
    ``class_imbalance_ratio``/``gap_count`` above anything the dataset will produce.
    """

    uncovered_rate: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description=(
            "Max allowable % of embedding observations flagged as uncovered. "
            "Default 10%. Lower for safety-critical datasets. Only applied when the "
            "headline result comes from coverage_naive — coverage_adaptive flags a fixed "
            "coverage_percent of observations, so its rate is not a health signal."
        ),
    )
    completeness_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum dimensional completeness score before warning. "
            "Default 0.5. Raise to 0.7–0.8 for high-dimensional models."
        ),
    )
    class_imbalance_ratio: float = Field(
        default=5.0,
        ge=1.0,
        description=(
            "Max allowable ratio between the largest and smallest class counts (max_class / min_class). Default 5:1."
        ),
    )
    gap_count: int = Field(
        default=3,
        ge=0,
        description=("Number of metadata coverage gaps before elevating to warning. Default 3."),
    )
    min_dispersion: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "Minimum per-class dispersion before warning. A class below this spreads less "
            "than half as far as a typical class — clustered. Default 0.5."
        ),
    )
    min_isotropy: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "Minimum per-class isotropy before warning. A class below this varies along too "
            "few independent directions — one-dimensional. Default 0.5."
        ),
    )
    max_near_duplicate_fraction: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Maximum share of a class allowed to sit in near-duplicate pairs before warning. "
            "Above this the class is padded with repeated frames. Default 0.1."
        ),
    )
    leaf_coverage: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fraction of sanctioned leaf species with any examples before warning. "
            "Default 0.9. Applied only to a configured ontology — against a synthesized one "
            "leaf coverage is 1.0 by construction."
        ),
    )
    dark_branch_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of wholly-unpopulated ontology branches tolerated before warning. Default "
            "0, so any dark branch warns. Applied only to a configured ontology."
        ),
    )
    unmatched_class_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of class names that may fail to resolve to an ontology concept before "
            "warning. Default 0. Applied only to a configured ontology."
        ),
    )


class DataCoverageParameters(WorkflowParametersBase, MetadataConfigMixin, StatsConfigMixin):
    """Parameters for data coverage workflow.

    Embedding-based analyses (coverage, completeness) require an extractor
    and are skipped when none is configured. Metadata-based analyses always run.
    """

    # --- Embedding coverage ---
    coverage_method: Literal["naive", "adaptive"] = Field(
        default="adaptive",
        description=(
            "Coverage radius method. 'adaptive' flags the sparsest coverage_percent of "
            "observations — a useful shortlist, but its rate restates the config, so the "
            "uncovered_rate threshold is not applied to it. 'naive' compares each radius "
            "against a fixed geometric radius, giving a data-driven rate, but that radius "
            "grows with embedding dimensionality and saturates on wide embeddings. Either "
            "way, the per-class dispersion / isotropy / near-duplicate signals are the "
            "data-driven health signal."
        ),
    )
    coverage_percent: float = Field(
        default=0.01,
        gt=0.0,
        lt=1.0,
        description="Proportion of observations considered uncovered for coverage_adaptive.",
    )
    num_observations: int = Field(
        default=50,
        ge=1,
        description="Number of neighbors for coverage functions.",
    )
    min_class_samples: int = Field(
        default=20,
        ge=1,
        description=(
            "Minimum samples for a class to get per-class variety signals. Smaller classes "
            "are reported with assessable=False and null dispersion/isotropy/near-duplicates."
        ),
    )
    isotropy_min_samples: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Minimum samples for a class's isotropy to be reported. None means the embedding "
            "dimensionality plus one — isotropy is undefined below that."
        ),
    )
    near_duplicate_factor: float = Field(
        default=0.5,
        gt=0.0,
        description=(
            "A within-class nearest-neighbor pair counts as a near-duplicate when it is "
            "closer than this multiple of the typical within-class neighbor distance."
        ),
    )

    # --- Completeness ---
    run_completeness: bool = Field(
        default=True,
        description="Compute dimensional completeness when an extractor is available.",
    )

    # --- Metadata distribution ---
    balance: bool = Field(
        default=True,
        description="Run Balance (MI) analysis on metadata factors.",
    )
    diversity_method: Literal["simpson", "shannon"] | None = Field(
        default="simpson",
        description="Diversity method (None = skip).",
    )

    # --- Metadata gap analysis ---
    run_gap_analysis: bool = Field(
        default=True,
        description="Compute mutual information and cross-reference per-class metadata to identify gaps.",
    )
    gap_mi_threshold: float = Field(
        default=0.1,
        ge=0.0,
        description="Minimum MI score for a factor to be included in gap analysis.",
    )
    gap_min_representation: int = Field(
        default=5,
        ge=1,
        description=(
            "Minimum expected count per class-factor-value combination. Classes with fewer samples are flagged as gaps."
        ),
    )

    # --- Ontology ---
    ontology: dict[str, Any] | str | None = Field(
        default=None,
        description=(
            "Sanctioned label space. A nested mapping of concept to children is read as an "
            "inline hierarchy; a string is read as a path to a serialized RDF artifact "
            "(.ttl/.rdf/.owl/.xml/.nt/.jsonld), resolved against the data root. When unset, a "
            "flat ontology is synthesized from the dataset's index2label — enough for a class "
            "balance worklist, but it can only name classes the dataset already declares."
        ),
    )
    ontology_expected: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]] | None = Field(
        default=None,
        description=(
            "Class name to its minimum expected share of the dataset, as a fraction in [0, 1]. "
            "Named classes use this floor as their collection target instead of the uniform "
            "share, and a dataset below the floor is reported as a violation."
        ),
    )
    ontology_label_pattern: str | None = Field(
        default=None,
        description=(
            "Regex that ontology concept labels must match, e.g. '^[a-z0-9_]+$' for a "
            "lowercase_snake_case lint. Labels that fail are reported. Ignored when the "
            "ontology is synthesized."
        ),
    )

    # --- Health thresholds ---
    health_thresholds: DataCoverageHealthThresholds = Field(
        default_factory=DataCoverageHealthThresholds,
        description="Warning thresholds for dataset coverage health status.",
    )
