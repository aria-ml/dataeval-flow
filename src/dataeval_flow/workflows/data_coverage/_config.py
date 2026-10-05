"""The ``data-coverage`` preset's config: the settings of its step types, and when a finding warns
(coverage spec §4.1, §4.3)."""

__all__ = [
    "CropParams",
    "DataCoverageChecks",
    "DataCoverageConfig",
    "DataCoverageCoverageSettings",
    "DataCoverageRepresentationSettings",
    "DiversitySettings",
    "FactorGapsSettings",
    "WrapSettings",
]

from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.workflows._base import WorkflowConfig


class DataCoverageCoverageSettings(BaseModel):
    """The `coverage` step's settings, with legacy data-coverage's defaults; each keeps its default when `coverage:`
    is written partly."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    method: Literal["naive", "adaptive"] = Field(
        default="adaptive",
        description=(
            "How the coverage radius is set: `adaptive`, a cutoff on the `percent` most sparsely neighbored items, or "
            "`naive`, a fixed analytic radius, judged by an `uncovered-items` step. DataEval's naive radius overflows "
            "past about 340 embedding dimensions; the step is then skipped with `failed: OverflowError`."
        ),
    )
    percent: float = Field(
        default=0.01, gt=0.0, lt=1.0, description="Fraction of items flagged as uncovered, for `adaptive` only."
    )
    num_observations: int = Field(
        default=50, gt=0, description="Neighbors an item needs within the radius to count as covered."
    )
    min_class_samples: int = Field(
        default=20, gt=0, description="Items a class needs before its dispersion and isotropy are judged."
    )
    isotropy_min_samples: int | None = Field(
        default=None,
        gt=0,
        description="Items a class needs before its isotropy is measured; unset uses DataEval's default.",
    )
    near_duplicate_factor: float = Field(
        default=0.5, gt=0.0, description="The fraction of the radius within which two items are near-duplicates."
    )


class CropParams(BaseModel):
    """`DetectionCrops`' settings, used on detection data only."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    padding: float = Field(
        default=0.0, ge=0.0, description="Fraction of each box's size added around it before cropping."
    )
    min_size: int = Field(
        default=1, ge=1, description="The smallest box side, in pixels, cropped; smaller boxes are dropped."
    )


class FactorGapsSettings(BaseModel):
    """The `factor-gaps` step's settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    mi_threshold: float = Field(
        default=0.1, ge=0.0, description="The least mutual information with the class a factor needs to be searched."
    )
    min_representation: int = Field(
        default=5, ge=1, description="A combination is a gap under this count while its expected count is over it."
    )


class WrapSettings(BaseModel):
    """The `wrap` step's settings, for the crops data-coverage measures detection data on: `DetectionCrops`' `params`.
    The preset fixes the wrapper and `other_kinds`."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    params: CropParams = Field(default_factory=CropParams, description="`DetectionCrops`' parameters.")


class DiversitySettings(BaseModel):
    """The `diversity` step's settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    method: Literal["simpson", "shannon"] = Field(default="simpson", description="The diversity index.")


class DataCoverageRepresentationSettings(BaseModel):
    """The `representation` step's settings: each class's minimum share, for the worklist."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    expected: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]] | None = Field(
        default=None,
        description=(
            "Class name to its minimum expected share of the dataset, a fraction in [0, 1]; a name that is no class "
            "is ignored and noted."
        ),
    )


class DataCoverageClassImbalanceSettings(BaseModel):
    """The `class-imbalance` check's fields, with legacy data-coverage's defaults. Named for the preset, so it
    reaches the schema `$defs` apart from data-splitting's and data-cleaning's limits."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: float | None = Field(
        default=5.0,
        ge=1.0,
        description=(
            "Largest class count over smallest, among the classes with labels, past which the Class Imbalance "
            "finding warns; `null` judges nothing but an empty class, which always warns."
        ),
    )
    info: float | None = Field(
        default=2.0,
        ge=1.0,
        description=(
            "The ratio at or under which the finding is ok, between which and `warning` it informs; `null` makes every "
            "ratio under `warning` information. Must not exceed `warning`. Defaults to 2.0, or `warning` where "
            "that is lower and `info` is unset."
        ),
    )

    @model_validator(mode="after")
    def _info_under_ratio(self) -> Self:
        """An unset `info` follows a `warning` under it; two written bounds that cross
        are refused here, where the user wrote them."""
        if self.warning is None or self.info is None:
            return self
        if "info" not in self.model_fields_set:
            # derived, so still unset: a matrix varies `warning` alone
            object.__setattr__(self, "info", min(self.info, self.warning))
        elif self.info > self.warning:
            raise ValueError(f"`info` ({self.info}) must not exceed `warning` ({self.warning}).")
        return self


class FactorCoverageGapsSettings(BaseModel):
    """The `factor-coverage-gaps` check's field, with legacy data-coverage's default."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: int | None = Field(
        default=2,
        ge=0,
        description=(
            "The most under-represented class-factor-value combinations before the Factor Coverage Gaps finding "
            "warns; `null` never warns."
        ),
    )


class ClassCoverageSettings(BaseModel):
    """The `class-coverage` check's fields, with legacy data-coverage's defaults."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    dispersion: float | None = Field(
        default=0.5,
        ge=0.0,
        description=(
            "An assessable class's dispersion under which it is clustered, and the Class Coverage finding warns; "
            "`null` turns this criterion off."
        ),
    )
    isotropy: float | None = Field(
        default=0.5,
        ge=0.0,
        description=(
            "An assessable class's isotropy under which it is one-dimensional, and the Class Coverage finding "
            "warns; `null` turns this criterion off."
        ),
    )
    near_duplicates: float | None = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "An assessable class's share in near-duplicate pairs over which it is duplicate-padded, and the Class "
            "Coverage finding warns; `null` turns this criterion off."
        ),
    )


class DataCoverageUncoveredItemsSettings(BaseModel):
    """The `uncovered-items` check's field, with legacy data-coverage's default, read under `naive` coverage only.
    Named for the preset, as `DataCoverageClassImbalanceSettings` is."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: float | None = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description=(
            "The percent of the items uncovered past which the Uncovered Items finding warns, under `naive` coverage "
            "only; `null` judges nothing."
        ),
    )


class DimensionalCompletenessSettings(BaseModel):
    """The `dimensional-completeness` check's fields, with legacy data-coverage's defaults."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: float | None = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "The completeness score under which the Dimensional Completeness finding warns; `null` turns this band "
            "off. Must not exceed `info`."
        ),
    )
    info: float | None = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description=(
            "The score under which the finding informs, at or over which it is ok; `null` turns this band off, and "
            "with `warning` also `null` the finding judges nothing. An unset `info` is 0.8, or `warning` where that "
            "is higher."
        ),
    )

    @model_validator(mode="after")
    def _warning_under_info(self) -> Self:
        """An unset `info` follows a `warning` over it, as legacy's unreachable band did; two written bounds that cross
        are refused here, where the user wrote them."""
        if self.warning is None or self.info is None:
            return self
        if "info" not in self.model_fields_set:
            # derived, so still unset: a matrix varies `warning` alone
            object.__setattr__(self, "info", max(self.info, self.warning))
        elif self.warning > self.info:
            raise ValueError(f"`warning` ({self.warning}) must not exceed `info` ({self.info}).")
        return self


class DataCoverageChecks(BaseModel):
    """When data-coverage's findings warn: each check's fields, keyed by check type."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    class_imbalance: DataCoverageClassImbalanceSettings = Field(
        default_factory=DataCoverageClassImbalanceSettings,
        alias="class-imbalance",
        description="The `class-imbalance` check's thresholds.",
    )
    factor_coverage_gaps: FactorCoverageGapsSettings = Field(
        default_factory=FactorCoverageGapsSettings,
        alias="factor-coverage-gaps",
        description="The `factor-coverage-gaps` check's threshold.",
    )
    class_coverage: ClassCoverageSettings = Field(
        default_factory=ClassCoverageSettings,
        alias="class-coverage",
        description="The `class-coverage` check's thresholds.",
    )
    uncovered_items: DataCoverageUncoveredItemsSettings = Field(
        default_factory=DataCoverageUncoveredItemsSettings,
        alias="uncovered-items",
        description="The `uncovered-items` check's threshold, under `naive` coverage.",
    )
    dimensional_completeness: DimensionalCompletenessSettings = Field(
        default_factory=DimensionalCompletenessSettings,
        alias="dimensional-completeness",
        description="The `dimensional-completeness` check's thresholds.",
    )


_NO_ONTOLOGY = (
    "data-coverage no longer judges an ontology: run a `label-space` entry on the same source, with this `ontology:` "
    "(and `ontology-validation.label_pattern:`)"
)

_MOVED: dict[str, str] = {
    "coverage_method": "it is `coverage.method`",
    "coverage_percent": "it is `coverage.percent`",
    "num_observations": "it is `coverage.num_observations`",
    "min_class_samples": "it is `coverage.min_class_samples`",
    "isotropy_min_samples": "it is `coverage.isotropy_min_samples`",
    "near_duplicate_factor": "it is `coverage.near_duplicate_factor`",
    "crop_padding": "it is `wrap.params.padding`",
    "crop_min_size": "it is `wrap.params.min_size`",
    "run_completeness": "it is `completeness`",
    "balance": "balance always runs, as a report section; `factor-gaps: false` leaves out the gap analysis",
    "diversity_method": "diversity always runs, as a report section, and `diversity.method` picks the method",
    "run_gap_analysis": "write `factor-gaps: false` to leave out the gap analysis",
    "gap_mi_threshold": "it is `factor-gaps.mi_threshold`",
    "gap_min_representation": "it is `factor-gaps.min_representation`",
    "ontology_label_pattern": _NO_ONTOLOGY,
    "ontology_expected": (
        "it is `representation.expected`, or `label-space`'s `representation.expected` where an ontology is set"
    ),
    "metadata_auto_bin_method": "name a policy under `metadata:`",
    "metadata_exclude": "name a policy under `metadata:`",
    "metadata_continuous_factor_bins": "name a policy under `metadata:`",
    "metadata_factor_source": "name a policy under `metadata:`",
    "value_range": "set `value_range` on the dataset",
    "stats": "no step of data-coverage reads statistics",
}

_THRESHOLDS_MOVED: dict[str, str] = {
    "class_imbalance_ratio": "`checks.class-imbalance.warning`",
    "gap_count": (
        "`checks.factor-coverage-gaps.warning`, set one less: legacy warned at `gap_count` gaps, this warns past the "
        "bound, so `gap_count: N` is `warning: N-1`"
    ),
    "min_dispersion": "`checks.class-coverage.dispersion`",
    "min_isotropy": "`checks.class-coverage.isotropy`",
    "max_near_duplicate_fraction": "`checks.class-coverage.near_duplicates`",
    "uncovered_rate": "`checks.uncovered-items.warning`",
    "completeness_score": "`checks.dimensional-completeness.warning`",
    "leaf_coverage": "`label-space`'s `checks.leaf-coverage.coverage`",
    "dark_branch_count": "`label-space`'s `checks.leaf-coverage.empty_branches`",
    "unmatched_class_count": "`label-space`'s `checks.label-conformance.warning`",
}


class DataCoverageConfig(WorkflowConfig[ChainResult], MetadataConfigMixin):
    """The settings of one ``data-coverage`` entry: each step type's settings, keyed by the step type, and when a
    finding warns.

    Example YAML::

        workflows:
          - name: coverage
            type: data-coverage
            metadata: standard
            coverage: {method: adaptive, num_observations: 50}
            wrap: {params: {padding: 0.1}}
            factor-gaps: {mi_threshold: 0.1}
    """

    type: str = Field(default="data-coverage", description="The workflow type this entry configures: `data-coverage`.")
    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.METADATA}), optional=frozenset({InputKind.EMBEDDINGS}), sources=SourceCount.ONE
    )

    representation: DataCoverageRepresentationSettings = Field(
        default_factory=DataCoverageRepresentationSettings, description="The `representation` step's settings."
    )
    coverage: DataCoverageCoverageSettings = Field(
        default_factory=DataCoverageCoverageSettings,
        description="The `coverage` step's settings, run when the task names an extractor.",
    )
    wrap: WrapSettings = Field(
        default_factory=WrapSettings, description="The `wrap` step's settings, used on detection data only."
    )
    completeness: bool = Field(
        default=True, description="Whether the completeness steps run, when the task names an extractor."
    )
    diversity: DiversitySettings = Field(
        default_factory=DiversitySettings, description="The `diversity` step's settings."
    )
    factor_gaps: FactorGapsSettings | Literal[False] = Field(
        default_factory=FactorGapsSettings,
        alias="factor-gaps",
        description="The `factor-gaps` step's settings; `false` leaves out the gap analysis and its check.",
    )
    checks: DataCoverageChecks = Field(
        default_factory=DataCoverageChecks, description="When findings warn, keyed by check type."
    )

    @model_validator(mode="before")
    @classmethod
    def _refuse_legacy_fields(cls, data: Any) -> Any:
        """Refuse every legacy data-coverage field by name, saying what replaced it (coverage spec §4.3)."""
        if not isinstance(data, dict):
            return data
        if data.get("ontology") is not None:
            raise ValueError(
                f"{_NO_ONTOLOGY}. A data-coverage run on a conformed source then records no label space of its own; "
                "`label-space` carries the join key."
            )
        for key, message in _MOVED.items():
            if key in data:
                raise ValueError(f"data-coverage's `{key}` is refused: {message}.")
        thresholds = data.get("health_thresholds")
        if isinstance(thresholds, dict):
            for key, replacement in _THRESHOLDS_MOVED.items():
                # legacy's values were numbers; a mapping under a snake_case name is a check type's own limits
                if key in thresholds and not isinstance(thresholds[key], dict):
                    raise ValueError(f"`health_thresholds.{key}` is refused: it is {replacement}.")
        return data
