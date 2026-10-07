"""The ``scope`` preset's config: the settings of its step types, and when a finding warns
(coverage spec §4.1, §4.3). Class balance and the metadata factors are `bias`'s."""

__all__ = [
    "CoverageSettings",
    "CropParams",
    "RepresentationSettings",
    "ScopeChecks",
    "ScopeConfig",
    "UncoveredItemsSettings",
    "WrapSettings",
]

from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.workflows._base import WorkflowConfig


class CoverageSettings(BaseModel):
    """The `coverage` step's settings; each unset one is DataEval's default."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    method: Literal["naive", "adaptive"] | None = Field(
        default=None,
        description=(
            "How the coverage radius is set: `adaptive`, a cutoff on the `percent` most sparsely neighbored items, or "
            "`naive`, a fixed analytic radius, judged by an `uncovered-items` step. DataEval's naive radius overflows "
            "past about 340 embedding dimensions; the step is then skipped with `failed: OverflowError`. Unset is "
            "DataEval's default, `adaptive`."
        ),
    )
    percent: float | None = Field(
        default=None,
        gt=0.0,
        lt=1.0,
        description="Fraction of items flagged as uncovered, for `adaptive` only. Unset is DataEval's default, 0.01.",
    )
    num_observations: int | None = Field(
        default=None,
        gt=0,
        description="Neighbors an item needs within the radius to count as covered. Unset is DataEval's default, 20.",
    )
    min_class_samples: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Items a class needs before its dispersion and isotropy are judged. Unset is DataEval's default, 20."
        ),
    )
    isotropy_min_samples: int | None = Field(
        default=None,
        gt=0,
        description="Items a class needs before its isotropy is measured; unset uses DataEval's default.",
    )
    near_duplicate_factor: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "The fraction of the radius within which two items are near-duplicates. Unset is DataEval's default, 0.5."
        ),
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


class WrapSettings(BaseModel):
    """The `wrap` step's settings, for the crops scope measures detection data on: `DetectionCrops`' `params`.
    The preset fixes the wrapper and `other_kinds`."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    params: CropParams = Field(default_factory=CropParams, description="`DetectionCrops`' parameters.")


class RepresentationSettings(BaseModel):
    """The `representation` step's settings: each class's minimum share, for the worklist."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    expected: dict[str, Annotated[float, Field(ge=0.0, le=1.0)]] | None = Field(
        default=None,
        description=(
            "Class name to its minimum expected share of the dataset, a fraction in [0, 1]. A name that is no class, "
            "or, under an ontology, resolves to no concept or to several, is ignored and noted."
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


class UncoveredItemsSettings(BaseModel):
    """The `uncovered-items` check's field, with legacy data-coverage's default, read under `naive` coverage only."""

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


class ScopeChecks(BaseModel):
    """When scope's findings warn: each check's fields, keyed by check type."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    class_coverage: ClassCoverageSettings = Field(
        default_factory=ClassCoverageSettings,
        alias="class-coverage",
        description="The `class-coverage` check's thresholds.",
    )
    uncovered_items: UncoveredItemsSettings = Field(
        default_factory=UncoveredItemsSettings,
        alias="uncovered-items",
        description="The `uncovered-items` check's threshold, under `naive` coverage.",
    )
    dimensional_completeness: DimensionalCompletenessSettings = Field(
        default_factory=DimensionalCompletenessSettings,
        alias="dimensional-completeness",
        description="The `dimensional-completeness` check's thresholds.",
    )


_NO_ONTOLOGY = (
    "scope no longer judges an ontology: run a `taxonomy` entry on the same source, with this `ontology:` "
    "(and `ontology-validation.label_pattern:`)"
)


class ScopeConfig(WorkflowConfig[ChainResult]):
    """The settings of one ``scope`` entry: each step type's settings, keyed by the step type, and when a
    finding warns.

    Example YAML::

        workflows:
          - name: coverage
            type: scope
            coverage: {method: adaptive, num_observations: 50}
            wrap: {params: {padding: 0.1}}
    """

    type: str = Field(default="scope", description="The workflow type this entry configures: `scope`.")
    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.LABELS}), optional=frozenset({InputKind.EMBEDDINGS}), sources=SourceCount.ONE
    )

    representation: RepresentationSettings = Field(
        default_factory=RepresentationSettings, description="The `representation` step's settings."
    )
    coverage: CoverageSettings = Field(
        default_factory=CoverageSettings,
        description="The `coverage` step's settings, run when the task names an extractor.",
    )
    wrap: WrapSettings = Field(
        default_factory=WrapSettings, description="The `wrap` step's settings, used on detection data only."
    )
    completeness: bool = Field(
        default=True, description="Whether the completeness steps run, when the task names an extractor."
    )
    checks: ScopeChecks = Field(default_factory=ScopeChecks, description="When findings warn, keyed by check type.")

    @model_validator(mode="before")
    @classmethod
    def _refuse_ontology(cls, data: Any) -> Any:
        """Refuse an `ontology`: no step of this preset judges labels against one."""
        if isinstance(data, dict) and data.get("ontology") is not None:
            raise ValueError(
                f"{_NO_ONTOLOGY}. A scope run on a conformed source then records no label space of its own; "
                "`taxonomy` carries the join key."
            )
        return data
