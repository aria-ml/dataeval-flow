"""The ``data-bias`` preset's config: the settings of its step types, and when a finding warns."""

__all__ = [
    "DataBiasChecks",
    "DataBiasConfig",
    "DiversitySettings",
    "FactorGapsSettings",
    "FactorParitySettings",
    "ShortcutRiskSettings",
]

from typing import ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.workflows._base import WorkflowConfig


class FactorGapsSettings(BaseModel):
    """The `factor-gaps` step's settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    mi_threshold: float = Field(
        default=0.1, ge=0.0, description="The least mutual information with the class a factor needs to be searched."
    )
    min_representation: int = Field(
        default=5, ge=1, description="A combination is a gap under this count while its expected count is over it."
    )


class DiversitySettings(BaseModel):
    """The `diversity` step's settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    method: Literal["simpson", "shannon"] = Field(default="simpson", description="The diversity index.")


class DataBiasClassImbalanceSettings(BaseModel):
    """The `class-imbalance` check's fields, with legacy data-coverage's defaults. Named for the preset, so it reaches
    the schema `$defs` apart from audit's limits."""

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


class ShortcutRiskSettings(BaseModel):
    """The `shortcut-risk` check's settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: float | None = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "The mutual information with the class, from 0 to 1, past which a factor warns; `null` judges nothing."
        ),
    )


class FactorParitySettings(BaseModel):
    """The `factor-parity` check's settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: float | None = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "The bias-corrected Cramér's V with the class, from 0 to 1, past which a significant factor warns; `null` "
            "judges nothing."
        ),
    )
    p_value: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="The chi-square p-value at or under which a factor's association counts as significant.",
    )


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


class DataBiasChecks(BaseModel):
    """When data-bias's findings warn: each check's fields, keyed by check type."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    class_imbalance: DataBiasClassImbalanceSettings = Field(
        default_factory=DataBiasClassImbalanceSettings,
        alias="class-imbalance",
        description="The `class-imbalance` check's thresholds.",
    )
    shortcut_risk: ShortcutRiskSettings = Field(
        default_factory=ShortcutRiskSettings,
        alias="shortcut-risk",
        description="The `shortcut-risk` check's threshold.",
    )
    factor_parity: FactorParitySettings = Field(
        default_factory=FactorParitySettings,
        alias="factor-parity",
        description="The `factor-parity` check's thresholds.",
    )
    factor_coverage_gaps: FactorCoverageGapsSettings = Field(
        default_factory=FactorCoverageGapsSettings,
        alias="factor-coverage-gaps",
        description="The `factor-coverage-gaps` check's threshold.",
    )


class DataBiasConfig(WorkflowConfig[ChainResult], MetadataConfigMixin):
    """The settings of one ``data-bias`` entry: each step type's settings, keyed by the step type, and when a finding
    warns.

    Example YAML::

        workflows:
          - name: bias
            type: data-bias
            metadata: standard
            factor-gaps: {mi_threshold: 0.2}
            checks:
              shortcut-risk: {warning: 0.2}
    """

    type: str = Field(default="data-bias", description="The workflow type this entry configures: `data-bias`.")
    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.METADATA}), sources=SourceCount.ONE)

    diversity: DiversitySettings = Field(
        default_factory=DiversitySettings, description="The `diversity` step's settings."
    )
    factor_gaps: FactorGapsSettings | Literal[False] = Field(
        default_factory=FactorGapsSettings,
        alias="factor-gaps",
        description="The `factor-gaps` step's settings; `false` leaves out the gap analysis and its check.",
    )
    checks: DataBiasChecks = Field(
        default_factory=DataBiasChecks, description="When findings warn, keyed by check type."
    )
