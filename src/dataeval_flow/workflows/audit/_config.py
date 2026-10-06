"""The ``audit`` preset's config: the settings of its step types, when a finding warns, and the verdict's criteria
(audit spec §6, §8)."""

__all__ = [
    "AuditChecks",
    "AuditClassImbalanceSettings",
    "AuditConfig",
    "ClassSufficiencySettings",
    "DistributionShiftSettings",
    "DivergenceSettings",
    "EvalCoverageSettings",
    "FactorLeakageSettings",
    "LeakageSettings",
    "OODKNeighborsSettings",
    "UntrainedClassesSettings",
]

from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin, StatsConfigMixin
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps.checks._ood import OODThresholds
from dataeval_flow.steps.checks._stratification import StratificationThresholds
from dataeval_flow.workflows._base import WorkflowConfig
from dataeval_flow.workflows.data_bias import DiversitySettings, FactorGapsSettings, ShortcutRiskSettings
from dataeval_flow.workflows.data_bias._config import FactorCoverageGapsSettings
from dataeval_flow.workflows.data_cleaning import ImageDuplicatesSettings, ImageOutliersSettings, OutliersSettings
from dataeval_flow.workflows.data_coverage import DataCoverageCoverageSettings, WrapSettings
from dataeval_flow.workflows.data_coverage._config import (
    ClassCoverageSettings,
    DataCoverageUncoveredItemsSettings,
    DimensionalCompletenessSettings,
)
from dataeval_flow.workflows.label_space import LabelConformanceSettings
from dataeval_flow.workflows.metadata_triage import MetadataIssuesSettings


class FactorLeakageSettings(BaseModel):
    """The `factor-leakage` step's settings: the group factors whose values must stay in one split."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    factors: list[str] = Field(
        min_length=1, description="The group factors, such as a scene or site, whose values must not sit in two splits."
    )


class DivergenceSettings(BaseModel):
    """The `divergence` step's settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    method: Literal["mst", "fnn"] = Field(
        default="mst",
        description=(
            "How divergence is counted: over the minimum spanning tree of both splits, or by nearest-neighbour "
            "disagreement."
        ),
    )


class OODKNeighborsSettings(BaseModel):
    """The `ood-kneighbors` step's settings, fitted on train and run on each evaluation split."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    k: int | None = Field(
        default=None, gt=0, description="Neighbours each evaluation item is compared with; unset is DataEval's default."
    )
    distance_metric: Literal["cosine", "euclidean"] | None = Field(
        default=None, description="The embedding distance; unset is DataEval's default."
    )
    threshold_perc: float = Field(
        default=99.0,
        gt=0.0,
        lt=100.0,
        description="An item is flagged when it lies farther from train than this percent of train lies from itself.",
    )


class AuditClassImbalanceSettings(BaseModel):
    """The `class-imbalance` check's settings in audit, judged on each split. Named for the preset, so it reaches the
    schema `$defs` apart from the other presets' class-imbalance settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: float | None = Field(
        default=5.0,
        ge=1.0,
        description=(
            "Largest class count over smallest, among the classes with labels, past which the finding warns; `null` "
            "judges nothing but an empty class, which warns unless `empty` is false."
        ),
    )
    info: float | None = Field(
        default=None,
        ge=1.0,
        description=(
            "A ratio at or under which the finding is ok, between which and `warning` it informs; `null` makes every "
            "ratio under `warning` information. Must not exceed `warning`."
        ),
    )
    empty: bool = Field(
        default=False,
        description=(
            "Whether a declared class with no labels warns; audit leaves that to `untrained-classes` and "
            "`class-sufficiency`."
        ),
    )

    @model_validator(mode="after")
    def _info_under_warning(self) -> Self:
        """Two bounds that cross are refused here, where the user wrote them."""
        if self.info is not None and self.warning is not None and self.info > self.warning:
            raise ValueError(f"`info` ({self.info}) must not exceed `warning` ({self.warning}).")
        return self


class ClassSufficiencySettings(BaseModel):
    """The `class-sufficiency` check's settings: the fewest labels a class needs to learn and to evaluate."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    train: int | None = Field(
        default=20,
        ge=0,
        description="The fewest labels each class train holds needs in train; `null` judges no minimum there.",
    )
    eval: int | None = Field(
        default=30,
        ge=0,
        description=(
            "The fewest labels each class train holds needs in each evaluation split, a class the split lacks "
            "included. At 30, a per-class metric's 95% interval is about ±18 points."
        ),
    )


class UntrainedClassesSettings(BaseModel):
    """The `untrained-classes` check's settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    declared: bool = Field(
        default=False,
        description=(
            "Whether a declared class with no labels in train also warns. Off, it is listed: a vocabulary conformed "
            "to an ontology carries abstract concepts with no items."
        ),
    )


class LeakageSettings(BaseModel):
    """The `leakage` check's settings: how many items and group values may sit in two splits."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    exact: int | None = Field(
        default=0,
        ge=0,
        description=(
            "Most items, counted over every exact-duplicate group with members in two splits, that may leak before the "
            "finding warns; `null` judges nothing."
        ),
    )
    near: int | None = Field(default=0, ge=0, description="The same for near-duplicate groups; `null` judges nothing.")
    groups: int | None = Field(
        default=0,
        ge=0,
        description=(
            "Most group values (a factor's value held by both splits of a pair) that may be shared before the finding "
            "warns; `null` judges nothing."
        ),
    )


class EvalCoverageSettings(OODThresholds):
    """The `eval-coverage` check's settings: the share of an evaluation split that may lie beyond train."""

    info: float | None = Field(
        default=2.0,
        ge=0.0,
        le=100.0,
        description=(
            "The percent flagged past which the finding is `info`, at or below which it is `ok`; `null` is never "
            "`info`. A split drawn like train has about 100 - `threshold_perc` percent flagged by construction, so "
            "`2.0` suits `threshold_perc: 99`."
        ),
    )


class DistributionShiftSettings(BaseModel):
    """The `distribution-shift` check's settings. An unset `info` is derived here, as the check derives it, because
    the chain hands the check every setting, and a written `null` would mean no `info` band."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: float | None = Field(
        default=0.5, ge=0.0, le=1.0, description="The divergence above which the finding warns; `null` never warns."
    )
    info: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "The divergence above which the finding is `info`, at or below which it is `ok`. Unset, it is 0.4 times "
            "`warning`; `null` has no `info` band. Must not exceed `warning`."
        ),
    )

    @model_validator(mode="after")
    def _info_under_warning(self) -> Self:
        """An unset `info` is 0.4 times `warning`; two written bounds that cross are refused here."""
        if self.warning is None:
            return self
        if "info" not in self.model_fields_set:
            # derived, so it stays out of model_fields_set; rounded, because 0.4 * 0.2 is 0.08000000000000002
            object.__setattr__(self, "info", round(0.4 * self.warning, 10))
        elif self.info is not None and self.info > self.warning:
            raise ValueError(f"`info` ({self.info}) must not exceed `warning` ({self.warning}).")
        return self


class AuditChecks(BaseModel):
    """When audit's findings warn: each check's settings, keyed by check type. A check over each split applies its
    settings to every split."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    image_outliers: ImageOutliersSettings = Field(
        default_factory=ImageOutliersSettings,
        alias="image-outliers",
        description="The `image-outliers` check's settings.",
    )
    image_duplicates: ImageDuplicatesSettings = Field(
        default_factory=ImageDuplicatesSettings,
        alias="image-duplicates",
        description="The `image-duplicates` check's settings.",
    )
    metadata_issues: MetadataIssuesSettings = Field(
        default_factory=MetadataIssuesSettings,
        alias="metadata-issues",
        description="The `metadata-issues` check's settings.",
    )
    class_imbalance: AuditClassImbalanceSettings = Field(
        default_factory=AuditClassImbalanceSettings,
        alias="class-imbalance",
        description="The `class-imbalance` check's settings.",
    )
    class_sufficiency: ClassSufficiencySettings = Field(
        default_factory=ClassSufficiencySettings,
        alias="class-sufficiency",
        description="The `class-sufficiency` check's settings.",
    )
    untrained_classes: UntrainedClassesSettings = Field(
        default_factory=UntrainedClassesSettings,
        alias="untrained-classes",
        description="The `untrained-classes` check's settings.",
    )
    label_conformance: LabelConformanceSettings = Field(
        default_factory=LabelConformanceSettings,
        alias="label-conformance",
        description="The `label-conformance` check's settings, read where `ontology` is set.",
    )
    class_coverage: ClassCoverageSettings = Field(
        default_factory=ClassCoverageSettings,
        alias="class-coverage",
        description="The `class-coverage` check's settings.",
    )
    uncovered_items: DataCoverageUncoveredItemsSettings = Field(
        default_factory=DataCoverageUncoveredItemsSettings,
        alias="uncovered-items",
        description="The `uncovered-items` check's settings, under `naive` coverage.",
    )
    dimensional_completeness: DimensionalCompletenessSettings = Field(
        default_factory=DimensionalCompletenessSettings,
        alias="dimensional-completeness",
        description="The `dimensional-completeness` check's settings.",
    )
    factor_coverage_gaps: FactorCoverageGapsSettings = Field(
        default_factory=FactorCoverageGapsSettings,
        alias="factor-coverage-gaps",
        description="The `factor-coverage-gaps` check's settings.",
    )
    shortcut_risk: ShortcutRiskSettings = Field(
        default_factory=ShortcutRiskSettings,
        alias="shortcut-risk",
        description="The `shortcut-risk` check's settings.",
    )
    leakage: LeakageSettings = Field(default_factory=LeakageSettings, description="The `leakage` check's settings.")
    eval_coverage: EvalCoverageSettings = Field(
        default_factory=EvalCoverageSettings,
        alias="eval-coverage",
        description="The `eval-coverage` check's settings.",
    )
    stratification: StratificationThresholds = Field(
        default_factory=StratificationThresholds, description="The `stratification` check's settings."
    )
    distribution_shift: DistributionShiftSettings = Field(
        default_factory=DistributionShiftSettings,
        alias="distribution-shift",
        description="The `distribution-shift` check's settings.",
    )


_POLICY = "name a policy under `metadata:`"

MOVED: dict[str, str] = {
    "outlier_method": "it is `outliers.outlier_threshold`: the method, or `[method, threshold]`",
    "outlier_threshold": "it is `outliers.outlier_threshold`: the method, or `[method, threshold]`",
    "outlier_flags": "it is `outliers.flags`",
    "balance": "`balance` always runs, and is skipped on metadata with no factors",
    "diversity_method": "it is `diversity.method`",
    "divergence_method": "it is `divergence.method`",
    "include_image_stats": "it is the metadata policy's `intrinsic_factors`",
    "value_range": "set `value_range` on the dataset",
    "metadata_auto_bin_method": _POLICY,
    "metadata_exclude": _POLICY,
    "metadata_continuous_factor_bins": _POLICY,
    "metadata_factor_source": _POLICY,
}
"""Each data-analysis field audit refuses, and where it went (audit spec §8.2)."""

CHECKS_MOVED: dict[str, str] = {
    "image_outliers": "`checks.image-outliers.warning`",
    "exact_duplicates": "`checks.image-duplicates.exact`",
    "near_duplicates": "`checks.image-duplicates.near`",
    "class_label_imbalance": "`checks.class-imbalance.warning`",
    "distribution_shift": "`checks.distribution-shift.warning`",
}
"""Each data-analysis `health_thresholds` field, and the check setting it went to (audit spec §8.2)."""

_Reason = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AuditConfig(WorkflowConfig[ChainResult], MetadataConfigMixin, StatsConfigMixin):
    """The settings of one ``audit`` entry: each step type's settings, keyed by the step type, when a finding warns,
    and which warnings block or are accepted.

    Example YAML::

        workflows:
          - name: release_audit
            type: audit
            metadata: standard
            outliers: {flags: [dimension, pixel, visual], outlier_threshold: modzscore}
            factor-leakage: {factors: [scene]}
            accepted: {class-imbalance: "Rare class by design."}
            checks:
              leakage: {near: null}
    """

    type: str = Field(default="audit", description="The workflow type this entry configures: `audit`.")
    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.METADATA, InputKind.STATS}),
        optional=frozenset({InputKind.EMBEDDINGS}),
        sources=SourceCount.ONE_OR_MORE,
    )

    outliers: OutliersSettings = Field(description="The `outliers` step's settings, run on each split.")
    coverage: DataCoverageCoverageSettings = Field(
        default_factory=DataCoverageCoverageSettings,
        description="The `coverage` step's settings, run on train when the task names an extractor.",
    )
    wrap: WrapSettings = Field(
        default_factory=WrapSettings, description="The `wrap` step's settings, used on detection data only."
    )
    factor_gaps: FactorGapsSettings | Literal[False] = Field(
        default_factory=FactorGapsSettings,
        alias="factor-gaps",
        description="The `factor-gaps` step's settings; `false` leaves out the gap analysis and its check.",
    )
    factor_leakage: FactorLeakageSettings | None = Field(
        default=None,
        alias="factor-leakage",
        description="The `factor-leakage` step's settings; unset leaves group leakage out.",
    )
    diversity: DiversitySettings = Field(
        default_factory=DiversitySettings, description="The `diversity` step's settings."
    )
    divergence: DivergenceSettings = Field(
        default_factory=DivergenceSettings,
        description="The `divergence` step's settings, run when the task names an extractor.",
    )
    ood_kneighbors: OODKNeighborsSettings = Field(
        default_factory=OODKNeighborsSettings,
        alias="ood-kneighbors",
        description="The `ood-kneighbors` step's settings, run when the task names an extractor.",
    )
    blocking: list[str] = Field(
        default_factory=lambda: ["leakage", "untrained-classes"],
        description="The check types whose warning makes the verdict not ready, unless accepted.",
    )
    accepted: dict[str, _Reason] = Field(
        default_factory=dict,
        description=(
            "Why each warning is accepted, by check type (`image-outliers`), by check step for all its runs "
            "(`image-outliers-evals`), or by `step[split]` for one run of a step that runs once per evaluation split "
            "(`image-outliers-evals[test]`), as the verdict's `warnings[].step` names it: an accepted warning can't "
            "make the verdict not ready, but it still leaves it ready with caveats."
        ),
    )
    checks: AuditChecks = Field(default_factory=AuditChecks, description="When findings warn, keyed by check type.")

    @model_validator(mode="before")
    @classmethod
    def _refuse_legacy_fields(cls, data: Any) -> Any:
        """Refuse every data-analysis field by name, saying where it went (audit spec §8.2)."""
        if not isinstance(data, dict):
            return data
        for key, message in MOVED.items():
            if key in data:
                raise ValueError(f"audit's `{key}` is refused: {message}.")
        if "health_thresholds" in data:
            thresholds = data["health_thresholds"]
            for key, replacement in CHECKS_MOVED.items():
                if isinstance(thresholds, dict) and key in thresholds:
                    raise ValueError(f"`health_thresholds.{key}` is refused: it is {replacement}.")
            raise ValueError("`health_thresholds` is now `checks:`, keyed by check type.")
        return data

    @model_validator(mode="after")
    def _blocking_and_accepted_name_checks(self) -> Self:
        """Refuse a `blocking` entry or an `accepted` key that names no check in this entry's chain, and a
        `step[split]` key whose step doesn't run once per evaluation split."""
        from dataeval_flow.workflows.audit._workflow import PER_EVALUATION_SPLIT, AuditWorkflow

        steps = AuditWorkflow.chain(self).steps
        entries = [step for step in steps if isinstance(step, Mapping) and "check" in step]
        checks = {step["check"] for step in entries}
        check_steps = {step["name"] for step in entries}
        for key in self.blocking:
            if key not in checks:
                raise ValueError(
                    f"`blocking` names `{key}`, which this audit's chain has no check for. Its checks: "
                    f"{', '.join(sorted(checks))}."
                )
        for key in self.accepted:
            step = key.split("[", 1)[0]
            if step in check_steps and key != step and key.endswith("]") and step not in PER_EVALUATION_SPLIT:
                raise ValueError(
                    f"`accepted` names `{key}`, but `{step}` runs once, not once per evaluation split; key it "
                    f"`{step}` alone. The check steps that take `[split]`: "
                    f"{', '.join(sorted(PER_EVALUATION_SPLIT & check_steps))}."
                )
            if key in checks or (step in check_steps and (key == step or key.endswith("]"))):
                continue
            raise ValueError(
                f"`accepted` names `{key}`, which this audit's chain has no check or check step for. Its checks: "
                f"{', '.join(sorted(checks))}. Its check steps: {', '.join(sorted(check_steps))}."
            )
        return self
