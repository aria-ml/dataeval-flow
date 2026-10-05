"""The ``ood-detection`` preset's config: its detectors, its factor steps, and when a finding warns."""

__all__ = ["FactorDeviationSettings", "OODDetectionConfig", "OODDetectionChecks", "evaluator_entry"]

from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, SerializeAsAny, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin, StatsConfigMixin
from dataeval_flow.evaluators.shift import OODDomainClassifierConfig, OODKNeighborsConfig
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps.checks._ood import OODThresholds
from dataeval_flow.workflows._base import WorkflowConfig

_BASES: dict[str, type[BaseModel]] = {
    "ood-kneighbors": OODKNeighborsConfig,
    "ood-domain-classifier": OODDomainClassifierConfig,
}
_EXTRACTOR = "An `extractors:` entry this detector's steps embed with, instead of the task's."
_RESERVED = ("agreement", "factor-predictors", "factor-deviation")


class OODKNeighborsDetector(OODKNeighborsConfig):
    """An `ood-kneighbors` detector: its evaluator entry, and the extractor its steps embed with."""

    extractor: str | None = Field(default=None, description=_EXTRACTOR)


class OODDomainClassifierDetector(OODDomainClassifierConfig):
    """An `ood-domain-classifier` detector: its evaluator entry, and the extractor its steps embed with."""

    extractor: str | None = Field(default=None, description=_EXTRACTOR)


_DETECTORS: dict[str, type[BaseModel]] = {
    "ood-kneighbors": OODKNeighborsDetector,
    "ood-domain-classifier": OODDomainClassifierDetector,
}


def evaluator_entry(detector: Any) -> Any:
    """A detector's evaluator entry: its OOD config without `extractor`, which DataEval's detectors would take as
    their own argument."""
    return _BASES[detector.type].model_validate(detector.model_dump(exclude={"extractor"}))


def _detector_entry(entry: Any) -> Any:
    """One `detectors:` item, validated with the OOD evaluator config its `type` names."""
    type_id = entry.get("type") if isinstance(entry, Mapping) else getattr(entry, "type", None)
    if not isinstance(type_id, str):
        legacy = " Legacy's `method: kneighbors` is now `type: ood-kneighbors`: see the CHANGELOG for every rename."
        raise ValueError(
            f"Each detector needs a `type`, one of {', '.join(_DETECTORS)}."
            + (legacy if isinstance(entry, Mapping) and "method" in entry else "")
        )
    if type_id not in _DETECTORS:
        raise ValueError(f"`detectors:` takes {', '.join(_DETECTORS)} entries, not `{type_id}`.")
    return _DETECTORS[type_id].model_validate(entry if isinstance(entry, Mapping) else entry.model_dump())


# Typed callers may pass an OOD config; the validator turns either into its detector entry, and the detector members
# carry `extractor` into the schema.
OODDetector = Annotated[
    SerializeAsAny[
        OODKNeighborsDetector | OODDomainClassifierDetector | OODKNeighborsConfig | OODDomainClassifierConfig
    ],
    BeforeValidator(_detector_entry),
]


class OODDetectionChecks(BaseModel):
    """When ood-detection's findings warn: each check's fields, keyed by check type."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    ood: OODThresholds = Field(
        default_factory=OODThresholds, description="The `ood` check's thresholds, applied to each detector's check."
    )
    ood_agreement: OODThresholds = Field(
        default_factory=OODThresholds,
        alias="ood-agreement",
        description=(
            "The `ood-agreement` check's thresholds, applied to the agreement findings. Its defaults are `ood`'s, as "
            "legacy judged both with one pair."
        ),
    )


class FactorDeviationSettings(BaseModel):
    """The `factor-deviation` step's own setting."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    max_items: int = Field(
        default=50,
        gt=0,
        description="The most out-of-distribution agreed images explained per test source.",
    )


class OODDetectionConfig(WorkflowConfig[ChainResult], MetadataConfigMixin, StatsConfigMixin):
    """The settings of one ``ood-detection`` entry: the detectors each test source is scored with, whether the metadata
    behind what they flag is explained, and when a finding warns.

    Example YAML::

        workflows:
          - name: ood
            type: ood-detection
            detectors:
              - {type: ood-kneighbors, k: 10}
              - {type: ood-domain-classifier, n_folds: 5}
            checks:
              ood: {warning: 10.0, info: 1.0}
    """

    type: str = Field(default="ood-detection", description="The workflow type this entry configures: `ood-detection`.")
    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.EMBEDDINGS}),
        optional=frozenset({InputKind.METADATA, InputKind.STATS}),
        sources=SourceCount.TWO_OR_MORE,
        detection_rows=True,
    )

    detectors: list[OODDetector] = Field(
        min_length=1,
        description=(
            "OOD evaluator entries (`ood-kneighbors`, `ood-domain-classifier`), each scoring every test source against "
            "the reference. An entry's `name` names its step. An entry may name its own `extractor:`."
        ),
    )
    factor_predictors: Literal[False] | None = Field(
        default=None,
        alias="factor-predictors",
        description="`false` leaves out the `factor-predictors` step; it takes no settings.",
    )
    factor_deviation: FactorDeviationSettings | Literal[False] = Field(
        default_factory=FactorDeviationSettings,
        alias="factor-deviation",
        description="The `factor-deviation` step's settings; `false` leaves it out.",
    )
    checks: OODDetectionChecks = Field(
        default_factory=OODDetectionChecks, description="When findings warn, keyed by check type."
    )

    @model_validator(mode="after")
    def _names(self) -> Self:
        names = [detector.name for detector in self.detectors]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(
                f"There are two detectors named {', '.join(f'`{n}`' for n in twice)}: give each a distinct `name`."
            )
        reserved = [name for name in names if name in _RESERVED or name.endswith("-check")]
        if reserved:
            raise ValueError(
                f"Detector {', '.join(f'`{n}`' for n in reserved)} is a name the preset's own steps use (`agreement`, "
                "`factor-predictors`, `factor-deviation`, or a name ending in `-check`): rename it."
            )
        return self
