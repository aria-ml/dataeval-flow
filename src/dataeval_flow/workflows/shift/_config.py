"""The ``shift`` preset's config: its drift and OOD detectors, which drift detectors also run by class, its factor
steps, and when a finding warns."""

__all__ = ["FactorDeviationSettings", "ShiftChecks", "ShiftConfig", "evaluator_entry", "is_drift"]

from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, SerializeAsAny, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin, StatsConfigMixin
from dataeval_flow.evaluators.shift import (
    DriftDomainClassifierConfig,
    DriftKNeighborsConfig,
    DriftMMDConfig,
    DriftUnivariateConfig,
    OODDomainClassifierConfig,
    OODKNeighborsConfig,
)
from dataeval_flow.steps._by import ByConfig
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps.checks._drift import DriftThresholds
from dataeval_flow.steps.checks._ood import OODThresholds
from dataeval_flow.workflows._base import WorkflowConfig

_BASES: dict[str, type[BaseModel]] = {
    "drift-univariate": DriftUnivariateConfig,
    "drift-mmd": DriftMMDConfig,
    "drift-kneighbors": DriftKNeighborsConfig,
    "drift-domain-classifier": DriftDomainClassifierConfig,
    "ood-kneighbors": OODKNeighborsConfig,
    "ood-domain-classifier": OODDomainClassifierConfig,
}
_EXTRACTOR = "An `extractors:` entry this detector's steps embed with, instead of the task's."
_RESERVED_NAMES = ("ood-union", "ood-agreement", "factor-predictors", "factor-deviation")
_RESERVED_SUFFIXES = ("-check", "-by-class", "-unchunked")


class DriftUnivariateDetector(DriftUnivariateConfig):
    """A `drift-univariate` detector: its evaluator entry, and the extractor its steps embed with."""

    extractor: str | None = Field(default=None, description=_EXTRACTOR)


class DriftMMDDetector(DriftMMDConfig):
    """A `drift-mmd` detector: its evaluator entry, and the extractor its steps embed with."""

    extractor: str | None = Field(default=None, description=_EXTRACTOR)


class DriftKNeighborsDetector(DriftKNeighborsConfig):
    """A `drift-kneighbors` detector: its evaluator entry, and the extractor its steps embed with."""

    extractor: str | None = Field(default=None, description=_EXTRACTOR)


class DriftDomainClassifierDetector(DriftDomainClassifierConfig):
    """A `drift-domain-classifier` detector: its evaluator entry, and the extractor its steps embed with."""

    extractor: str | None = Field(default=None, description=_EXTRACTOR)


class OODKNeighborsDetector(OODKNeighborsConfig):
    """An `ood-kneighbors` detector: its evaluator entry, and the extractor its steps embed with."""

    extractor: str | None = Field(default=None, description=_EXTRACTOR)


class OODDomainClassifierDetector(OODDomainClassifierConfig):
    """An `ood-domain-classifier` detector: its evaluator entry, and the extractor its steps embed with."""

    extractor: str | None = Field(default=None, description=_EXTRACTOR)


_DETECTORS: dict[str, type[BaseModel]] = {
    "drift-univariate": DriftUnivariateDetector,
    "drift-mmd": DriftMMDDetector,
    "drift-kneighbors": DriftKNeighborsDetector,
    "drift-domain-classifier": DriftDomainClassifierDetector,
    "ood-kneighbors": OODKNeighborsDetector,
    "ood-domain-classifier": OODDomainClassifierDetector,
}


def is_drift(detector: Any) -> bool:
    """Whether `detector` is a drift detector, judged by `drift`; any other is an OOD detector, judged by `ood`."""
    return detector.type.startswith("drift-")


def evaluator_entry(detector: Any) -> Any:
    """A detector's evaluator entry: its config without `extractor`, which DataEval's detectors would take as their
    own argument."""
    return _BASES[detector.type].model_validate(detector.model_dump(exclude={"extractor"}))


def _detector_entry(entry: Any) -> Any:
    """One `detectors:` item, validated with the evaluator config its `type` names."""
    type_id = entry.get("type") if isinstance(entry, Mapping) else getattr(entry, "type", None)
    if not isinstance(type_id, str):
        raise ValueError(f"Each detector needs a `type`, one of {', '.join(_DETECTORS)}.")
    if type_id == "drift-wasserstein":
        raise ValueError(
            "`drift-wasserstein` needs a validation set, a third source shift does not take: run it as a step of a "
            "custom workflow."
        )
    if type_id not in _DETECTORS:
        raise ValueError(f"`detectors:` takes {', '.join(_DETECTORS)} entries, not `{type_id}`.")
    return _DETECTORS[type_id].model_validate(entry if isinstance(entry, Mapping) else entry.model_dump())


# Typed callers may pass an evaluator config; the validator turns either into its detector entry, and the detector
# members carry `extractor` into the schema.
ShiftDetector = Annotated[
    SerializeAsAny[
        DriftUnivariateDetector
        | DriftMMDDetector
        | DriftKNeighborsDetector
        | DriftDomainClassifierDetector
        | OODKNeighborsDetector
        | OODDomainClassifierDetector
        | DriftUnivariateConfig
        | DriftMMDConfig
        | DriftKNeighborsConfig
        | DriftDomainClassifierConfig
        | OODKNeighborsConfig
        | OODDomainClassifierConfig
    ],
    BeforeValidator(_detector_entry),
]


def _default_detectors() -> list[Any]:
    """`drift-univariate` and `ood-kneighbors` with DataEval's settings: a KS test per embedding dimension, with
    Bonferroni correction, and each test item's distance against the reference's 95th percentile."""
    return [_detector_entry({"type": "drift-univariate"}), _detector_entry({"type": "ood-kneighbors"})]


class ShiftChecks(BaseModel):
    """When shift's findings warn: each check's fields, keyed by check type."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    drift: DriftThresholds = Field(
        default_factory=DriftThresholds,
        description="The `drift` check's thresholds, whole-set and by class, applied to each drift detector's checks.",
    )
    ood: OODThresholds = Field(
        default_factory=OODThresholds, description="The `ood` check's thresholds, applied to each OOD detector's check."
    )
    ood_agreement: OODThresholds = Field(
        default_factory=OODThresholds,
        alias="ood-agreement",
        description="The `ood-agreement` check's thresholds, applied to the agreement findings.",
    )


class FactorDeviationSettings(BaseModel):
    """The `factor-deviation` step's own setting."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    max_items: int = Field(
        default=50, gt=0, description="The most out-of-distribution agreed images explained per test source."
    )


class ShiftConfig(WorkflowConfig[ChainResult], MetadataConfigMixin, StatsConfigMixin):
    """The settings of one ``shift`` entry: the drift and OOD detectors each test source is tested with against the
    reference, which drift detectors also run by class, whether the metadata behind OOD flags is explained, and when a
    finding warns.

    Example YAML::

        workflows:
          - name: shift
            type: shift
            detectors:
              - {type: drift-univariate}
              - {name: mmd, type: drift-mmd, chunking: {chunk_count: 5}}
              - {type: ood-kneighbors, k: 10}
            classwise:
              mmd: class
    """

    type: str = Field(default="shift", description="The workflow type this entry configures: `shift`.")
    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.EMBEDDINGS}),
        optional=frozenset({InputKind.LABELS, InputKind.METADATA, InputKind.STATS}),
        sources=SourceCount.TWO_OR_MORE,
        detection_rows=True,
    )
    detectors: list[ShiftDetector] = Field(
        default_factory=_default_detectors,
        min_length=1,
        description=(
            "Drift (`drift-univariate`, `drift-mmd`, `drift-kneighbors`, `drift-domain-classifier`) and OOD "
            "(`ood-kneighbors`, `ood-domain-classifier`) evaluator entries, each testing every test source against "
            "the reference. An entry's `name` names its step, and it may name its own `extractor:`. Unset is "
            "`drift-univariate` and `ood-kneighbors`."
        ),
    )
    classwise: dict[str, ByConfig] = Field(
        default_factory=dict,
        description=(
            "Drift detectors to also run per key, unchunked, each with its `by:`: `{drift-mmd: class}`, "
            "`{uncertainty: predicted}`, or with settings; `min_items` is 2 unless written."
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
    checks: ShiftChecks = Field(default_factory=ShiftChecks, description="When findings warn, keyed by check type.")

    @model_validator(mode="after")
    def _names(self) -> Self:
        names = [detector.name for detector in self.detectors]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(
                f"There are two detectors named {', '.join(f'`{n}`' for n in twice)}: give each a distinct `name`."
            )
        reserved = [name for name in names if name in _RESERVED_NAMES or name.endswith(_RESERVED_SUFFIXES)]
        if reserved:
            raise ValueError(
                f"Detector {', '.join(f'`{n}`' for n in reserved)} is a name the preset's own steps use "
                f"({', '.join(f'`{n}`' for n in _RESERVED_NAMES)}, or a name ending in "
                f"{' or '.join(f'`{s}`' for s in _RESERVED_SUFFIXES)}): rename it."
            )
        unknown = [name for name in self.classwise if name not in names]
        if unknown:
            raise ValueError(f"`classwise` names {', '.join(f'`{n}`' for n in unknown)}, which no detector is.")
        ood = [
            detector.name for detector in self.detectors if detector.name in self.classwise and not is_drift(detector)
        ]
        if ood:
            raise ValueError(
                f"`classwise` names {', '.join(f'`{n}`' for n in ood)}, an OOD detector: only drift detectors run "
                "by class."
            )
        return self
