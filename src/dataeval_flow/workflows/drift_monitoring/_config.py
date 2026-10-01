"""The ``drift-monitoring`` preset's config: its detectors, which run by class, and when a finding warns."""

__all__ = ["DriftMonitoringConfig", "DriftMonitoringThresholds", "evaluator_entry"]

from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, SerializeAsAny, field_validator, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.evaluators.shift import (
    DriftDomainClassifierConfig,
    DriftKNeighborsConfig,
    DriftMMDConfig,
    DriftUnivariateConfig,
)
from dataeval_flow.steps._by import ByConfig
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps.checks._drift import DriftThresholds
from dataeval_flow.workflows._base import WorkflowConfig

_BASES: dict[str, type[BaseModel]] = {
    "drift-univariate": DriftUnivariateConfig,
    "drift-mmd": DriftMMDConfig,
    "drift-kneighbors": DriftKNeighborsConfig,
    "drift-domain-classifier": DriftDomainClassifierConfig,
}
_EXTRACTOR = "An `extractors:` entry this detector's steps embed with, instead of the task's."


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


_DETECTORS: dict[str, type[BaseModel]] = {
    "drift-univariate": DriftUnivariateDetector,
    "drift-mmd": DriftMMDDetector,
    "drift-kneighbors": DriftKNeighborsDetector,
    "drift-domain-classifier": DriftDomainClassifierDetector,
}


def evaluator_entry(detector: Any) -> Any:
    """A detector's evaluator entry: its drift config without `extractor`, which DataEval's drift detectors would
    take as their own argument."""
    return _BASES[detector.type].model_validate(detector.model_dump(exclude={"extractor"}))


_RESERVED = ("-check", "-classes", "-unchunked")


def _detector_entry(entry: Any) -> Any:
    """One `detectors:` item, validated with the drift evaluator config its `type` names."""
    type_id = entry.get("type") if isinstance(entry, Mapping) else getattr(entry, "type", None)
    if not isinstance(type_id, str):
        legacy = " Legacy's `method: mmd` is now `type: drift-mmd`: see the CHANGELOG for every rename."
        raise ValueError(
            f"Each detector needs a `type`, one of {', '.join(_DETECTORS)}."
            + (legacy if isinstance(entry, Mapping) and "method" in entry else "")
        )
    if type_id == "drift-wasserstein":
        raise ValueError(
            "`drift-wasserstein` needs a validation set, a third source drift-monitoring does not take: run it as a "
            "step of a custom workflow."
        )
    if type_id not in _DETECTORS:
        raise ValueError(f"`detectors:` takes {', '.join(_DETECTORS)} entries, not `{type_id}`.")
    return _DETECTORS[type_id].model_validate(entry if isinstance(entry, Mapping) else entry.model_dump())


DriftDetector = Annotated[
    SerializeAsAny[DriftUnivariateConfig | DriftMMDConfig | DriftKNeighborsConfig | DriftDomainClassifierConfig],
    BeforeValidator(
        _detector_entry,
        json_schema_input_type=(
            DriftUnivariateDetector | DriftMMDDetector | DriftKNeighborsDetector | DriftDomainClassifierDetector
        ),
    ),
]


class DriftMonitoringThresholds(BaseModel):
    """When drift-monitoring's findings warn: the `drift` check's fields, applied to every detector's checks."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    drift: DriftThresholds = Field(
        default_factory=DriftThresholds, description="The `drift` check's thresholds, whole-set and by class."
    )


class DriftMonitoringConfig(WorkflowConfig[ChainResult]):
    """The settings of one ``drift-monitoring`` entry: the detectors each test source is tested with, which also run
    by class, and when a finding warns.

    Example YAML::

        workflows:
          - name: drift
            type: drift-monitoring
            detectors:
              - {type: drift-mmd}
              - {name: ks, type: drift-univariate, chunking: {chunk_count: 5}}
            classwise:
              drift-mmd: class
    """

    type: str = Field(
        default="drift-monitoring", description="The workflow type this entry configures: `drift-monitoring`."
    )
    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.EMBEDDINGS}),
        optional=frozenset({InputKind.LABELS}),
        sources=SourceCount.TWO_OR_MORE,
        detection_rows=True,
    )
    detectors: list[DriftDetector] = Field(
        min_length=1,
        description=(
            "Drift evaluator entries (`drift-univariate`, `drift-mmd`, `drift-kneighbors`, `drift-domain-classifier`), "
            "each tested on every test source against the reference. An entry's `name` names its step. "
            "An entry may name its own `extractor:`."
        ),
    )
    # Values also accept "class" / "predicted" strings, validated into ByConfig.
    classwise: dict[str, ByConfig] = Field(
        default_factory=dict,
        description=(
            "Detectors to also run per key, unchunked, each with its `by:`: `{drift-mmd: class}`, "
            "`{uncertainty: predicted}`, or with settings; `min_items` is 2 unless written."
        ),
    )
    health_thresholds: DriftMonitoringThresholds = Field(
        default_factory=DriftMonitoringThresholds, description="When findings warn, keyed by check type."
    )

    @field_validator("classwise", mode="before")
    @classmethod
    def _mapping(cls, value: Any) -> Any:
        if isinstance(value, list):
            raise ValueError(
                "`classwise:` maps each detector to its `by:`, such as `{drift-mmd: class}`; the list form is gone."
            )
        return value

    @model_validator(mode="after")
    def _names(self) -> Self:
        names = [detector.name for detector in self.detectors]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(
                f"There are two detectors named {', '.join(f'`{n}`' for n in twice)}: give each a distinct `name`."
            )
        reserved = [name for name in names if name.endswith(_RESERVED)]
        if reserved:
            raise ValueError(
                f"Detector {', '.join(f'`{n}`' for n in reserved)} ends in {' or '.join(_RESERVED)}, which the "
                "preset's own steps use: rename it."
            )
        unknown = [name for name in self.classwise if name not in names]
        if unknown:
            raise ValueError(f"`classwise` names {', '.join(f'`{n}`' for n in unknown)}, which no detector is.")
        return self
