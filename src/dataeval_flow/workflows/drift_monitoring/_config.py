"""The ``drift-monitoring`` preset's config: its detectors, which run by class, and when a finding warns."""

__all__ = ["DriftMonitoringConfig", "DriftMonitoringThresholds"]

from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, SerializeAsAny, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.evaluators.shift import (
    DriftDomainClassifierConfig,
    DriftKNeighborsConfig,
    DriftMMDConfig,
    DriftUnivariateConfig,
)
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps.checks._drift import DriftThresholds
from dataeval_flow.workflows._base import WorkflowConfig

_DETECTORS: dict[str, type[BaseModel]] = {
    "drift-univariate": DriftUnivariateConfig,
    "drift-mmd": DriftMMDConfig,
    "drift-kneighbors": DriftKNeighborsConfig,
    "drift-domain-classifier": DriftDomainClassifierConfig,
}
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
    return _DETECTORS[type_id].model_validate(entry) if isinstance(entry, Mapping) else entry


DriftDetector = Annotated[
    SerializeAsAny[DriftUnivariateConfig | DriftMMDConfig | DriftKNeighborsConfig | DriftDomainClassifierConfig],
    BeforeValidator(_detector_entry),
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
            classwise: [drift-mmd]
    """

    type: str = Field(
        default="drift-monitoring", description="The workflow type this entry configures: `drift-monitoring`."
    )
    inputs: ClassVar[InputSpec] = InputSpec(
        required=frozenset({InputKind.EMBEDDINGS}),
        optional=frozenset({InputKind.LABELS}),
        sources=SourceCount.TWO_OR_MORE,
    )
    detectors: list[DriftDetector] = Field(
        min_length=1,
        description=(
            "Drift evaluator entries (`drift-univariate`, `drift-mmd`, `drift-kneighbors`, `drift-domain-classifier`), "
            "each tested on every test source against the reference. An entry's `name` names its step."
        ),
    )
    classwise: list[str] = Field(
        default_factory=list,
        description="Names of detectors to also run per class, unchunked, on classes with 2 or more items each side.",
    )
    health_thresholds: DriftMonitoringThresholds = Field(
        default_factory=DriftMonitoringThresholds, description="When findings warn, keyed by check type."
    )

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
