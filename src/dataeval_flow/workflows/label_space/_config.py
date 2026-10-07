"""The ``label-space`` preset's config: the ontology, the minimum shares and label pattern, and when a finding warns
(coverage spec §3.1)."""

__all__ = [
    "LabelConformanceSettings",
    "LabelSpaceChecks",
    "LabelSpaceConfig",
    "LeafCoverageSettings",
    "OntologyValidationSettings",
]

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.workflows._base import WorkflowConfig
from dataeval_flow.workflows.data_coverage import RepresentationSettings


class LeafCoverageSettings(BaseModel):
    """The `leaf-coverage` check's fields, with legacy data-coverage's defaults."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    coverage: float | None = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description=("The least share of the ontology's leaves with examples; `null` turns it off."),
    )
    empty_branches: int | None = Field(
        default=0,
        ge=0,
        description="Wholly empty branches tolerated; `null` turns it off.",
    )


class LabelConformanceSettings(BaseModel):
    """The `label-conformance` check's field, with legacy data-coverage's default."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    warning: int | None = Field(
        default=0,
        ge=0,
        description="Class names that may resolve to no concept; `null` turns it off.",
    )


class OntologyValidationSettings(BaseModel):
    """The `ontology-validation` step's settings."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    label_pattern: str | None = Field(default=None, description="A regex every ontology label should match.")


class LabelSpaceChecks(BaseModel):
    """When label-space's findings warn: each check's fields, keyed by check type."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    leaf_coverage: LeafCoverageSettings = Field(
        default_factory=LeafCoverageSettings,
        alias="leaf-coverage",
        description="The `leaf-coverage` check's thresholds.",
    )
    label_conformance: LabelConformanceSettings = Field(
        default_factory=LabelConformanceSettings,
        alias="label-conformance",
        description="The `label-conformance` check's threshold.",
    )


class LabelSpaceConfig(WorkflowConfig[ChainResult]):
    """The settings of one ``label-space`` entry: the ontology to judge labels against, and when a finding warns.

    Example YAML::

        workflows:
          - name: vocab
            type: label-space
            ontology: vehicles
            representation: {expected: {truck: 0.2}}
    """

    type: str = Field(default="label-space", description="The workflow type this entry configures: `label-space`.")
    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True, serialize_by_alias=True)
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.LABELS}), sources=SourceCount.ONE)

    # Overridden to be required, which the schema then shows; the base declares it optional.
    ontology: dict[str, Any] | str = Field(  # pyright: ignore[reportIncompatibleVariableOverride, reportGeneralTypeIssues]
        description=(
            "The ontology to judge labels against: a name under the top-level `ontologies:` key, a path to a "
            "serialized RDF artifact resolved against the data root, or a nested mapping of concept to children."
        ),
    )
    representation: RepresentationSettings = Field(
        default_factory=RepresentationSettings, description="The `representation` step's settings."
    )
    ontology_validation: OntologyValidationSettings = Field(
        default_factory=OntologyValidationSettings,
        alias="ontology-validation",
        description="The `ontology-validation` step's settings.",
    )
    checks: LabelSpaceChecks = Field(
        default_factory=LabelSpaceChecks, description="When findings warn, keyed by check type."
    )

    @model_validator(mode="before")
    @classmethod
    def _ontology_named(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("ontology") is None:
            raise ValueError("`label-space` judges labels against an ontology: name one with `ontology:`.")
        return data
