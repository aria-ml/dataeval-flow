"""The ``metadata-triage`` workflow's config."""

from typing import ClassVar

from pydantic import Field

from dataeval_flow._input_spec import InputKind, InputSpec, SourceCount
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin
from dataeval_flow.workflows._base import WorkflowConfig
from dataeval_flow.workflows.metadata_triage._outputs import MetadataTriageResult

__all__ = ["MetadataTriageConfig"]


class MetadataTriageConfig(WorkflowConfig[MetadataTriageResult], MetadataConfigMixin):
    """The settings of one ``metadata-triage`` entry: what counts as a finding, what it shows, and whether it verifies.

    Deliberately thin.  There is no ``suggest`` toggle — suggesting is what the workflow is
    for — and no category filter, because severity is what a reader filters on and a
    category nobody wants is a section they skip.

    No extractor, no embeddings and no statistics unless the policy declares
    ``intrinsic_factors``: this needs the metadata walk and nothing else, which makes it the
    cheapest workflow in the suite and the natural first task in a pipeline.

    Example YAML::

        workflows:
          - name: triage
            type: metadata-triage
            metadata: standard
            max_examples: 20
    """

    type: str = Field(
        default="metadata-triage", description="The workflow type this entry configures: `metadata-triage`."
    )

    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.METADATA}), sources=SourceCount.ONE)

    max_examples: int = Field(
        default=20,
        ge=1,
        description=(
            "Distinct values shown per kind per factor in the report. Display only — a "
            "suggested correction always enumerates every value, because one covering a "
            "truncated set would read as complete and not be."
        ),
    )
    verify: bool = Field(
        default=True,
        description=(
            "Re-read the metadata under the complete suggestions and report what they "
            "recover. Costs no second dataset walk: `repair` returns a copy sharing the store."
        ),
    )
    default_bins: int = Field(
        default=10,
        ge=2,
        description=(
            "Bin count a suggestion falls back to where the run left no fit to read. Where "
            "there is one, the populated bins of the derived cut are carried forward instead, "
            "which pins the cut the run used rather than substituting a different one."
        ),
    )
    min_missing_fraction: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Share of rows recording no value above which a factor is called degenerate.",
    )
