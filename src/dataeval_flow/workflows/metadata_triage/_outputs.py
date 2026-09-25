"""Metadata triage workflow outputs."""

from typing import Any

from pydantic import BaseModel, Field

from dataeval_flow._result import ResultMetadata
from dataeval_flow._triage import TriageFinding
from dataeval_flow.workflows._base import WorkflowOutput, WorkflowRawOutput, WorkflowReport
from dataeval_flow.workflows._result import WorkflowResult

__all__ = [
    "MetadataTriageMetadata",
    "MetadataTriageOutput",
    "MetadataTriageRawOutput",
    "MetadataTriageReport",
    "MetadataTriageResult",
    "VerificationEntry",
]


class VerificationEntry(BaseModel):
    """What one suggestion actually did when it was read back.

    ``recovered`` is the question worth asking.  A suggestion can be well-formed, run
    cleanly and still not work — upstream is explicit that a reading leaving every row
    holding its own value has not made the column a factor — so a stanza is worth checking
    before it is committed to a config.
    """

    factor: str
    applied: bool
    recovered: bool
    detail: str


class MetadataTriageRawOutput(WorkflowRawOutput):
    """Machine-readable triage results."""

    findings: list[TriageFinding] = Field(
        default_factory=list,
        description=(
            "Each issue found in how the metadata was read, worst first: its `factor`, `category`, `severity`, "
            "`reasons` and `remedy`, and a `suggestion` where one repairs it."
        ),
    )
    suggested_policy: dict[str, Any] = Field(
        default_factory=dict, description="Every suggestion merged into one metadata policy body."
    )
    suggested_policy_yaml: str = Field(
        default="", description="`suggested_policy` as YAML, ready to paste under a config's `metadata:` key."
    )
    verification: list[VerificationEntry] = Field(
        default_factory=list,
        description=(
            "Per suggestion, whether reading the metadata back under it `applied` and `recovered` the factor. "
            "Empty unless `verify` is on."
        ),
    )
    verification_error: str | None = Field(
        default=None,
        description=(
            "Set when verification was attempted and raised, rather than left empty the way "
            "`verify: false` or nothing to verify both leave it. A reader cannot otherwise "
            "tell those three states apart."
        ),
    )
    counts: dict[str, int] = Field(
        default_factory=dict, description="How many findings fall in each category and severity."
    )
    factor_count: int = Field(default=0, description="How many metadata factors the run read.")


class MetadataTriageReport(WorkflowReport):
    """Human-readable triage report."""


class MetadataTriageOutput(WorkflowOutput[MetadataTriageRawOutput, MetadataTriageReport]):
    """Complete metadata triage workflow output."""


class MetadataTriageMetadata(ResultMetadata):
    """Metadata for the metadata-triage workflow."""

    blocking: int = Field(default=0, description="How many findings are `blocking`.")
    verified: int = Field(default=0, description="How many suggestions verification showed recovered their factor.")


class MetadataTriageResult(WorkflowResult[MetadataTriageMetadata, MetadataTriageOutput]):
    """The result of a ``metadata-triage`` run.

    ``isinstance`` narrows a :class:`~dataeval_flow.Result` to it, which types ``output`` and ``metadata`` with the
    fields below; ``output`` is readable only where ``success`` is true. Every workflow result also carries
    ``output.raw.dataset_size`` (:class:`~dataeval_flow.workflows.WorkflowRawOutput`), ``output.report.summary`` and
    ``output.report.findings`` (:class:`~dataeval_flow.workflows.WorkflowReport`), and the envelope fields of
    :class:`~dataeval_flow.ResultMetadata`.

    Fields
    ------
    output.raw.findings
        Each issue found in how the metadata was read, worst first: its ``factor``, ``category``, ``severity``,
        ``reasons`` and ``remedy``, and a ``suggestion`` where one repairs it.
    output.raw.suggested_policy
        Every suggestion merged into one metadata policy body.
    output.raw.suggested_policy_yaml
        ``suggested_policy`` as YAML, ready to paste under a config's ``metadata:`` key.
    output.raw.verification
        Per suggestion, whether reading the metadata back under it ``applied`` and ``recovered`` the factor. Empty
        unless ``verify`` is on.
    output.raw.verification_error
        Set when verification was attempted and raised, rather than left empty the way ``verify: false`` or nothing to
        verify both leave it. A reader cannot otherwise tell those three states apart.
    output.raw.counts
        How many findings fall in each category and severity.
    output.raw.factor_count
        How many metadata factors the run read.
    metadata.blocking
        How many findings are ``blocking``.
    metadata.verified
        How many suggestions verification showed recovered their factor.
    """
