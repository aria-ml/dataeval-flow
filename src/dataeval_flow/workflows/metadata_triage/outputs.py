"""Metadata triage workflow outputs."""

from typing import Any

from pydantic import BaseModel, Field

from dataeval_flow.config.schemas import ResultMetadata
from dataeval_flow.triage import Finding
from dataeval_flow.workflow.base import Reportable, WorkflowOutputsBase, WorkflowReportBase

__all__ = [
    "MetadataTriageMetadata",
    "MetadataTriageOutputs",
    "MetadataTriageRawOutputs",
    "MetadataTriageReport",
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


class MetadataTriageRawOutputs(WorkflowOutputsBase):
    """Machine-readable triage results."""

    findings: list[Finding] = Field(default_factory=list)
    suggested_policy: dict[str, Any] = Field(default_factory=dict)
    suggested_policy_yaml: str = ""
    verification: list[VerificationEntry] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    factor_count: int = 0


class MetadataTriageReport(WorkflowReportBase):
    """Human-readable triage report."""

    findings: list[Reportable] = Field(default_factory=list)


class MetadataTriageOutputs(BaseModel):
    """Complete metadata triage workflow output."""

    raw: MetadataTriageRawOutputs
    report: MetadataTriageReport


class MetadataTriageMetadata(ResultMetadata):
    """Metadata for the metadata-triage workflow."""

    blocking: int = 0
    verified: int = 0
