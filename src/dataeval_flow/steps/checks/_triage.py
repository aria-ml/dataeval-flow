"""The metadata triage check: metadata-triage's findings, as a step (spec §10.10)."""

__all__ = ["MetadataIssuesCheck", "MetadataIssuesConfig"]

from collections.abc import Mapping
from typing import Any, ClassVar

from pydantic import Field

from dataeval_flow._triage_report import build_findings
from dataeval_flow.evaluators.quality._result import FactorTriageOutput
from dataeval_flow.steps._check import Check, CheckConfig, CheckContext
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.workflows._base import Finding


class MetadataIssuesConfig(CheckConfig):
    """A `metadata-issues` step's input, and how many of each factor's values its findings show."""

    input: str = Field(description="A `factor-triage` Output.")
    max_examples: int = Field(
        default=20,
        ge=1,
        description=(
            "Distinct values shown per kind per factor in a finding. Display only — a suggested correction always "
            "enumerates every value, because one covering a truncated set would read as complete and not be."
        ),
    )


class MetadataIssuesCheck(Check[MetadataIssuesConfig]):
    """``metadata-issues``: one finding per kind of issue factor-triage found, a warning where any of them is
    blocking; then the suggested policy, and what verification recovered or that it failed. It has no thresholds: an
    issue is blocking where the run did less than its configuration asked."""

    name: ClassVar[str] = "metadata-issues"
    description: ClassVar[str] = "Warns where metadata triage found a factor the run could not read as configured."
    title: ClassVar[str] = "Metadata Issues"
    inputs: ClassVar[tuple[Port, ...]] = (Port("input", DataType.OUTPUT, classes=(FactorTriageOutput,)),)

    def run(self, config: MetadataIssuesConfig, inputs: Mapping[str, Any], context: CheckContext) -> list[Finding]:  # noqa: ARG002
        """metadata-triage's findings, from the `factor-triage` Output's data."""
        return build_findings(inputs["input"].value.data(), config.max_examples)
