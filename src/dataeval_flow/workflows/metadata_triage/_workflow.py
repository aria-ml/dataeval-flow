"""Metadata triage workflow: what a run failed to read, and what to do about it.

Its logic is the ``triage`` evaluator's and the ``metadata-issues`` check's; this adapter keeps the legacy result until
the port to a preset replaces it (spec §10.10).
"""

from typing import ClassVar

from dataeval_flow._binning import attach_binning
from dataeval_flow._metadata import build_metadata
from dataeval_flow._policy import policy_for
from dataeval_flow._triage_report import build_findings
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators.quality._config import TriageConfig
from dataeval_flow.evaluators.quality._triage import TriageEvaluator
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._context import WorkflowContext
from dataeval_flow.workflows.metadata_triage._config import MetadataTriageConfig
from dataeval_flow.workflows.metadata_triage._outputs import (
    MetadataTriageMetadata,
    MetadataTriageOutput,
    MetadataTriageRawOutput,
    MetadataTriageReport,
    MetadataTriageResult,
)

__all__ = ["MetadataTriageWorkflow"]


class MetadataTriageWorkflow(Workflow[MetadataTriageConfig, MetadataTriageResult]):
    """Surface what a metadata run silently failed to read, and suggest how to fix it."""

    name: ClassVar[str] = "metadata-triage"
    title: ClassVar[str] = "Metadata Triage"
    description: ClassVar[str] = "Report unreadable and unpinned metadata factors, with suggested corrections"

    def run(self, config: MetadataTriageConfig, context: WorkflowContext) -> MetadataTriageResult:
        """Build the metadata, triage it, and report what it could not read."""
        from dataeval_flow._view import build_view

        policy = policy_for(context, config)
        source, dc = next(iter(context.dataset_contexts.items()))
        dataset = dc.dataset
        if dc.view_operations:
            dataset = build_view(dataset, dc.view_operations)  # type: ignore[arg-type]
        metadata = build_metadata(dataset, policy)
        triage = TriageConfig(
            verify=config.verify, default_bins=config.default_bins, min_missing_fraction=config.min_missing_fraction
        )
        inputs = [EvaluatorInputs(source=source, metadata=metadata, metadata_policy=policy)]
        data = TriageEvaluator().run(triage, inputs).data()
        raw = MetadataTriageRawOutput(
            dataset_size=len(dataset),
            findings=data["findings"],
            suggested_policy=data["suggested_policy"],
            suggested_policy_yaml=data["suggested_policy_yaml"],
            verification=data["verification"],
            verification_error=data["verification_error"],
            counts=data["counts"],
            factor_count=data["factor_count"],
        )
        result_metadata = MetadataTriageMetadata(
            blocking=sum(1 for f in raw.findings if f.severity == "blocking"),
            verified=sum(1 for v in raw.verification if v.recovered),
        )
        attach_binning(result_metadata, metadata, policy)
        report = MetadataTriageReport(
            summary=f"{raw.factor_count} factors, {len(raw.findings)} findings ({result_metadata.blocking} blocking).",
            findings=build_findings(data, config.max_examples),
        )
        return MetadataTriageResult(
            type=self.name,
            success=True,
            output=MetadataTriageOutput(raw=raw, report=report),
            metadata=result_metadata,
            dataset=dataset,
        )
