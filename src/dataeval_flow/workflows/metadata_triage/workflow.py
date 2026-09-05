"""Metadata triage workflow: what a run failed to read, and what to do about it."""

import logging
from typing import Any, cast

from pydantic import BaseModel

from dataeval_flow.binning import attach_binning, describe_binning
from dataeval_flow.metadata import build_metadata, expand_declared_bins
from dataeval_flow.policy import policy_for
from dataeval_flow.triage import find_issues, incomplete_factors, render_stanza, to_policy_stanza
from dataeval_flow.workflow import WorkflowContext, WorkflowProtocol, WorkflowResult
from dataeval_flow.workflows.metadata_triage.outputs import (
    MetadataTriageMetadata,
    MetadataTriageOutputs,
    MetadataTriageRawOutputs,
    MetadataTriageReport,
)
from dataeval_flow.workflows.metadata_triage.params import MetadataTriageParameters
from dataeval_flow.workflows.metadata_triage.report import build_findings, summarize

__all__ = ["MetadataTriageWorkflow"]

_logger: logging.Logger = logging.getLogger(__name__)


def _check(context: Any, params: BaseModel | None) -> str | None:
    """The guard clauses `execute` needs, pulled out so it does not trip C901.

    Returns the error message for the first guard that fails, or None once `context` and
    `params` are known to be the types `execute` goes on to use.
    """
    if not isinstance(context, WorkflowContext):
        return f"Expected WorkflowContext, got {type(context).__name__}"
    if params is None:
        return "MetadataTriageParameters required"
    if not isinstance(params, MetadataTriageParameters):
        return f"Expected MetadataTriageParameters, got {type(params).__name__}"
    return None


class MetadataTriageWorkflow(WorkflowProtocol[MetadataTriageMetadata, MetadataTriageOutputs]):
    """Surface what a metadata run silently failed to read, and suggest how to fix it."""

    @property
    def name(self) -> str:
        """Workflow identifier."""
        return "metadata-triage"

    @property
    def description(self) -> str:
        """Human-readable description."""
        return "Report unreadable and unpinned metadata factors, with suggested corrections"

    @property
    def params_schema(self) -> type[MetadataTriageParameters]:
        """Pydantic model for workflow parameters."""
        return MetadataTriageParameters

    @property
    def output_schema(self) -> type[MetadataTriageOutputs]:
        """Pydantic model for workflow output."""
        return MetadataTriageOutputs

    def execute(
        self,
        context: WorkflowContext,
        params: BaseModel | None = None,
    ) -> WorkflowResult[MetadataTriageMetadata, MetadataTriageOutputs]:
        """Build the metadata, describe it, and report what it could not read."""
        error = _check(context, params)
        if error is not None:
            return self._failed(error)
        context = cast(WorkflowContext, context)
        params = cast(MetadataTriageParameters, params)

        policy = policy_for(context, params)
        try:
            from dataeval_flow.view import build_view

            dc = next(iter(context.dataset_contexts.values()))
            dataset = dc.dataset
            if dc.view_operations:
                dataset = build_view(dataset, dc.view_operations)  # type: ignore[arg-type]
            metadata = build_metadata(dataset, policy)
            record = self._describe(metadata, policy)

            findings = find_issues(
                record,
                min_missing_fraction=params.min_missing_fraction,
                default_bins=params.default_bins,
            )
            stanza = to_policy_stanza(findings)
            raw = MetadataTriageRawOutputs(
                dataset_size=len(dataset),
                findings=findings,
                suggested_policy=stanza,
                suggested_policy_yaml=render_stanza(stanza, incomplete=incomplete_factors(findings)),
                factor_count=len(record.get("factors") or {}),
            )
            raw.counts = summarize(raw)

            result_metadata = MetadataTriageMetadata(
                blocking=sum(1 for f in findings if f.severity == "blocking"),
            )
            attach_binning(result_metadata, metadata, policy)
            report = MetadataTriageReport(
                summary=(
                    f"{raw.factor_count} factors, {len(findings)} findings ({result_metadata.blocking} blocking)."
                ),
                findings=build_findings(raw, params.max_examples),
            )
            return WorkflowResult(
                name=self.name,
                success=True,
                data=MetadataTriageOutputs(raw=raw, report=report),
                metadata=result_metadata,
                dataset=dataset,
            )
        except Exception as e:
            _logger.exception("Workflow '%s' failed", self.name)
            return self._failed(str(e))

    @staticmethod
    def _describe(metadata: Any, policy: Any) -> dict[str, Any]:
        """The binning record, built the way ``attach_binning`` builds it.

        The bins as applied, not as spelled: ``unmatched_bin_requests`` is a set difference
        against the factor names, and a bare declared name is not one of those.
        """
        declared = dict(policy.continuous_factor_bins) or None
        requested = expand_declared_bins(declared, metadata.factor_names, metadata.levels) if declared else None
        return describe_binning(
            metadata,
            excluded=list(policy.exclude) or None,
            requested_bins=requested,
            factor_source=policy.factor_source,
            declared_bins=declared,
        )

    def _failed(self, message: str) -> WorkflowResult[MetadataTriageMetadata, MetadataTriageOutputs]:
        """An unsuccessful result carrying one error, as every workflow's guards return."""
        return WorkflowResult(
            name=self.name,
            success=False,
            data=MetadataTriageOutputs(
                raw=MetadataTriageRawOutputs(dataset_size=0),
                report=MetadataTriageReport(summary="Workflow failed", findings=[]),
            ),
            metadata=MetadataTriageMetadata(),
            errors=[message],
        )
