"""Metadata triage workflow: what a run failed to read, and what to do about it."""

import logging
from dataclasses import replace
from typing import Any, cast

from pydantic import BaseModel

from dataeval_flow.binning import attach_binning, describe_binning
from dataeval_flow.metadata import build_metadata, expand_declared_bins
from dataeval_flow.policy import build_correction, policy_for
from dataeval_flow.triage import Finding, find_issues, incomplete_factors, render_stanza, to_policy_stanza
from dataeval_flow.workflow import WorkflowContext, WorkflowProtocol, WorkflowResult
from dataeval_flow.workflows.metadata_triage.outputs import (
    MetadataTriageMetadata,
    MetadataTriageOutputs,
    MetadataTriageRawOutputs,
    MetadataTriageReport,
    VerificationEntry,
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

            if params.verify:
                try:
                    raw.verification = self._verify(metadata, policy, findings)
                except Exception as e:  # the findings are worth having without it
                    _logger.warning("Verification unavailable", exc_info=True)
                    raw.verification_error = str(e) or type(e).__name__

            result_metadata = MetadataTriageMetadata(
                blocking=sum(1 for f in findings if f.severity == "blocking"),
                verified=sum(1 for v in raw.verification if v.recovered),
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

    def _verify(
        self,
        metadata: Any,
        policy: Any,
        findings: "list[Finding]",
    ) -> "list[VerificationEntry]":
        """Read the metadata back under the complete suggestions, and say what they recovered.

        Costs no second dataset walk: ``repair`` applies corrections to the values the walk
        already kept and returns a derived copy sharing the immutable store.  The policy's
        own corrections are passed through alongside the suggested ones because ``repair``
        **replaces** rather than accumulates.

        A correction and a bin suggestion claim different things, so ``recovered`` is
        checked differently for each — see :func:`_factor_recovered`, which the shape of
        each finding's suggestion (corrections vs. a bin count) routes to the right check.

        Incomplete suggestions are never applied, so this can never report recovery from a
        placeholder.  A suggestion can also be well-formed, run cleanly and still not
        recover what it claims — which is the case this exists to catch before a stanza is
        committed to a config.
        """
        from dataeval_flow.config.schemas import MetadataPolicyConfig

        entries: list[VerificationEntry] = []
        runnable: list[Any] = []
        bins: dict[str, Any] = {}
        correction_factors: set[str] = set()
        bin_factors: set[str] = set()
        for finding in findings:
            suggestion = finding.suggestion
            if suggestion is None:
                continue
            if not suggestion.complete:
                entries.append(_unapplied(finding))
                continue
            factor_bins = suggestion.policy.get("continuous_factor_bins") or {}
            bins.update(factor_bins)
            bin_factors.update(factor_bins)
            runnable.extend(suggestion.corrections)
            correction_factors.update(c["factor"] for c in suggestion.corrections)

        if not runnable and not bins:
            return entries

        # Validated as a policy first, so a malformed suggestion fails here naming the field
        # rather than inside DataEval naming a constructor argument.
        validated = MetadataPolicyConfig.model_validate({"name": "_triage", "corrections": runnable})
        built = [build_correction(entry) for entry in validated.corrections or ()]
        repaired = metadata.repair([*policy.correction_specs, *built])
        described_policy = policy
        if bins:
            merged_bins = {**dict(policy.continuous_factor_bins), **bins}
            repaired.continuous_factor_bins = merged_bins
            # `_describe` reads `continuous_factor_bins` off the policy, not off `repaired`,
            # so it has to see the bins actually assigned here — otherwise the re-described
            # record's `requested_bins`/`bin_expansion` would describe the policy this
            # verification started from rather than what it just ran.
            described_policy = replace(policy, continuous_factor_bins=merged_bins)
        after = self._describe(repaired, described_policy)
        factors = after.get("factors") or {}

        for name in sorted(correction_factors | bin_factors):
            # A name a correction also names wins the correction check: that is the more
            # fundamental claim (the column exists at all), and no finding actually
            # produces both for one factor today.
            pinned = name in bin_factors and name not in correction_factors
            recovered = _factor_recovered(after, name, pinned=pinned)
            entries.append(
                VerificationEntry(
                    factor=name,
                    applied=True,
                    recovered=recovered,
                    detail=_recovery_detail(factors.get(name), recovered, pinned=pinned),
                )
            )
        return entries


def _unapplied(finding: "Finding") -> "VerificationEntry":
    """A verification entry for a suggestion left incomplete, never applied.

    Pulled out of :meth:`MetadataTriageWorkflow._verify` so that method does not trip C901
    — this is the one branch that does not touch the dataset at all.
    """
    suggestion = finding.suggestion
    corrections = suggestion.corrections if suggestion is not None else ()
    holes = sum(1 for c in corrections for rule in c.get("rules", ()) if rule.get("to") is None)
    return VerificationEntry(
        factor=finding.factor,
        applied=False,
        recovered=False,
        detail=f"not applied; {holes} values still need codes",
    )


def _factor_recovered(after: "dict[str, Any]", name: str, *, pinned: bool) -> bool:
    """Whether one factor's suggestion actually did what it claimed.

    A correction (an ``unreadable`` finding) claims a held-back column becomes a factor:
    recovered means present in ``factors`` and absent from ``unusable``.

    A bin suggestion (an ``unbinned`` finding) is only ever raised for a factor already
    present and already readable — what it lacks is a pinned cut, not existence, per
    ``triage._encodings`` — so that same check would be true before the suggestion runs
    and after it regardless of what the suggested count did, and could never say no.
    Recovered there instead means the factor is still present *and* its cut no longer reads
    ``provenance="derived"`` in the re-described record — presence alone is not enough, or a
    factor that vanished from the re-described record entirely (absent from ``factors``, and
    so absent from ``unreviewed`` too) would satisfy this by having disappeared rather than
    by having been pinned.
    """
    factors = after.get("factors") or {}
    if pinned:
        return name in factors and name not in (after.get("unreviewed") or ())
    unusable = after.get("unusable") or {}
    return name in factors and name not in unusable


def _recovery_detail(info: "dict[str, Any] | None", recovered: bool, *, pinned: bool) -> str:
    """One line saying what the reading actually produced.

    Worded for whichever claim :func:`_factor_recovered` checked: a bin suggestion reports
    the cut it produced, a correction reports whether the column became one at all.
    """
    fit = (info or {}).get("fit") or {}
    bin_buckets = fit.get("bins")
    buckets = bin_buckets if bin_buckets is not None else fit.get("levels") or []
    kind = "bins" if bin_buckets is not None else "levels"
    if pinned:
        if not recovered:
            return "applied, but the cut still reads as derived rather than pinned"
        return f"{len(buckets)} {kind}, {len(fit.get('empty') or ())} empty"
    if not recovered:
        return "applied, but still unreadable; try a different reading"
    return f"became a factor, {len(buckets)} {kind}"
