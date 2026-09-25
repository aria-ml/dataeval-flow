"""Tests for the metadata-triage workflow."""

from typing import Any

import numpy as np
from dataeval import Metadata
from dataeval.protocols import DatasetMetadata

from dataeval_flow._policy import ResolvedPolicy, build_correction
from dataeval_flow.config import ParseValueCorrectionConfig
from dataeval_flow.workflows import DatasetContext, WorkflowContext
from dataeval_flow.workflows.metadata_triage import MetadataTriageConfig, MetadataTriageWorkflow
from dataeval_flow.workflows.metadata_triage._outputs import (
    MetadataTriageOutput,
    MetadataTriageRawOutput,
    MetadataTriageReport,
    VerificationEntry,
)


def test_parameters_default_to_verifying():
    params = MetadataTriageConfig()
    assert params.verify is True
    assert params.max_examples == 20
    assert params.default_bins == 10
    assert params.min_missing_fraction == 0.2


def test_parameters_accept_a_named_policy():
    # MetadataConfigMixin is what makes `metadata: standard` resolve.
    assert MetadataTriageConfig(metadata="standard").metadata == "standard"


def test_outputs_round_trip_as_json():
    raw = MetadataTriageRawOutput(dataset_size=10)
    outputs = MetadataTriageOutput(raw=raw, report=MetadataTriageReport(summary="none"))
    assert outputs.model_dump(mode="json")["raw"]["findings"] == []


class _MixedWeightDataset:
    """Classification items whose ``weight`` reading mixes numerals with numerals wearing
    commas.

    Built by walking the dataset; ``Metadata.from_factors`` refuses a mixed-dtype column
    outright (``reject_mixed_values``). The held-back path this fixture needs exists only
    for metadata read off a dataset — see ``tests/test_binning.py::_MixedDataset``,
    which this mirrors.
    """

    def __init__(self, n: int = 60) -> None:
        self._n = n
        rng = np.random.default_rng(0)
        self._weight = rng.integers(1000, 9000, n)

    @property
    def metadata(self) -> DatasetMetadata:
        return {"id": "mixed-weight", "index2label": {0: "cat", 1: "dog"}}

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        one_hot = np.zeros(2, dtype=np.float32)
        one_hot[index % 2] = 1.0
        image = np.zeros((3, 8, 8), dtype=np.float32)
        raw = int(self._weight[index])
        # Every tenth reading is a numeral wearing commas rather than a plain number.
        weight: Any = f"{raw:,}" if index % 10 == 0 else raw
        datum: dict[str, Any] = {"id": index, "weight": weight}
        return image, one_hot, datum


def _mixed_metadata(n: int = 60) -> Metadata:
    """Metadata whose ``weight`` factor mixes numerals with numerals wearing commas."""
    return Metadata(_MixedWeightDataset(n))


class _AltitudeDataset:
    """Classification items with a continuous ``altitude`` factor nobody pinned.

    All-numeric: the point is a column that reads cleanly and lands as ``unbinned``
    (a cut DataEval derived from this draw), so its suggestion is a bin count,
    not a correction.
    """

    def __init__(self, n: int = 60) -> None:
        self._n = n
        rng = np.random.default_rng(1)
        self._altitude = rng.uniform(0.0, 1000.0, n)

    @property
    def metadata(self) -> DatasetMetadata:
        return {"id": "altitude", "index2label": {0: "cat", 1: "dog"}}

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        one_hot = np.zeros(2, dtype=np.float32)
        one_hot[index % 2] = 1.0
        image = np.zeros((3, 8, 8), dtype=np.float32)
        datum: dict[str, Any] = {"id": index, "altitude": float(self._altitude[index])}
        return image, one_hot, datum


def _altitude_metadata(n: int = 60) -> Metadata:
    """Metadata whose ``altitude`` factor is continuous and cut from this draw."""
    return Metadata(_AltitudeDataset(n))


def _describe(metadata: Metadata) -> dict:
    from dataeval_flow._binning import describe_binning

    return describe_binning(metadata)


def test_the_workflow_reports_its_name():
    assert MetadataTriageWorkflow().name == "metadata-triage"


def test_a_mixed_column_is_found_and_a_correction_suggested():
    from dataeval_flow._triage import find_issues, to_policy_stanza

    findings = find_issues(_describe(_mixed_metadata()))
    weight = [f for f in findings if f.factor == "weight"]
    assert weight
    assert weight[0].category == "unreadable"
    assert to_policy_stanza(findings)["corrections"][0]["kind"] == "parse_value"


def test_execute_runs_end_to_end_on_a_real_dataset():
    context = WorkflowContext(
        dataset_contexts={"default": DatasetContext(name="default", dataset=_MixedWeightDataset())},
    )
    result = MetadataTriageWorkflow().run(MetadataTriageConfig(), context)

    assert result.success is True
    assert any(f.factor == "weight" for f in result.output.raw.findings)
    assert result.output.raw.suggested_policy_yaml
    assert "parse_value" in result.output.raw.suggested_policy_yaml
    weight = [v for v in result.output.raw.verification if v.factor == "weight"]
    assert weight
    assert weight[0].recovered is True
    assert result.output.report.findings
    assert result.metadata.blocking >= 1


def test_build_correction_is_public():
    correction = build_correction(ParseValueCorrectionConfig(factor="weight", drop=[","]))
    assert correction.factor == "weight"


def test_verification_recovers_a_mixed_column():
    from dataeval_flow._triage import find_issues

    workflow = MetadataTriageWorkflow()
    metadata = _mixed_metadata()
    record = _describe(metadata)
    findings = find_issues(record)
    entries = workflow._verify(metadata, ResolvedPolicy(), findings)
    weight = [e for e in entries if e.factor == "weight"]
    assert weight
    assert weight[0].applied is True
    assert weight[0].recovered is True


def test_an_incomplete_suggestion_is_never_applied():
    from dataeval_flow._triage import Suggestion, TriageFinding

    finding = TriageFinding(
        factor="direction",
        category="unreadable",
        severity="blocking",
        repairable=True,
        suggestion=Suggestion(
            corrections=[{"kind": "remap", "factor": "direction", "rules": [{"match": "N", "to": None}]}],
            complete=False,
        ),
    )
    entries = MetadataTriageWorkflow()._verify(_mixed_metadata(), ResolvedPolicy(), [finding])
    (entry,) = entries
    assert entry.applied is False
    assert entry.recovered is False
    assert "need codes" in entry.detail


def test_verification_pins_a_derived_bin_suggestion():
    """A bin suggestion's claim is different from a correction's: check it separately.

    ``altitude`` is already a factor before the suggestion runs — it is continuous and
    reads cleanly. A check reusing the correction test (present in ``factors``, absent
    from ``unusable``) would say ``recovered`` no matter what the suggested count did.
    The real check: applying the suggested count turns the encoding's ``provenance``
    from ``"derived"`` into a pinned value.
    """
    from dataeval_flow._triage import find_issues

    workflow = MetadataTriageWorkflow()
    metadata = _altitude_metadata()
    record = _describe(metadata)
    findings = find_issues(record)
    altitude = [f for f in findings if f.factor == "altitude"]
    assert altitude
    assert altitude[0].category == "unbinned"
    assert altitude[0].suggestion is not None
    assert altitude[0].suggestion.policy.get("continuous_factor_bins")

    entries = workflow._verify(metadata, ResolvedPolicy(), findings)
    (entry,) = [e for e in entries if e.factor == "altitude"]
    assert entry.applied is True
    assert entry.recovered is True
    assert "bins" in entry.detail
    assert "empty" in entry.detail


def test_a_bin_suggestion_that_stays_derived_is_not_recovered():
    """The bin-recovery check can say no; this proves it at the predicate.

    Assigning ``Metadata.continuous_factor_bins`` routes through DataEval's
    ``digitize_data``, not its auto-cut ``bin_data``; the two set ``provenance`` to
    pinned versus ``"derived"`` respectively (see ``dataeval.core._bin``). A complete,
    correctly shaped bin suggestion therefore cannot come back ``"derived"`` through the
    real ``Metadata`` API: assigning the suggested count and reading the record back
    always pins it. The failure case is a suggestion whose factor never bound — a stale
    name, or a run where the assignment did not stick. The hand-built ``after`` record
    captures that case directly.
    """
    from dataeval_flow.workflows.metadata_triage._workflow import _factor_recovered

    still_derived = {"unreviewed": ["altitude"], "factors": {"altitude": {}}, "unusable": {}}
    assert _factor_recovered(still_derived, "altitude", pinned=True) is False

    pinned_now = {"unreviewed": [], "factors": {"altitude": {}}, "unusable": {}}
    assert _factor_recovered(pinned_now, "altitude", pinned=True) is True


def test_a_vanished_factor_is_not_recovered_even_when_absent_from_unreviewed():
    """Absence must not read as success.

    A factor missing from the re-described record's `factors` is absent from `unreviewed`
    as well: `unreviewed` only names factors that exist. Recovery requires the factor to
    be present. Being un-listed as unreviewed is not being pinned.
    """
    from dataeval_flow.workflows.metadata_triage._workflow import _factor_recovered

    vanished = {"unreviewed": [], "factors": {}, "unusable": {}}
    assert _factor_recovered(vanished, "altitude", pinned=True) is False


def test_the_workflow_is_discoverable():
    from dataeval_flow.workflows import get_workflow, list_workflows

    assert get_workflow("metadata-triage").name == "metadata-triage"
    assert any(w.name == "metadata-triage" for w in list_workflows())


def test_the_workflow_config_parses():
    from dataeval_flow.workflows.metadata_triage import MetadataTriageConfig

    cfg = MetadataTriageConfig(name="triage", metadata="standard")
    assert cfg.type == "metadata-triage"
    assert cfg.verify is True


def test_a_pipeline_config_accepts_a_triage_workflow():
    from dataeval_flow import PipelineConfig

    PipelineConfig.model_validate(
        {
            "workflows": [{"name": "triage", "type": "metadata-triage", "metadata": "standard"}],
        }
    )


def test_findings_render_a_stanza_and_a_report():
    """What `render_stanza` and `build_findings` add beyond the executed run.

    ``test_execute_runs_end_to_end_on_a_real_dataset`` drives ``execute`` and checks that
    the rendered YAML contains "parse_value" and that verification recovers ``weight``.
    It does not pin the exact stanza shape ``to_policy_stanza`` produces, or that
    ``build_findings`` groups an unreadable factor under a title naming it. Both are
    checked here, directly.
    """
    from dataeval_flow._triage import find_issues, incomplete_factors, render_stanza, to_policy_stanza
    from dataeval_flow.workflows.metadata_triage._report import build_findings

    metadata = _mixed_metadata()
    findings = find_issues(_describe(metadata))
    stanza = to_policy_stanza(findings)

    assert stanza["corrections"][0] == {"kind": "parse_value", "factor": "weight", "drop": [","]}
    assert "kind: parse_value" in render_stanza(stanza, incomplete=incomplete_factors(findings))

    raw = MetadataTriageRawOutput(dataset_size=60, findings=findings)
    reportables = build_findings(raw, max_examples=20)
    assert any("Unreadable" in r.title for r in reportables)


# ---------------------------------------------------------------------------
# report.py — build_findings / summarize
# ---------------------------------------------------------------------------


def _bare_finding(category: str, severity: str, factor: str = "weight") -> Any:
    from dataeval_flow._triage import TriageFinding

    return TriageFinding(factor=factor, category=category, severity=severity, remedy="do something")  # type: ignore[arg-type]


def test_a_category_with_a_blocking_finding_is_a_warning_reportable():
    """`WorkflowResult.health` counts exactly the categories this marks `"warning"`."""
    from dataeval_flow.workflows.metadata_triage._report import build_findings

    raw = MetadataTriageRawOutput(
        dataset_size=10,
        findings=[
            _bare_finding("unreadable", "blocking"),
            _bare_finding("unreadable", "note", factor="other"),
        ],
    )
    (reportable,) = build_findings(raw, max_examples=20)
    assert reportable.severity == "warning"


def test_a_category_with_only_warning_and_note_findings_is_an_info_reportable():
    # Hard-coding this branch to "info" would still pass; the blocking test above
    # catches that half.
    from dataeval_flow.workflows.metadata_triage._report import build_findings

    raw = MetadataTriageRawOutput(
        dataset_size=10,
        findings=[
            _bare_finding("unreviewed", "warning"),
            _bare_finding("unreviewed", "note", factor="other"),
        ],
    )
    (reportable,) = build_findings(raw, max_examples=20)
    assert reportable.severity == "info"


def test_a_suggested_policy_yaml_becomes_a_reportable():
    from dataeval_flow.workflows.metadata_triage._report import build_findings

    raw = MetadataTriageRawOutput(dataset_size=10, suggested_policy_yaml="metadata:\n  - name: standard\n")
    (reportable,) = build_findings(raw, max_examples=20)
    assert reportable.title == "Suggested policy"
    assert reportable.data["detail_lines"] == ["metadata:", "  - name: standard"]  # type: ignore[index]


def test_a_verification_entry_becomes_a_reportable():
    from dataeval_flow.workflows.metadata_triage._report import build_findings

    raw = MetadataTriageRawOutput(
        dataset_size=10,
        verification=[VerificationEntry(factor="weight", applied=True, recovered=True, detail="8 bins, 0 unread")],
    )
    (reportable,) = build_findings(raw, max_examples=20)
    assert reportable.title == "Verified"
    assert reportable.data["brief"] == "1 recovered"  # type: ignore[index]
    assert reportable.data["detail_lines"] == ["weight: 8 bins, 0 unread"]  # type: ignore[index]


def test_summarize_counts_by_category_and_by_severity():
    from dataeval_flow.workflows.metadata_triage._report import summarize

    raw = MetadataTriageRawOutput(
        dataset_size=10,
        findings=[
            _bare_finding("unreadable", "blocking"),
            _bare_finding("unreadable", "note", factor="other"),
            _bare_finding("degenerate", "note", factor="third"),
        ],
    )
    counts = summarize(raw)
    assert counts == {"unreadable": 2, "blocking": 1, "note": 2, "degenerate": 1}


# ---------------------------------------------------------------------------
# Verification failure, surfaced rather than silent
# ---------------------------------------------------------------------------


def test_a_verification_error_becomes_a_reportable_when_the_list_is_empty():
    # Distinct from `verify: false` and "nothing to verify", both of which also leave
    # `verification` empty but set no error -- and so add no Finding at all.
    from dataeval_flow.workflows.metadata_triage._report import build_findings

    raw = MetadataTriageRawOutput(dataset_size=10, verification_error="boom")
    (reportable,) = build_findings(raw, max_examples=20)
    assert reportable.title == "Verification failed"
    assert reportable.severity == "warning"
    assert reportable.data["detail_lines"] == ["boom"]  # type: ignore[index]


def test_no_verification_and_no_error_adds_no_reportable():
    from dataeval_flow.workflows.metadata_triage._report import build_findings

    raw = MetadataTriageRawOutput(dataset_size=10)
    assert build_findings(raw, max_examples=20) == []


def test_verification_failure_is_surfaced_not_silent():
    """A verification exception is reported as unverified, not swallowed.

    Upstream: 'reported as unverified, not as a failed run'. A reader must see that
    verification was attempted and failed, distinct from `verify: false` or having
    nothing to verify.
    """
    from unittest.mock import patch

    context = WorkflowContext(
        dataset_contexts={"default": DatasetContext(name="default", dataset=_MixedWeightDataset())},
    )
    with patch.object(MetadataTriageWorkflow, "_verify", side_effect=RuntimeError("boom")):
        result = MetadataTriageWorkflow().run(MetadataTriageConfig(), context)

    assert result.success is True  # the findings are worth having without verification
    assert result.output.raw.verification == []
    assert result.output.raw.verification_error == "boom"
    assert any(f.title == "Verification failed" for f in result.output.report.findings)
