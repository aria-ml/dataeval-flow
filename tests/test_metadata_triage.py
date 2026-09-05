"""Tests for the metadata-triage workflow."""

from typing import Any

import numpy as np
from dataeval import Metadata
from dataeval.protocols import DatasetMetadata

from dataeval_flow.config.schemas._metadata import ParseValueCorrectionConfig
from dataeval_flow.policy import ResolvedPolicy, build_correction
from dataeval_flow.workflow import DatasetContext, WorkflowContext
from dataeval_flow.workflows.metadata_triage import (
    MetadataTriageOutputs,
    MetadataTriageParameters,
    MetadataTriageRawOutputs,
    MetadataTriageReport,
)
from dataeval_flow.workflows.metadata_triage.workflow import MetadataTriageWorkflow


def test_parameters_default_to_verifying():
    params = MetadataTriageParameters()
    assert params.verify is True
    assert params.max_examples == 20
    assert params.default_bins == 10
    assert params.min_missing_fraction == 0.2


def test_parameters_accept_a_named_policy():
    # MetadataConfigMixin is what makes `metadata: standard` resolve.
    assert MetadataTriageParameters(metadata="standard").metadata == "standard"


def test_outputs_round_trip_as_json():
    raw = MetadataTriageRawOutputs(dataset_size=10)
    outputs = MetadataTriageOutputs(raw=raw, report=MetadataTriageReport(summary="none"))
    assert outputs.model_dump(mode="json")["raw"]["findings"] == []


class _MixedWeightDataset:
    """Classification items whose ``weight`` reading mixes numerals with numerals wearing
    commas.

    A walk, not ``Metadata.from_factors``: that constructor refuses a mixed-dtype column
    outright (``reject_mixed_values``), so the held-back path this fixture needs only
    exists for metadata read off a dataset — see ``tests/test_binning.py::_MixedDataset``,
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

    All-numeric, unlike ``_MixedWeightDataset``: the point is a column that reads cleanly
    and lands as ``unbinned`` (a cut DataEval derived from this draw) rather than
    ``unreadable``, so its suggestion is a bin count rather than a correction.
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
    from dataeval_flow.binning import describe_binning

    return describe_binning(metadata)


def test_the_workflow_reports_its_name():
    assert MetadataTriageWorkflow().name == "metadata-triage"


def test_a_mixed_column_is_found_and_a_correction_suggested():
    from dataeval_flow.triage import find_issues, to_policy_stanza

    findings = find_issues(_describe(_mixed_metadata()))
    weight = [f for f in findings if f.factor == "weight"]
    assert weight
    assert weight[0].category == "unreadable"
    assert to_policy_stanza(findings)["corrections"][0]["kind"] == "parse_value"


def test_execute_refuses_a_context_of_the_wrong_type():
    result = MetadataTriageWorkflow().execute(object(), MetadataTriageParameters())  # type: ignore[arg-type]
    assert result.success is False
    assert "WorkflowContext" in result.errors[0]


def test_execute_refuses_missing_parameters():
    result = MetadataTriageWorkflow().execute(WorkflowContext(), None)
    assert result.success is False


def test_execute_runs_end_to_end_on_a_real_dataset():
    context = WorkflowContext(
        dataset_contexts={"default": DatasetContext(name="default", dataset=_MixedWeightDataset())},
    )
    result = MetadataTriageWorkflow().execute(context, MetadataTriageParameters())

    assert result.success is True
    assert any(f.factor == "weight" for f in result.data.raw.findings)
    assert result.data.raw.suggested_policy_yaml
    assert "parse_value" in result.data.raw.suggested_policy_yaml
    weight = [v for v in result.data.raw.verification if v.factor == "weight"]
    assert weight
    assert weight[0].recovered is True
    assert result.data.report.findings
    assert result.metadata.blocking >= 1


def test_build_correction_is_public():
    correction = build_correction(ParseValueCorrectionConfig(factor="weight", drop=[","]))
    assert correction.factor == "weight"


def test_verification_recovers_a_mixed_column():
    from dataeval_flow.triage import find_issues

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
    from dataeval_flow.triage import Finding, Suggestion

    finding = Finding(
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
    reads cleanly — so a check reusing the correction test (present in ``factors``, absent
    from ``unusable``) would say ``recovered`` no matter what the suggested count did. This
    proves the real check instead: applying the suggested count actually turns the
    encoding's ``provenance`` from ``"derived"`` into something pinned.
    """
    from dataeval_flow.triage import find_issues

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
    """The bin-recovery check can say no — proven at the predicate, not through a live run.

    Explicitly assigning ``Metadata.continuous_factor_bins`` always routes through
    DataEval's ``digitize_data`` rather than its auto-cut ``bin_data``, and the two are
    exactly what set ``provenance`` to something other than ``"derived"`` versus
    ``"derived"`` (see ``dataeval.core._bin``). So a *complete*, correctly-shaped bin
    suggestion cannot come back still ``"derived"`` through the real ``Metadata`` API: the
    one honest attempt at constructing that case (assigning the suggested count and reading
    the record back) always pins it. What can still fail this check is a suggestion whose
    factor never bound — a stale name, or a run where the assignment did not stick — which
    is exactly what a hand-built ``after`` record captures without needing to fight
    ``Metadata`` into an inconsistent state.
    """
    from dataeval_flow.workflows.metadata_triage.workflow import _factor_recovered

    still_derived = {"unreviewed": ["altitude"], "factors": {"altitude": {}}, "unusable": {}}
    assert _factor_recovered(still_derived, "altitude", pinned=True) is False

    pinned_now = {"unreviewed": [], "factors": {"altitude": {}}, "unusable": {}}
    assert _factor_recovered(pinned_now, "altitude", pinned=True) is True
