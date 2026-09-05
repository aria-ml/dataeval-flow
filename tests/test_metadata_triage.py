"""Tests for the metadata-triage workflow."""

from dataeval_flow.workflows.metadata_triage import (
    MetadataTriageOutputs,
    MetadataTriageParameters,
    MetadataTriageRawOutputs,
    MetadataTriageReport,
)


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
