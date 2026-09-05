"""Tests for the metadata-triage workflow."""

from typing import Any

import numpy as np
from dataeval import Metadata
from dataeval.protocols import DatasetMetadata

from dataeval_flow.workflow import WorkflowContext
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
