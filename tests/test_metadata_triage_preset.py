"""The metadata-triage preset: the `triage` evaluator and the `metadata-issues` check, on the task's one source
(spec §10.10)."""

import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import run, run_tasks
from dataeval_flow._binning import descriptor_from_record
from dataeval_flow._cache import DatasetCache
from dataeval_flow._encoding_cli import _binning_records
from dataeval_flow.evaluators.quality import TriageConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.metadata_triage import MetadataTriageConfig, MetadataTriageWorkflow
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages
from tests.finding_blocks import tables
from tests.golden.triage import WEIGHTS
from tests.triage_toys import AltitudeDataset, LatitudeDataset, MixedWeightDataset


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def test_the_settings_expand_to_triage_and_its_check() -> None:
    config = MetadataTriageConfig(
        metadata="weights", verify=False, default_bins=4, min_missing_fraction=0.5, max_examples=3
    )
    chain = MetadataTriageWorkflow.chain(config)
    (triage,) = chain.evaluators
    assert isinstance(triage, TriageConfig)
    assert (triage.name, triage.metadata, triage.verify, triage.default_bins, triage.min_missing_fraction) == (
        "triage",
        "weights",
        False,
        4,
        0.5,
    )
    assert list(chain.steps) == [
        {"name": "triage", "evaluator": "triage", "input": "data"},
        {"name": "issues", "check": "metadata-issues", "input": "triage", "max_examples": 3},
    ]


def test_a_run_is_a_chain_result_of_its_two_steps() -> None:
    result = run(MetadataTriageConfig(), MixedWeightDataset())
    assert isinstance(result, ChainResult)
    assert result.type == "metadata-triage"
    assert list(result.steps) == ["triage", "issues"]
    assert result.health["status"] == "warning"
    assert [f.title for f in result.findings] == ["Unreadable factors", "Suggested policy", "Verified"]


def test_its_envelope_records_the_encoding_triage_read() -> None:
    result = run(MetadataTriageConfig(), MixedWeightDataset())
    record = result.metadata.metadata_binning
    assert record is not None
    assert "weight" in record["unusable"]
    assert result.metadata.encoding_digest == record["encoding_digest"]


def test_a_named_policy_reaches_triage() -> None:
    config = chain_pipeline(
        workflows=[{"name": "triage", "type": "metadata-triage", "metadata": "weights"}],
        tasks=[{"name": "t", "workflow": "triage", "sources": ["src"]}],
        datasets={"src": MixedWeightDataset()},
        extra={"metadata": [WEIGHTS]},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert "Unreadable factors" not in [f.title for f in result.findings]
    assert result.metadata.metadata_binning is not None
    assert "weight" in result.metadata.metadata_binning["factors"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metadata_auto_bin_method", "uniform_width"),
        ("metadata_exclude", ["id"]),
        ("metadata_continuous_factor_bins", {"weight": 4}),
        ("metadata_factor_source", "coded"),
    ],
)
def test_a_retired_field_is_refused(field: str, value: Any) -> None:
    with pytest.raises(ValidationError, match=re.escape(field)):
        MetadataTriageConfig.model_validate({field: value})


def test_a_dataset_with_nothing_to_triage_makes_no_findings() -> None:
    result = run(MetadataTriageConfig(), ToyImages(count=12))
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert result.findings == []
    assert result.health["status"] == "ok"
    assert result.metadata.metadata_binning is not None
    assert result.metadata.metadata_binning["factors"] == {}


def test_a_cached_rerun_still_pictures_where_the_problem_values_sit(tmp_path: Path) -> None:
    """The second run restores the Metadata from the cache's archive, which legacy triage never read."""
    result = None
    for _ in range(2):
        DatasetCache.clear_instances()
        result = run(MetadataTriageConfig(), LatitudeDataset(), cache_dir=tmp_path)
    assert isinstance(result, ChainResult)
    unreadable = next(f for f in result.findings if f.title == "Unreadable factors")
    (table,) = tables(unreadable)
    assert [row["value"] for row in table.rows] == ["N", "S"]


def test_as_a_step_over_a_list_it_triages_each_element() -> None:
    each = {"name": "triage", "workflow": "tri", "input": "splits"}
    config = chain_pipeline(
        workflows=[
            {"name": "tri", "type": "metadata-triage"},
            {"name": "w", "inputs": [{"name": "splits", "list": True}], "steps": [each]},
        ],
        tasks=[{"name": "t", "workflow": "w", "sources": ["a", "b"]}],
        datasets={"a": MixedWeightDataset(), "b": AltitudeDataset()},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert list(result.steps["triage/issues"].elements or {}) == ["a", "b"]
    assert result.metadata.metadata_binning is not None
    assert list(result.metadata.metadata_binning["per_split"]) == ["splits[a]", "splits[b]"]
    assert result.health["status"] == "warning"


def test_dataeval_flow_encoding_finds_the_record_and_writes_its_descriptor() -> None:
    result = run(MetadataTriageConfig(), AltitudeDataset())
    assert isinstance(result, ChainResult)
    records = _binning_records({"t": {"metadata": result.to_dict()["metadata"]}})
    assert "altitude" in descriptor_from_record(records["t"])["factors"]


def test_its_detailed_report_shows_the_metadata_factors() -> None:
    result = run(MetadataTriageConfig(), AltitudeDataset())
    assert "METADATA FACTORS" in result.report(detailed=True).upper()
    assert "altitude" in result.report(detailed=True)
