"""A chain's result records the encodings its steps read its Datasets under (spec §10.10)."""

import logging
from collections.abc import Sequence
from typing import Any

import pytest

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._reads import MetadataRead, attach_reads
from dataeval_flow._metadata import build_metadata
from dataeval_flow._policy import ResolvedPolicy
from dataeval_flow._result import ResultMetadata
from dataeval_flow.steps import ChainResult
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyFactors

_LABELS = {"name": "labels", "type": "label-health"}
_READ_LABELS = {"name": "labels", "evaluator": "labels", "input": "data"}
_BALANCE = {"name": "balance", "type": "balance"}
_READ_BALANCE = {"name": "balance", "evaluator": "balance", "input": "data"}
_BINNED = {"name": "binned", "continuous_factor_bins": {"angle": 3}}


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _chain(
    steps: Sequence[dict[str, Any]],
    *,
    inputs: Sequence[Any] = ("data",),
    datasets: dict[str, Any] | None = None,
    evaluators: Sequence[dict[str, Any]] = (_LABELS, _BALANCE),
    policies: Sequence[dict[str, Any]] = (),
) -> ChainResult:
    datasets = datasets if datasets is not None else {"src": ToyFactors()}
    config = chain_pipeline(
        evaluators=list(evaluators),
        workflows=[{"name": "w", "inputs": list(inputs), "steps": list(steps)}],
        tasks=[{"name": "t", "workflow": "w", "sources": list(datasets)}],
        datasets=datasets,
        extra={"metadata": list(policies)},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


def _binning(result: ChainResult | ResultMetadata) -> dict[str, Any]:
    """The record of a chain's result or of an envelope, which the test expects to be there."""
    metadata = result if isinstance(result, ResultMetadata) else result.metadata
    assert metadata.metadata_binning is not None
    return metadata.metadata_binning


def test_a_chain_reading_one_dataset_one_way_records_one_encoding() -> None:
    result = _chain([_READ_BALANCE])
    record = result.metadata.metadata_binning
    assert record is not None
    assert "per_split" not in record
    assert sorted(record["factors"]) == ["angle", "site"]
    assert record["unreviewed"] == ["angle", "site"]
    assert result.metadata.encoding_digest is not None
    assert result.metadata.encoding_digest == record["encoding_digest"]


def test_a_chain_reading_no_metadata_records_no_encoding() -> None:
    result = _chain(
        [{"name": "dupes", "evaluator": "dupes", "input": "data"}], evaluators=[{"name": "dupes", "type": "duplicates"}]
    )
    assert result.metadata.metadata_binning is None
    assert result.metadata.encoding_digest is None
    # Only DataEval's own diagnostics, which this run raised, could still fill the section: no encoding is shown.
    assert "Auto-bin method" not in result.report(detailed=True)


def test_an_evaluator_reading_labels_alone_records_no_encoding() -> None:
    """label-health reads each item's class, which no factor's cuts change, so its read has no encoding to record."""
    result = _chain([_READ_LABELS])
    assert result.metadata.metadata_binning is None
    assert result.metadata.encoding_digest is None
    assert "Auto-bin method" not in result.report(detailed=True)


def test_a_part_read_for_its_labels_leaves_the_whole_s_record_alone() -> None:
    """data-splitting's shape: the whole set's balance reads its factors, and each part's label-health its labels."""
    result = _chain(
        [
            _READ_BALANCE,
            {"name": "split", "transform": "split", "input": "data", "test_frac": 0.25},
            {"name": "labels", "evaluator": "labels", "input": "split.train"},
        ]
    )
    record = _binning(result)
    assert "per_split" not in record
    assert sorted(record["factors"]) == ["angle", "site"]


def test_a_chain_reading_metadata_reports_its_factors() -> None:
    report = _chain([_READ_BALANCE]).report(detailed=True)
    assert "METADATA FACTORS" in report.upper()
    assert "angle" in report


def test_two_steps_reading_one_encoding_make_one_record() -> None:
    result = _chain([_READ_BALANCE, {"name": "again", "evaluator": "balance", "input": "data"}])
    assert "per_split" not in _binning(result)


def test_a_dataset_read_two_ways_keeps_both_and_names_the_second_by_its_policy() -> None:
    result = _chain(
        [_READ_BALANCE, {"name": "binned", "evaluator": "balance-binned", "input": "data"}],
        evaluators=[_BALANCE, {"name": "balance-binned", "type": "balance", "metadata": "binned"}],
        policies=[_BINNED],
    )
    per_split = _binning(result)["per_split"]
    assert list(per_split) == ["data", "data (binned)"]
    assert per_split["data"]["unreviewed"] == ["angle", "site"]
    assert per_split["data (binned)"]["unreviewed"] == ["site"]
    assert per_split["data (binned)"]["requested_bins"] == {"angle": 3}
    assert per_split["data"]["encoding_digest"] != per_split["data (binned)"]["encoding_digest"]
    assert result.metadata.encoding_digest is None


def test_each_element_of_a_list_is_its_own_record() -> None:
    result = _chain(
        [{"name": "balance", "evaluator": "balance", "input": "splits"}],
        inputs=[{"name": "splits", "list": True}],
        datasets={"a": ToyFactors(count=60), "b": ToyFactors(count=45)},
    )
    assert list(_binning(result)["per_split"]) == ["splits[a]", "splits[b]"]


def test_a_transform_s_read_is_recorded_beside_an_evaluator_s() -> None:
    result = _chain(
        [
            {"name": "split", "transform": "split", "input": "data", "test_frac": 0.25},
            {"name": "balance", "evaluator": "balance", "input": "split.train"},
        ]
    )
    assert list(_binning(result)["per_split"]) == ["data", "split.train"]


_OUTLIERS = {"name": "outliers", "evaluator": "outliers", "input": "data"}


def test_a_combine_s_read_reaches_the_record_when_no_evaluator_read_metadata() -> None:
    evaluators = [{"name": "outliers", "type": "outliers"}]
    assert _chain([_OUTLIERS], evaluators=evaluators).metadata.metadata_binning is None
    result = _chain(
        [_OUTLIERS, {"name": "by-class", "combine": "outliers-by-class", "input": "data", "outliers": "outliers"}],
        evaluators=evaluators,
    )
    record = _binning(result)
    assert "per_split" not in record
    assert sorted(record["factors"]) == ["angle", "site"]
    assert result.metadata.encoding_digest == record["encoding_digest"]


def test_a_record_that_cannot_be_described_costs_the_record_not_the_run(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import dataeval_flow._binning as binning

    def _renamed_upstream(*_: object) -> None:
        raise RuntimeError("a column DataEval renamed")

    monkeypatch.setattr(binning, "describe_under", _renamed_upstream)
    with caplog.at_level(logging.WARNING):
        result = _chain([_READ_BALANCE])
    assert result.metadata.metadata_binning is None
    assert result.metadata.encoding_digest is None
    assert "Binning record unavailable" in caplog.text


def _read(address: str, policy: ResolvedPolicy | None, name: str | None = None) -> MetadataRead:
    """A read of `ToyFactors`' Metadata, built afresh under `policy`, so no two reads share one object."""
    return MetadataRead(address, name, policy, build_metadata(ToyFactors(), policy))


def test_reads_whose_policies_differ_only_in_value_range_are_one_record() -> None:
    """data-cleaning's `label-health` carries the dataset's value range on its policy, and its combine reads with none;
    both read one encoding."""
    envelope = ResultMetadata()
    attach_reads(envelope, [_read("data", ResolvedPolicy(value_range=(0.0, 255.0))), _read("data", None)])
    assert envelope.metadata_binning is not None
    assert "per_split" not in envelope.metadata_binning


def test_a_second_unnamed_reading_of_one_dataset_is_numbered() -> None:
    envelope = ResultMetadata()
    reads = [
        _read("data", None),
        _read("data", ResolvedPolicy(continuous_factor_bins={"angle": 3})),
        _read("data", ResolvedPolicy(continuous_factor_bins={"angle": 4})),
    ]
    attach_reads(envelope, reads)
    assert list(_binning(envelope)["per_split"]) == ["data", "data (no policy)", "data (3)"]


def test_no_reads_record_nothing() -> None:
    envelope = ResultMetadata()
    attach_reads(envelope, [])
    assert envelope.metadata_binning is None
    assert envelope.encoding_digest is None


@pytest.mark.parametrize("name", ["representation", "coverage", "prioritization", "label-alignment"])
def test_the_scope_evaluators_read_no_factors(name: str) -> None:
    from dataeval_flow.evaluators import get_evaluator

    assert get_evaluator(name).reads_factors is False


def test_a_chain_reading_labels_through_representation_records_no_encoding() -> None:
    """`representation` reads each item's class through the Metadata, and no factor (coverage spec §3.3)."""
    result = _chain(
        [{"name": "rep", "evaluator": "rep", "input": "data"}],
        evaluators=[{"name": "rep", "type": "representation"}],
    )
    assert result.metadata.metadata_binning is None


def test_data_splitting_records_the_whole_set_once() -> None:
    """Each part's steps read its labels; only the whole set's factors are read (coverage spec §6.3)."""
    from tests.golden.splitting import pipeline

    result = run_tasks(pipeline("coverage", legacy=False))["t"]
    assert isinstance(result, ChainResult)
    record = _binning(result)
    assert "per_split" not in record


def test_data_prioritization_records_no_binning() -> None:
    """Its steps read no factor, so its envelope has no binning record or encoding digest."""
    from tests.golden.prioritization import pipeline

    result = run_tasks(pipeline("plain"))["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert result.metadata.metadata_binning is None
    assert result.metadata.encoding_digest is None
