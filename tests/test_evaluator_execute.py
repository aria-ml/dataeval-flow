"""The shared execute: views, cache and producers in, one EvaluatorResult out, never an exception."""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar
from unittest.mock import patch

import polars as pl
import pytest
from dataeval.flags import ImageStats
from dataeval.types import DataFrameOutput

import dataeval_flow._cache as cache_module
from dataeval_flow import InputKind, InputSpec, SourceCount
from dataeval_flow.__main__ import _evaluator_entry
from dataeval_flow._cache import DatasetCache
from dataeval_flow.config import ViewOperation
from dataeval_flow.evaluators import (
    Evaluator,
    EvaluatorConfig,
    EvaluatorInputs,
    EvaluatorResult,
    get_evaluator,
    list_evaluators,
)
from dataeval_flow.evaluators._execute import execute
from dataeval_flow.workflows import DatasetContext, WorkflowContext
from tests.evaluator_toys import ToyImages, output_json


class _ToyResult(EvaluatorResult[DataFrameOutput]):
    """What the stats-reading toy evaluators here return."""


class _MetadataResult(EvaluatorResult[DataFrameOutput]):
    """What the metadata-reading toy returns: distinct, so a test can tell which config's class a result took."""


class _HashParams(EvaluatorConfig[_ToyResult]):
    type: str = "test.stat-names"
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.STATS}), sources=SourceCount.ONE_OR_MORE)

    def stats_request(self) -> dict[str, Any]:
        return {"duplicate_flags": ImageStats.HASH_DUPLICATES_BASIC}


class _MetadataParams(EvaluatorConfig[_MetadataResult]):
    type: str = "test.metadata"
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.METADATA}), sources=SourceCount.ONE)


class _StatsOnly(EvaluatorConfig[_ToyResult]):
    """Wants stats but declares no ``stats_request()``, so it reads the base's default."""

    type: str = "test.stats-only"
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.STATS}), sources=SourceCount.ONE)


class _ClustersWithoutFields(EvaluatorConfig[_ToyResult]):
    """Wants clusters but declares none of the clustering fields, tripping the producer's own guard."""

    type: str = "test.clusters-without-fields"
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.CLUSTERS}), sources=SourceCount.ONE)


class _ClusterParams(EvaluatorConfig[_ToyResult]):
    """Wants clusters and declares the clustering fields, but names no extractor in these tests."""

    type: str = "test.cluster-params"
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.CLUSTERS}), sources=SourceCount.ONE)
    cluster_algorithm: str | None = None
    n_clusters: int | None = None


def _stat_names(inputs: Sequence[EvaluatorInputs]) -> DataFrameOutput:
    """Which statistics each source arrived with."""
    return DataFrameOutput(
        pl.DataFrame(
            {
                "source": [i.source for i in inputs],
                "stats": [",".join(sorted(i.stats["stats"])) if i.stats else "" for i in inputs],
            }
        )
    )


class _StatNames(Evaluator[_HashParams, DataFrameOutput]):
    """Reports which statistics each source arrived with."""

    name: ClassVar[str] = "test.stat-names"
    description: ClassVar[str] = "Statistic names per source"
    dataeval_class: ClassVar[type] = object
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.STATS: "__init__"}

    def run(self, config: _HashParams, inputs: Sequence[EvaluatorInputs]) -> DataFrameOutput:
        return _stat_names(inputs)


class _Boom(_StatNames):
    name: ClassVar[str] = "test.boom"

    def run(self, config: _HashParams, inputs: Sequence[EvaluatorInputs]) -> DataFrameOutput:
        raise RuntimeError("boom")


class _NeedsMetadata(Evaluator[_MetadataParams, DataFrameOutput]):
    name: ClassVar[str] = "test.metadata"
    description: ClassVar[str] = "Reads metadata"
    dataeval_class: ClassVar[type] = object
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.METADATA: "__init__"}

    def run(self, config: _MetadataParams, inputs: Sequence[EvaluatorInputs]) -> DataFrameOutput:
        return _stat_names(inputs)


class _NeedsStatsOnly(Evaluator[_StatsOnly, DataFrameOutput]):
    name: ClassVar[str] = "test.stats-only"
    description: ClassVar[str] = "Reads stats under the default request"
    dataeval_class: ClassVar[type] = object
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.STATS: "__init__"}

    def run(self, config: _StatsOnly, inputs: Sequence[EvaluatorInputs]) -> DataFrameOutput:
        return _stat_names(inputs)


class _NeedsClustersWithoutFields(Evaluator[_ClustersWithoutFields, DataFrameOutput]):
    name: ClassVar[str] = "test.clusters-without-fields"
    description: ClassVar[str] = "Reads clusters without clustering fields"
    dataeval_class: ClassVar[type] = object
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.CLUSTERS: "__init__"}

    def run(self, config: _ClustersWithoutFields, inputs: Sequence[EvaluatorInputs]) -> DataFrameOutput:
        return _stat_names(inputs)


class _NeedsClusterParams(Evaluator[_ClusterParams, DataFrameOutput]):
    name: ClassVar[str] = "test.cluster-params"
    description: ClassVar[str] = "Reads clusters"
    dataeval_class: ClassVar[type] = object
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.CLUSTERS: "__init__"}

    def run(self, config: _ClusterParams, inputs: Sequence[EvaluatorInputs]) -> DataFrameOutput:
        return _stat_names(inputs)


def _context(*names: str, dataset: Any = None, cache: DatasetCache | None = None, **extra: Any) -> WorkflowContext:
    toy = dataset if dataset is not None else ToyImages()
    return WorkflowContext(
        dataset_contexts={name: DatasetContext(name=name, dataset=toy, cache=cache, **extra) for name in names}
    )


class TestExecute:
    def test_each_source_arrives_with_its_stats(self):
        result = execute(_StatNames(), _context("src"), _HashParams())
        assert result.success, result.errors
        assert isinstance(result, _ToyResult)
        assert output_json(result)["rows"] == [{"source": "src", "stats": "dhash,phash,xxhash"}]
        assert result.metadata.evaluator == "test.stat-names"
        assert result.dataset is not None
        assert result.sources is None

    def test_several_sources_arrive_in_order(self):
        result = execute(_StatNames(), _context("a", "b"), _HashParams())
        assert [row["source"] for row in output_json(result)["rows"]] == ["a", "b"]
        assert result.dataset is None
        assert result.sources is not None
        assert set(result.sources) == {"a", "b"}

    def test_the_view_is_applied_before_producing(self):
        context = _context("src", view_operations=[ViewOperation(type="Limit", params={"size": 4})])
        result = execute(_StatNames(), context, _HashParams())
        assert result.dataset is not None
        assert len(result.dataset) == 4

    def test_the_wrong_config_fails_the_result(self):
        """The failed result uses the evaluator's own result class."""
        result = execute(_StatNames(), _context("src"), _MetadataParams())
        assert not result.success
        assert type(result) is _ToyResult
        assert result.metadata.evaluator == "test.stat-names"
        assert result.errors == ["Expected _HashParams, got _MetadataParams"]
        assert result.to_dict()["errors"] == result.errors

    def test_an_exception_becomes_a_failed_result(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.ERROR):
            result = execute(_Boom(), _context("src"), _HashParams())
        assert not result.success
        assert type(result) is _ToyResult
        assert result.metadata.evaluator == "test.boom"
        assert result.errors == ["RuntimeError: boom"]
        assert "boom" in caplog.text

    def test_a_kind_without_a_producer_fails_the_result(self):
        result = execute(_NeedsMetadata(), _context("src"), _MetadataParams())
        assert not result.success
        assert "No producer for metadata" in result.errors[0]

    def test_a_stats_consumer_without_stats_request_measures_every_family(self):
        result = execute(_NeedsStatsOnly(), _context("src"), _StatsOnly())
        assert result.success, result.errors
        (row,) = output_json(result)["rows"]
        assert {"xxhash", "mean", "height"} <= set(row["stats"].split(","))

    def test_a_cluster_consumer_without_clustering_fields_fails_the_result(self):
        result = execute(_NeedsClustersWithoutFields(), _context("src"), _ClustersWithoutFields())
        assert not result.success
        assert "consumes clusters but declares no clustering fields" in result.errors[0]

    def test_cluster_mode_without_an_extractor_fails_the_result(self):
        result = execute(_NeedsClusterParams(), _context("src"), _ClusterParams())
        assert not result.success
        assert "Cluster mode needs an extractor on the task." in result.errors[0]

    def test_a_second_run_reads_the_cache(self, tmp_path):
        cache = DatasetCache.get_or_create(cache_dir=tmp_path, name="toy", cache_key="k")
        with patch.object(cache_module, "_do_compute_stats", wraps=cache_module._do_compute_stats) as compute:
            execute(_StatNames(), _context("src", cache=cache), _HashParams())
            execute(_StatNames(), _context("src", cache=cache), _HashParams())
        assert compute.call_count == 1


class TestRegistry:
    @pytest.fixture(autouse=True)
    def _serve_the_test_evaluator(self, plugins: dict[str, list[tuple[str, str]]]):
        plugins["dataeval_flow.evaluators"] = [("test.stat-names", f"{__name__}:_StatNames")]

    def test_lookup_by_type(self):
        assert get_evaluator("test.stat-names") is _StatNames

    def test_an_unknown_type_names_the_known_ones(self):
        with pytest.raises(ValueError, match=r"Unknown evaluator: 'nope'. Installed: \[.*'test.stat-names'\]"):
            get_evaluator("nope")

    def test_the_listing_says_what_each_consumes(self):
        assert _StatNames in list_evaluators()
        assert _evaluator_entry(_StatNames) == {
            "name": "test.stat-names",
            "description": "Statistic names per source",
            "consumes": "stats",
            "sources": "1+",
        }
