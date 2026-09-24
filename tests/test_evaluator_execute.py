"""The shared execute: views, cache and producers in, one EvaluatorResult out, never an exception."""

import logging
from collections.abc import Sequence
from typing import Any, ClassVar
from unittest.mock import patch

import polars as pl
import pytest
from dataeval.flags import ImageStats
from dataeval.types import DataFrameOutput

import dataeval_flow.cache as cache_module
import dataeval_flow.evaluator as evaluator_module
from dataeval_flow.cache import DatasetCache
from dataeval_flow.config.schemas import ViewOperation
from dataeval_flow.evaluator import get_evaluator, list_evaluators
from dataeval_flow.evaluator.base import EvaluatorParametersBase, InputKind, InputSpec, SourceCount
from dataeval_flow.evaluator.inputs import Inputs
from dataeval_flow.evaluator.protocol import EvaluatorBase
from dataeval_flow.workflow import DatasetContext, WorkflowContext
from tests.evaluator_toys import ToyImages


class _HashParams(EvaluatorParametersBase):
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.STATS}), sources=SourceCount.ONE_OR_MORE)

    def stats_request(self) -> dict[str, Any]:
        return {"duplicate_flags": ImageStats.HASH_DUPLICATES_BASIC}


class _MetadataParams(EvaluatorParametersBase):
    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.METADATA}), sources=SourceCount.ONE)


class _StatNames(EvaluatorBase):
    """Reports which statistics each source arrived with."""

    name = "test.stat-names"
    description = "Statistic names per source"
    params_schema = _HashParams
    dataeval_class = object
    entry_points: ClassVar = {InputKind.STATS: "__init__"}

    def run(self, params: _HashParams, inputs: Sequence[Inputs]) -> DataFrameOutput:
        return DataFrameOutput(
            pl.DataFrame(
                {
                    "source": [i.source for i in inputs],
                    "stats": [",".join(sorted(i.stats["stats"])) if i.stats else "" for i in inputs],
                }
            )
        )


class _Boom(_StatNames):
    name = "test.boom"

    def run(self, params: _HashParams, inputs: Sequence[Inputs]) -> DataFrameOutput:
        raise RuntimeError("boom")


class _NeedsMetadata(_StatNames):
    name = "test.metadata"
    params_schema = _MetadataParams


class _StatsOnly(EvaluatorParametersBase):
    """Wants stats but declares no ``stats_request()``, tripping the producer's own guard."""

    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.STATS}), sources=SourceCount.ONE)


class _ClustersWithoutFields(EvaluatorParametersBase):
    """Wants clusters but declares none of the clustering fields, tripping the same guard."""

    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.CLUSTERS}), sources=SourceCount.ONE)


class _ClusterParams(EvaluatorParametersBase):
    """Wants clusters and declares the clustering fields, but names no extractor in these tests."""

    inputs: ClassVar[InputSpec] = InputSpec(required=frozenset({InputKind.CLUSTERS}), sources=SourceCount.ONE)
    cluster_algorithm: str | None = None
    n_clusters: int | None = None


class _NeedsStatsOnly(_StatNames):
    name = "test.stats-only"
    params_schema = _StatsOnly


class _NeedsClustersWithoutFields(_StatNames):
    name = "test.clusters-without-fields"
    params_schema = _ClustersWithoutFields


class _NeedsClusterParams(_StatNames):
    name = "test.cluster-params"
    params_schema = _ClusterParams


def _context(*names: str, dataset: Any = None, cache: DatasetCache | None = None, **extra: Any) -> WorkflowContext:
    toy = dataset if dataset is not None else ToyImages()
    return WorkflowContext(
        dataset_contexts={name: DatasetContext(name=name, dataset=toy, cache=cache, **extra) for name in names}
    )


class TestExecute:
    def test_each_source_arrives_with_its_stats(self):
        result = _StatNames().execute(_context("src"), _HashParams())
        assert result.success, result.errors
        assert result.output["rows"] == [{"source": "src", "stats": "dhash,phash,xxhash"}]
        assert result.metadata.evaluator == "test.stat-names"
        assert result.dataset is not None
        assert result.sources is None

    def test_several_sources_arrive_in_order(self):
        result = _StatNames().execute(_context("a", "b"), _HashParams())
        assert [row["source"] for row in result.output["rows"]] == ["a", "b"]
        assert result.dataset is None
        assert result.sources is not None
        assert set(result.sources) == {"a", "b"}

    def test_the_view_is_applied_before_producing(self):
        context = _context("src", view_operations=[ViewOperation(type="Limit", params={"size": 4})])
        result = _StatNames().execute(context, _HashParams())
        assert result.dataset is not None
        assert len(result.dataset) == 4

    def test_the_wrong_params_fail_the_result(self):
        result = _StatNames().execute(_context("src"), _MetadataParams())
        assert not result.success
        assert result.errors == ["Expected _HashParams, got _MetadataParams"]
        assert result.output == {}

    def test_an_exception_becomes_a_failed_result(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.ERROR):
            result = _Boom().execute(_context("src"), _HashParams())
        assert not result.success
        assert result.errors == ["Evaluator execution failed: boom"]
        assert "boom" in caplog.text

    def test_a_kind_without_a_producer_fails_the_result(self):
        result = _NeedsMetadata().execute(_context("src"), _MetadataParams())
        assert not result.success
        assert "No producer for metadata" in result.errors[0]

    def test_a_stats_consumer_without_stats_request_fails_the_result(self):
        result = _NeedsStatsOnly().execute(_context("src"), _StatsOnly())
        assert not result.success
        assert "consumes stats but declares no stats_request()" in result.errors[0]

    def test_a_cluster_consumer_without_clustering_fields_fails_the_result(self):
        result = _NeedsClustersWithoutFields().execute(_context("src"), _ClustersWithoutFields())
        assert not result.success
        assert "consumes clusters but declares no clustering fields" in result.errors[0]

    def test_cluster_mode_without_an_extractor_fails_the_result(self):
        result = _NeedsClusterParams().execute(_context("src"), _ClusterParams())
        assert not result.success
        assert "Cluster mode needs an extractor on the task." in result.errors[0]

    def test_a_second_run_reads_the_cache(self, tmp_path):
        cache = DatasetCache.get_or_create(cache_dir=tmp_path, name="toy", cache_key="k")
        with patch.object(cache_module, "_do_compute_stats", wraps=cache_module._do_compute_stats) as compute:
            _StatNames().execute(_context("src", cache=cache), _HashParams())
            _StatNames().execute(_context("src", cache=cache), _HashParams())
        assert compute.call_count == 1


class TestRegistry:
    @pytest.fixture(autouse=True)
    def _only_the_test_evaluator(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(evaluator_module, "_EVALUATORS", {})
        monkeypatch.setattr(evaluator_module, "_registered", lambda: [_StatNames])

    def test_lookup_by_type(self):
        assert isinstance(get_evaluator("test.stat-names"), _StatNames)

    def test_an_unknown_type_names_the_known_ones(self):
        with pytest.raises(ValueError, match=r"Unknown evaluator: 'nope'. Available: \['test.stat-names'\]"):
            get_evaluator("nope")

    def test_the_listing_says_what_each_consumes(self):
        assert list_evaluators() == [
            {
                "name": "test.stat-names",
                "description": "Statistic names per source",
                "consumes": "stats",
                "sources": "1+",
            }
        ]
