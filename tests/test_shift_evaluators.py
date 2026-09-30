"""The shift evaluators against real DataEval: a reference, and test images brightened out of its distribution."""

from typing import Any

import pytest
from dataeval.utils.thresholds import resolve_threshold
from pydantic import ValidationError

from dataeval_flow import run
from dataeval_flow.evaluators.shift import (
    ChunkedDriftConfig,
    DriftDomainClassifierConfig,
    DriftDomainClassifierResult,
    DriftKNeighborsConfig,
    DriftKNeighborsResult,
    DriftMMDConfig,
    DriftMMDResult,
    DriftUnivariateConfig,
    DriftUnivariateResult,
    DriftWassersteinConfig,
    DriftWassersteinResult,
    OODDomainClassifierConfig,
    OODDomainClassifierResult,
    OODKNeighborsConfig,
    OODKNeighborsResult,
)
from dataeval_flow.evaluators.shift._evaluator import chunked_arguments
from tests.evaluator_toys import FLAT, output_json, shifted_sources

_DRIFT: list[tuple[Any, type]] = [
    (DriftUnivariateConfig, DriftUnivariateResult),
    (DriftMMDConfig, DriftMMDResult),
    (DriftKNeighborsConfig, DriftKNeighborsResult),
    (DriftDomainClassifierConfig, DriftDomainClassifierResult),
]


class TestDrift:
    @pytest.mark.parametrize(("config_type", "result_type"), _DRIFT)
    def test_brightened_test_data_drifts(self, config_type: Any, result_type: type):
        result = run(config_type(), shifted_sources(), extractor=FLAT)
        assert isinstance(result, result_type)
        assert result.success, result.errors
        data = output_json(result)["data"]
        assert {"drifted", "distance", "threshold", "metric_name", "details", "feature_names"} <= set(data)
        assert data["drifted"] is True

    def test_wasserstein_calibrates_on_the_middle_source(self):
        result = run(DriftWassersteinConfig(), shifted_sources(validation=True), extractor=FLAT)
        assert isinstance(result, DriftWassersteinResult)
        assert result.success, result.errors
        assert output_json(result)["data"]["drifted"] is True

    def test_a_setting_reaches_dataeval(self):
        result = run(DriftUnivariateConfig(method="cvm"), shifted_sources(), extractor=FLAT)
        assert result.success, result.errors
        assert output_json(result)["data"]["metric_name"] == "cvm_distance"

    def test_chunking_tests_each_chunk(self):
        config = DriftMMDConfig(chunking=ChunkedDriftConfig(chunk_count=4))
        result = run(config, shifted_sources(), extractor=FLAT)
        assert result.success, result.errors
        details = output_json(result)["data"]["details"]
        assert details["shape"] == "table"
        assert len(details["rows"]) == 4

    def test_wasserstein_chunks_with_its_validation_source(self):
        config = DriftWassersteinConfig(chunking=ChunkedDriftConfig(chunk_count=4))
        result = run(config, shifted_sources(validation=True), extractor=FLAT)
        assert result.success, result.errors


class TestChunkedDriftConfig:
    @pytest.mark.parametrize(
        "threshold", ["zscore", 2.5, ["zscore", 2.5], ["iqr", [1.0, 3.0]], ["zscore", 3.0, [0.0, 1.0]]]
    )
    def test_every_yaml_threshold_resolves(self, threshold: Any):
        chunking = ChunkedDriftConfig.model_validate({"chunk_count": 4, "threshold": threshold})
        assert repr(chunked_arguments(chunking)["threshold"]) == repr(resolve_threshold(threshold))

    def test_an_unset_threshold_is_left_to_dataeval(self):
        assert chunked_arguments(ChunkedDriftConfig(chunk_size=10)) == {"chunk_size": 10}

    def test_a_size_or_a_count_is_required(self):
        with pytest.raises(ValidationError, match="Set `chunk_size` or `chunk_count`"):
            ChunkedDriftConfig()

    def test_incomplete_reaches_dataeval(self):
        chunking = ChunkedDriftConfig(chunk_size=10, incomplete="drop")
        assert chunked_arguments(chunking) == {"chunk_size": 10, "incomplete": "drop"}

    def test_incomplete_needs_a_chunk_size(self):
        """DataEval refuses it beside `chunk_count`; the config load says so first."""
        with pytest.raises(ValidationError, match="`incomplete` applies only to `chunk_size`"):
            ChunkedDriftConfig(chunk_count=4, incomplete="drop")

    def test_a_misspelled_key_fails_the_load(self):
        with pytest.raises(ValidationError, match="chunk_counts"):
            DriftMMDConfig.model_validate({"chunking": {"chunk_counts": 4}})


_OOD: list[tuple[Any, type]] = [
    (OODKNeighborsConfig, OODKNeighborsResult),
    (OODDomainClassifierConfig, OODDomainClassifierResult),
]


class TestOOD:
    @pytest.mark.parametrize(("config_type", "result_type"), _OOD)
    def test_each_test_item_is_scored(self, config_type: Any, result_type: type):
        result = run(config_type(), shifted_sources(), extractor=FLAT)
        assert isinstance(result, result_type)
        assert result.success, result.errors
        data = output_json(result)["data"]
        assert len(data["is_ood"]) == 40
        assert len(data["instance_score"]) == 40

    def test_the_distance_metric_reaches_dataeval(self):
        """Brightening barely turns an embedding, so cosine distance misses it and euclidean does not."""
        cosine = output_json(run(OODKNeighborsConfig(distance_metric="cosine"), shifted_sources(), extractor=FLAT))
        euclidean = output_json(
            run(OODKNeighborsConfig(distance_metric="euclidean"), shifted_sources(), extractor=FLAT)
        )
        assert sum(euclidean["data"]["is_ood"]) > sum(cosine["data"]["is_ood"])
        assert sum(euclidean["data"]["is_ood"]) >= 30

    def test_the_console_report_cuts_the_per_item_lists(self):
        result = run(OODKNeighborsConfig(), shifted_sources(), extractor=FLAT)
        assert "… and 30 more" in result.report(detailed=False)
        assert "… and 30 more" not in result.report(detailed=True)
