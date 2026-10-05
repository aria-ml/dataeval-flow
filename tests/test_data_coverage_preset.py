"""data-coverage: embeddings, labels and metadata judged as chained steps (coverage spec §4)."""

import re
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import MatrixResult, PipelineConfig, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.data_coverage import DataCoverageConfig, DataCoverageWorkflow
from dataeval_flow.workflows.data_coverage._config import _MOVED, _THRESHOLDS_MOVED
from tests.chain_toys import chain_pipeline
from tests.golden.coverage import CoverageDetections, CoverageImages


def _pipeline(entry: dict[str, Any], dataset: Any, *, extractor: bool = False) -> PipelineConfig:
    DatasetCache.clear_instances()
    task = {"name": "t", "workflow": "w", "sources": ["src"]} | ({"extractor": "flat"} if extractor else {})
    return chain_pipeline(
        workflows=[{"name": "w", "type": "data-coverage", **entry}],
        tasks=[task],
        datasets={"src": dataset},
        extractor=extractor,
    )


def _run(entry: dict[str, Any], dataset: Any, *, extractor: bool = False) -> ChainResult:
    result = run_tasks(_pipeline(entry, dataset, extractor=extractor))["t"]
    assert isinstance(result, ChainResult)
    return result


def _names(config: DataCoverageConfig) -> list[str]:
    return [step["name"] for step in DataCoverageWorkflow.chain(config).steps]  # type: ignore[index]


def test_its_chain_follows_legacy_s_finding_order() -> None:
    assert _names(DataCoverageConfig(name="w")) == [
        "crops",
        "coverage",
        "class-coverage",
        "completeness",
        "completeness-check",
        "labels",
        "labels-check",
        "summary",
        "balance",
        "diversity",
        "gaps",
        "gaps-check",
        "worklist",
        "shortfall",
    ]


def test_naive_coverage_adds_the_uncovered_rate() -> None:
    assert "uncovered" in _names(DataCoverageConfig(name="w", coverage={"method": "naive"}))  # type: ignore[arg-type]


def test_settings_leave_out_their_steps() -> None:
    names = _names(DataCoverageConfig(name="w", completeness=False, gaps=None))
    assert not {"completeness", "completeness-check", "gaps", "gaps-check"} & set(names)


@pytest.mark.parametrize("key", sorted(_MOVED))
def test_a_legacy_field_is_refused_with_its_replacement(key: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(f"data-coverage's `{key}` is refused: {_MOVED[key]}.")):
        DataCoverageConfig.model_validate({"name": "w", key: 1})


@pytest.mark.parametrize("key", sorted(_THRESHOLDS_MOVED))
def test_a_flat_threshold_is_refused_with_its_check_type_key(key: str) -> None:
    message = f"`health_thresholds.{key}` is refused: it is {_THRESHOLDS_MOVED[key]}."
    with pytest.raises(ValidationError, match=re.escape(message)):
        DataCoverageConfig.model_validate({"name": "w", "health_thresholds": {key: 1}})


@pytest.mark.parametrize(
    ("legacy", "names"),
    [
        ({"ontology": {"a": None}}, "label-space"),
        ({"metadata_exclude": ["id"]}, "metadata:"),
    ],
)
def test_a_refusal_names_what_replaced_it(legacy: dict[str, Any], names: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(names)):
        DataCoverageConfig(name="w", **legacy)


def test_a_dumped_config_reloads() -> None:
    config = DataCoverageConfig(name="w", coverage={"method": "naive"})  # type: ignore[arg-type]
    assert DataCoverageConfig.model_validate(config.model_dump()) == config


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"class-imbalance": {"warning": 2.0, "info": 3.0}}, "`info` (3.0) must not exceed `warning` (2.0)."),
        ({"dimensional-completeness": {"warning": 0.9, "info": 0.7}}, "`warning` (0.9) must not exceed `info` (0.7)."),
    ],
)
def test_explicitly_crossed_bands_are_refused_where_they_were_written(limits: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(message)) as caught:
        DataCoverageConfig.model_validate({"name": "w", "checks": limits})
    assert caught.value.errors()[0]["loc"][0] == "checks"


def test_a_ratio_under_the_fixed_band_moves_the_band_as_legacy_did() -> None:
    config = DataCoverageConfig.model_validate({"name": "w", "checks": {"class-imbalance": {"warning": 1.5}}})
    assert config.checks.class_imbalance.info == 1.5
    result = _run({"checks": {"class-imbalance": {"warning": 1.5}}}, CoverageDetections())
    assert result.success, result.errors
    assert next(f.severity for f in result.findings if f.title == "Class Imbalance") == "warning"


def test_a_warning_over_the_fixed_band_moves_the_band_as_legacy_did() -> None:
    config = DataCoverageConfig.model_validate({"name": "w", "checks": {"dimensional-completeness": {"warning": 0.9}}})
    assert config.checks.dimensional_completeness.info == 0.9
    assert DataCoverageConfig.model_validate(config.model_dump()) == config
    assert DataCoverageConfig.model_validate(config.model_dump(by_alias=False)) == config


def test_a_matrix_over_a_warning_that_crosses_the_fixed_band_runs() -> None:
    DatasetCache.clear_instances()
    config = chain_pipeline(
        workflows=[{"name": "w", "type": "data-coverage"}],
        tasks=[
            {
                "name": "t",
                "workflow": "w",
                "sources": ["src"],
                "extractor": "flat",
                "matrix": {"checks.dimensional-completeness.warning": [0.5, 0.9]},
            }
        ],
        datasets={"src": CoverageImages()},
        extractor=True,
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, MatrixResult)
    assert all(isinstance(run.result, ChainResult) and run.result.success for run in result.runs)


def test_snake_case_threshold_names_reload() -> None:
    config = DataCoverageConfig(name="w")
    assert DataCoverageConfig.model_validate(config.model_dump(by_alias=False)) == config


def test_a_flat_number_under_a_snake_case_name_is_still_refused() -> None:
    with pytest.raises(ValidationError, match=re.escape("`health_thresholds.uncovered_rate` is refused: it is")):
        DataCoverageConfig.model_validate({"name": "w", "health_thresholds": {"uncovered_rate": 5}})


def test_every_box_dropped_says_there_is_nothing_to_embed() -> None:
    result = _run({"crops": {"min_size": 10000}}, CoverageDetections(), extractor=True)
    description = next(f.description for f in result.findings if f.title == "Class Coverage") or ""
    assert "no items to embed" in description
    assert description.endswith(".")
    assert not description.endswith("..")


def test_detection_data_with_no_extractor_runs_and_reports_not_assessed() -> None:
    result = _run({}, CoverageDetections())
    assert result.success, result.errors
    by_title = {finding.title: finding for finding in result.findings}
    assert by_title["Class Coverage"].brief == "not assessed"
    assert by_title["Dimensional Completeness"].brief == "not assessed"
    assert by_title["Class Imbalance"].severity == "warning"


def test_classification_data_reads_its_embeddings_once() -> None:
    from unittest.mock import patch

    import dataeval_flow._cache as cache

    real, calls = cache._do_compute_embeddings, []

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    with patch.object(cache, "_do_compute_embeddings", counting):
        result = _run({}, CoverageImages(), extractor=True)
    assert result.success, result.errors
    assert len(calls) == 1


def test_its_binning_record_is_one_record() -> None:
    result = _run({}, CoverageImages())
    record = result.metadata.metadata_binning
    assert record is not None
    assert "per_split" not in record


def test_a_matrix_varies_a_hyphenated_threshold() -> None:
    DatasetCache.clear_instances()
    config = chain_pipeline(
        workflows=[{"name": "w", "type": "data-coverage"}],
        tasks=[
            {
                "name": "t",
                "workflow": "w",
                "sources": ["src"],
                "extractor": "flat",
                "matrix": {"checks.class-coverage.dispersion": [0.5, 1.5]},
            }
        ],
        datasets={"src": CoverageImages()},
        extractor=True,
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, MatrixResult)
    severities = []
    for run in result.runs:
        assert isinstance(run.result, ChainResult)
        severities.append(next(f.severity for f in run.result.findings if f.title == "Class Coverage"))
    assert severities == ["info", "warning"]
