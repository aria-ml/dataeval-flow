"""data-bias: class balance and the metadata factors judged as chained steps, with no extractor."""

import re
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.data_bias import DataBiasConfig, DataBiasWorkflow
from tests.chain_toys import chain_pipeline
from tests.golden.coverage import CoverageDetections, CoverageImages


def _run(entry: dict[str, Any], dataset: Any) -> ChainResult:
    DatasetCache.clear_instances()
    config = chain_pipeline(
        workflows=[{"name": "w", "type": "data-bias", **entry}],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"]}],
        datasets={"src": dataset},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    return result


def _names(config: DataBiasConfig) -> list[str]:
    return [step["name"] for step in DataBiasWorkflow.chain(config).steps]  # type: ignore[index]


def test_its_chain() -> None:
    assert _names(DataBiasConfig(name="w")) == [
        "label-health",
        "class-imbalance",
        "factor-summary",
        "balance",
        "diversity",
        "shortcut-risk",
        "parity",
        "factor-parity",
        "factor-gaps",
        "factor-coverage-gaps",
    ]


def test_factor_gaps_false_leaves_out_the_gap_analysis() -> None:
    names = _names(DataBiasConfig.model_validate({"name": "w", "factor-gaps": False}))
    assert not {"factor-gaps", "factor-coverage-gaps"} & set(names)


def test_crossed_class_imbalance_bands_are_refused_where_they_were_written() -> None:
    message = "`info` (3.0) must not exceed `warning` (2.0)."
    with pytest.raises(ValidationError, match=re.escape(message)) as caught:
        DataBiasConfig.model_validate({"name": "w", "checks": {"class-imbalance": {"warning": 2.0, "info": 3.0}}})
    assert caught.value.errors()[0]["loc"][0] == "checks"


def test_a_lowered_warning_leaves_info_unset() -> None:
    config = DataBiasConfig.model_validate({"name": "w", "checks": {"class-imbalance": {"warning": 1.5}}})
    assert config.checks.class_imbalance.info is None  # `info` no longer defaults to 2.0 (the step's own default)
    result = _run({"checks": {"class-imbalance": {"warning": 1.5}}}, CoverageDetections())
    assert result.success, result.errors
    assert next(f.severity for f in result.findings if f.title == "Class Imbalance") == "warning"


def test_a_dumped_config_reloads() -> None:
    config = DataBiasConfig.model_validate({"name": "w", "factor-gaps": False, "diversity": {"method": "shannon"}})
    assert DataBiasConfig.model_validate(config.model_dump()) == config
    assert DataBiasConfig.model_validate(config.model_dump(by_alias=False)) == config


def test_a_run_judges_balance_and_each_factor_without_an_extractor() -> None:
    result = _run({}, CoverageImages())  # `site` follows the class except every tenth item
    assert result.success, result.errors
    by_title = {finding.title: finding for finding in result.findings}
    assert list(by_title) == ["Class Imbalance", "Shortcut Risk", "Factor Parity", "Factor Coverage Gaps"]
    assert by_title["Shortcut Risk"].severity == "warning"
    assert by_title["Factor Parity"].severity == "warning"
    assert "factors associated with the class: site (V=" in (by_title["Factor Parity"].brief or "")


def test_metadata_with_no_factors_reports_the_factor_checks_not_assessed() -> None:
    result = _run({}, CoverageImages(factors=False))
    assert result.success, result.errors
    by_title = {finding.title: finding for finding in result.findings}
    assert by_title["Class Imbalance"].brief != "not assessed"
    for title in ("Shortcut Risk", "Factor Parity", "Factor Coverage Gaps"):
        assert by_title[title].brief == "not assessed"


def test_its_binning_record_is_one_record() -> None:
    result = _run({}, CoverageImages())
    record = result.metadata.metadata_binning
    assert record is not None
    assert "per_split" not in record
