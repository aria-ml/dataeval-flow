"""Every config model refuses a key it does not define, so a misspelling fails the load instead of vanishing."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from dataeval_flow import PipelineConfig, load_config
from dataeval_flow.config._merge import merge_config_folder
from dataeval_flow.config._models import LoggingConfig, ResultConfig, SourceConfig
from dataeval_flow.config._schemas import (
    DatasetProtocolConfig,
    MetadataPolicyConfig,
    PreprocessorConfig,
    TaskConfig,
    ViewConfig,
)
from dataeval_flow.config._schemas._preprocessor import PreprocessingStep
from dataeval_flow.config._schemas._view import ViewOperation
from dataeval_flow.workflows.data_analysis._config import DataAnalysisHealthThresholds
from dataeval_flow.workflows.data_cleaning import DataCleaningConfig
from dataeval_flow.workflows.data_cleaning._config import DataCleaningChecks
from dataeval_flow.workflows.data_coverage._config import (
    ClassCoverageSettings,
    CoverageSettings,
    CropSettings,
    DataCoverageChecks,
    DataCoverageClassImbalanceSettings,
    DataCoverageConfig,
    DataCoverageUncoveredItemsSettings,
    DimensionalCompletenessSettings,
    FactorCoverageGapsSettings,
    GapSettings,
)
from dataeval_flow.workflows.drift_monitoring import DriftMonitoringChecks
from dataeval_flow.workflows.metadata_triage._config import MetadataTriageConfig
from dataeval_flow.workflows.ood_detection import OODDetectionChecks
from tests.chain_toys import chain_pipeline

pytestmark = pytest.mark.required


class TestTopLevelKeys:
    """A pipeline file's top-level keys: a misspelled section used to be dropped without a word."""

    def test_a_misspelled_section_fails_the_load_naming_the_section_it_resembles(self, tmp_path: Path) -> None:
        path = tmp_path / "pipeline.yaml"
        path.write_text("exprots:\n  - {name: dataset, source: merged}\n")  # codespell:ignore

        with pytest.raises(ValidationError, match=r"'exprots'.*did you mean 'exports'"):
            load_config(path)

    def test_a_key_resembling_no_section_is_refused_without_a_guess(self) -> None:
        with pytest.raises(ValidationError, match="'frobnicate'") as info:
            PipelineConfig.model_validate({"frobnicate": 1})

        assert "did you mean" not in str(info.value)

    def test_the_legacy_selections_key_is_still_read_as_views(self) -> None:
        with pytest.warns(DeprecationWarning, match="selections"):
            config = PipelineConfig.model_validate({"selections": [{"name": "first", "operations": []}]})

        assert [view.name for view in config.views or ()] == ["first"]

    def test_the_schema_lets_an_editor_flag_a_misspelled_section(self) -> None:
        # What an editor validating a file against params.schema.json reads to underline an unknown key.
        assert PipelineConfig.model_json_schema().get("additionalProperties") is False


# Each entry is valid as it stands, so the key the test adds is the only thing wrong with it.
_NESTED = [
    pytest.param(TaskConfig, {"name": "t", "workflow": "w", "sources": "s"}, id="task"),
    pytest.param(SourceConfig, {"name": "s", "dataset": "d"}, id="source"),
    pytest.param(ViewConfig, {"name": "v", "operations": []}, id="view"),
    pytest.param(ViewOperation, {"type": "Limit", "params": {"size": 5}}, id="view-operation"),
    pytest.param(MetadataPolicyConfig, {"name": "m"}, id="metadata-policy"),
    pytest.param(PreprocessorConfig, {"name": "p", "steps": []}, id="preprocessor"),
    pytest.param(PreprocessingStep, {"step": "ToRGB"}, id="preprocessing-step"),
    pytest.param(LoggingConfig, {}, id="logging"),
    pytest.param(ResultConfig, {}, id="result"),
    pytest.param(DatasetProtocolConfig, {"name": "d", "format": "maite", "dataset": []}, id="maite"),
    pytest.param(DataCoverageConfig, {}, id="workflow"),
    pytest.param(MetadataTriageConfig, {}, id="another-workflow"),
    pytest.param(DataAnalysisHealthThresholds, {}, id="analysis-thresholds"),
    pytest.param(DataCleaningChecks, {}, id="cleaning-thresholds"),
    pytest.param(DataCoverageChecks, {}, id="coverage-thresholds"),
    pytest.param(DataCoverageClassImbalanceSettings, {}, id="coverage-class-imbalance"),
    pytest.param(FactorCoverageGapsSettings, {}, id="factor-coverage-gaps"),
    pytest.param(ClassCoverageSettings, {}, id="coverage-class-coverage"),
    pytest.param(DataCoverageUncoveredItemsSettings, {}, id="coverage-uncovered-rate"),
    pytest.param(DimensionalCompletenessSettings, {}, id="coverage-completeness-score"),
    pytest.param(CoverageSettings, {}, id="coverage-settings"),
    pytest.param(CropSettings, {}, id="coverage-crops"),
    pytest.param(GapSettings, {}, id="coverage-gap-settings"),
    pytest.param(DriftMonitoringChecks, {}, id="drift-thresholds"),
    pytest.param(OODDetectionChecks, {}, id="ood-thresholds"),
]


@pytest.mark.parametrize(("model", "data"), _NESTED)
def test_a_nested_entry_refuses_a_key_it_does_not_define(model: type[BaseModel], data: dict[str, Any]) -> None:
    model.model_validate(data)  # valid without the extra key

    with pytest.raises(ValidationError) as info:
        model.model_validate({**data, "not_a_field": 1})

    assert [error["loc"] for error in info.value.errors() if error["type"] == "extra_forbidden"] == [("not_a_field",)]


class TestConfigFolder:
    """A folder's files: one holding a misspelled section used to be skipped whole, at DEBUG."""

    def test_a_file_with_a_misspelled_section_fails_the_merge_naming_the_file(self, tmp_path: Path) -> None:
        (tmp_path / "00-base.yaml").write_text("logging:\n  app_level: INFO\n")
        (tmp_path / "01-datasets.yaml").write_text("datasets: []\nexprots: []\n")

        with pytest.raises(ValueError, match=r"01-datasets\.yaml.*'exprots'.*did you mean 'exports'"):
            merge_config_folder(tmp_path)

    def test_a_file_sharing_no_key_with_a_pipeline_is_still_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "00-base.yaml").write_text("logging:\n  app_level: INFO\n")
        (tmp_path / "compose.yaml").write_text("services:\n  flow:\n    image: dataeval-flow\n")

        assert merge_config_folder(tmp_path) == {"logging": {"app_level": "INFO"}}


_STILL_WRITES_MODE = [
    pytest.param(
        DataCleaningConfig, {"outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}}, id="data-cleaning"
    ),
    pytest.param(DataCoverageConfig, {}, id="data-coverage"),
    pytest.param(MetadataTriageConfig, {}, id="metadata-triage"),
]


@pytest.mark.parametrize(("model", "data"), _STILL_WRITES_MODE)
def test_a_workflow_entry_that_still_writes_mode_is_refused(model: type[BaseModel], data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError) as info:
        model.model_validate({**data, "mode": "preparatory"})
    assert [error["loc"] for error in info.value.errors() if error["type"] == "extra_forbidden"] == [("mode",)]


def test_a_pipeline_whose_workflow_still_writes_mode_fails_to_load_naming_it() -> None:
    entry = {"name": "c", "type": "data-cleaning", "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}}
    with pytest.raises(ValidationError) as info:
        chain_pipeline(workflows=[{**entry, "mode": "advisory"}])
    assert [error["loc"][-1] for error in info.value.errors() if error["type"] == "extra_forbidden"] == ["mode"]
