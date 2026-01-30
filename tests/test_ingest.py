"""Tests for ingest package (parameters, outputs, base classes)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from dataeval_app.ingest import (
    DataCleaningOutputs,
    DataCleaningParameters,
    DataCleaningRawOutputs,
    DataCleaningReport,
    ModelConfig,
    Reportable,
    WorkflowConfig,
    WorkflowOutputsBase,
    WorkflowParametersBase,
    WorkflowReportBase,
    export_params_schema,
    load_config,
    load_params,
)

# Valid required parameters for reuse in tests
VALID_REQUIRED_PARAMS = {
    "outlier_method": "modzscore",
    "outlier_use_dimension": True,
    "outlier_use_pixel": True,
    "outlier_use_visual": True,
    "outlier_threshold": None,
}


class TestDataCleaningParameters:
    """Test DataCleaningParameters schema."""

    def test_required_fields_missing(self):
        """Missing required fields raise ValidationError (CR-4.14-G-1)."""
        with pytest.raises(ValidationError, match="outlier_method"):
            DataCleaningParameters()  # type: ignore[call-arg]

    def test_required_fields_partial(self):
        """Partial required fields raise ValidationError."""
        with pytest.raises(ValidationError, match="outlier_use"):
            DataCleaningParameters(outlier_method="iqr")  # type: ignore[call-arg]

    def test_required_fields_complete(self):
        """All required fields provided succeeds."""
        params = DataCleaningParameters(**VALID_REQUIRED_PARAMS)
        assert params.outlier_method == "modzscore"
        assert params.outlier_use_dimension is True
        assert params.outlier_use_pixel is True
        assert params.outlier_use_visual is True

    def test_optional_defaults(self):
        """Optional fields have safe defaults."""
        params = DataCleaningParameters(**VALID_REQUIRED_PARAMS)
        assert params.mode == "advisory"

    def test_custom_values(self):
        """Parameters accept custom values."""
        params = DataCleaningParameters(
            outlier_method="iqr",
            outlier_threshold=2.5,
            outlier_use_dimension=False,
            outlier_use_pixel=True,
            outlier_use_visual=True,
            mode="preparatory",
        )
        assert params.outlier_method == "iqr"
        assert params.outlier_threshold == 2.5
        assert params.outlier_use_dimension is False
        assert params.mode == "preparatory"

    def test_invalid_outlier_method(self):
        """Invalid outlier_method raises ValidationError."""
        with pytest.raises(ValidationError, match="outlier_method"):
            DataCleaningParameters(
                outlier_method="invalid",  # type: ignore[arg-type]
                outlier_use_dimension=True,
                outlier_use_pixel=True,
                outlier_use_visual=True,
                outlier_threshold=None,
            )

    def test_negative_threshold_rejected(self):
        """Negative outlier_threshold raises ValidationError."""
        with pytest.raises(ValidationError, match="outlier_threshold"):
            DataCleaningParameters(
                outlier_method="modzscore",
                outlier_use_dimension=True,
                outlier_use_pixel=True,
                outlier_use_visual=True,
                outlier_threshold=-1.0,
            )

    def test_invalid_mode_rejected(self):
        """Invalid mode raises ValidationError."""
        with pytest.raises(ValidationError, match="mode"):
            DataCleaningParameters(
                outlier_method="modzscore",
                outlier_use_dimension=True,
                outlier_use_pixel=True,
                outlier_use_visual=True,
                outlier_threshold=None,
                mode="invalid",  # type: ignore[arg-type]
            )


class TestLoadParams:
    """Test load_params function."""

    def test_missing_file_raises(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_params(Path("/nonexistent/params.yaml"))

    def test_load_from_file(self, tmp_path: Path):
        """Loads params from YAML file with nested data_cleaning section."""
        params_file = tmp_path / "params.yaml"
        params_file.write_text(
            "data_cleaning:\n"
            "  outlier_method: iqr\n"
            "  outlier_use_dimension: true\n"
            "  outlier_use_pixel: false\n"
            "  outlier_use_visual: true\n"
            "  outlier_threshold: 2.5\n"
            "  mode: preparatory\n"
        )

        params = load_params(params_file)
        assert params.outlier_method == "iqr"
        assert params.outlier_use_dimension is True
        assert params.outlier_use_pixel is False
        assert params.outlier_threshold == 2.5
        assert params.mode == "preparatory"

    def test_load_missing_section_raises(self, tmp_path: Path):
        """Missing data_cleaning section raises ValueError."""
        params_file = tmp_path / "params.yaml"
        params_file.write_text("other_key: value\n")

        with pytest.raises(ValueError, match="data_cleaning.*section not found"):
            load_params(params_file)


class TestYAMLValidationEdgeCases:
    """Test YAML validation edge cases for user error scenarios."""

    def test_malformed_yaml_raises(self, tmp_path: Path):
        """Malformed YAML raises yaml.YAMLError."""
        import yaml

        config_file = tmp_path / "params.yaml"
        config_file.write_text("data_cleaning:\n  outlier_method: [unclosed")

        with pytest.raises(yaml.YAMLError):
            load_params(config_file)

    def test_unknown_field_name_raises(self, tmp_path: Path):
        """Unknown field name (typo) shows as missing required field."""
        config_file = tmp_path / "params.yaml"
        # Using wrong field name simulates a typo - Pydantic ignores unknown fields
        # but requires all declared fields, so missing 'outlier_method' raises
        config_file.write_text(
            "data_cleaning:\n"
            "  outler_method: iqr\n"  # wrong field name
            "  outlier_use_dimension: true\n"
            "  outlier_use_pixel: true\n"
            "  outlier_use_visual: true\n"
            "  outlier_threshold: null\n"
        )

        with pytest.raises(ValidationError, match="outlier_method"):
            load_params(config_file)

    def test_wrong_type_string_instead_of_number(self, tmp_path: Path):
        """outlier_threshold as string raises ValidationError."""
        config_file = tmp_path / "params.yaml"
        config_file.write_text(
            "data_cleaning:\n"
            "  outlier_method: iqr\n"
            "  outlier_use_dimension: true\n"
            "  outlier_use_pixel: true\n"
            "  outlier_use_visual: true\n"
            "  outlier_threshold: 'high'\n"  # string not number
        )

        with pytest.raises(ValidationError, match="outlier_threshold"):
            load_params(config_file)

    def test_wrong_type_string_instead_of_bool(self, tmp_path: Path):
        """Boolean field as non-bool string raises ValidationError."""
        config_file = tmp_path / "params.yaml"
        config_file.write_text(
            "data_cleaning:\n"
            "  outlier_method: iqr\n"
            "  outlier_use_dimension: 'enabled'\n"  # string not bool (yes/no/true/false are coerced)
            "  outlier_use_pixel: true\n"
            "  outlier_use_visual: true\n"
            "  outlier_threshold: null\n"
        )

        with pytest.raises(ValidationError, match="outlier_use_dimension"):
            load_params(config_file)


class TestExportParamsSchema:
    """Test export_params_schema function."""

    def test_export_creates_file(self, tmp_path: Path):
        """export_params_schema creates JSON schema file."""
        schema_path = tmp_path / "schema.json"
        export_params_schema(schema_path)
        assert schema_path.exists()

    def test_export_creates_parent_dirs(self, tmp_path: Path):
        """export_params_schema creates parent directories."""
        schema_path = tmp_path / "nested" / "dir" / "schema.json"
        export_params_schema(schema_path)
        assert schema_path.exists()

    def test_export_valid_json(self, tmp_path: Path):
        """export_params_schema creates valid JSON with nested structure."""
        import json

        schema_path = tmp_path / "schema.json"
        export_params_schema(schema_path)
        content = schema_path.read_text()
        schema = json.loads(content)
        assert "properties" in schema
        assert "data_cleaning" in schema["properties"]


class TestUnifiedConfig:
    """Test unified config loading."""

    def test_load_config_unified(self, tmp_path: Path):
        """load_config loads data_cleaning section."""
        config_file = tmp_path / "params.yaml"
        config_file.write_text(
            "data_cleaning:\n"
            "  outlier_method: iqr\n"
            "  outlier_use_dimension: true\n"
            "  outlier_use_pixel: true\n"
            "  outlier_use_visual: true\n"
            "  outlier_threshold: null\n"
        )

        config = load_config(config_file)
        assert isinstance(config, WorkflowConfig)
        assert config.data_cleaning.outlier_method == "iqr"

    def test_load_config_file_not_found(self):
        """load_config raises FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(Path("/nonexistent/params.yaml"))


class TestDataCleaningOutputs:
    """Test output schemas."""

    def test_data_cleaning_raw_outputs_defaults(self):
        """DataCleaningRawOutputs has correct defaults."""
        raw = DataCleaningRawOutputs(dataset_size=100)
        assert raw.dataset_size == 100
        assert raw.duplicates == {}
        assert raw.img_outliers == {}
        assert raw.label_stats == {}
        assert raw.target_outliers is None

    def test_data_cleaning_report(self):
        """DataCleaningReport can be created."""
        report = DataCleaningReport(summary="Test summary")
        assert report.summary == "Test summary"
        assert report.findings == []

    def test_reportable(self):
        """Reportable can be created."""
        item = Reportable(
            report_type="table",
            title="Test",
            data={"key": "value"},
        )
        assert item.report_type == "table"
        assert item.title == "Test"

    def test_reportable_invalid_type(self):
        """Reportable rejects invalid report_type."""
        with pytest.raises(ValidationError, match="report_type"):
            Reportable(
                report_type="invalid",  # type: ignore[arg-type]
                title="Test",
                data={"key": "value"},
            )

    def test_data_cleaning_report_with_findings(self):
        """DataCleaningReport can have findings."""
        finding = Reportable(
            report_type="key_value",
            title="Finding",
            data={"count": 10},
            description="Test finding",
        )
        report = DataCleaningReport(summary="Summary", findings=[finding])
        assert len(report.findings) == 1
        assert report.findings[0].title == "Finding"

    def test_data_cleaning_outputs_combined(self):
        """DataCleaningOutputs combines raw and report."""
        raw = DataCleaningRawOutputs(dataset_size=50)
        report = DataCleaningReport(summary="Complete")
        outputs = DataCleaningOutputs(raw=raw, report=report)

        assert outputs.raw.dataset_size == 50
        assert outputs.report.summary == "Complete"


class TestBaseClasses:
    """Test base classes for workflow parameters and outputs."""

    def test_workflow_parameters_base(self):
        """WorkflowParametersBase has mode with default."""
        params = WorkflowParametersBase()
        assert params.mode == "advisory"

    def test_workflow_parameters_base_custom_mode(self):
        """WorkflowParametersBase accepts custom mode."""
        params = WorkflowParametersBase(mode="preparatory")
        assert params.mode == "preparatory"

    def test_workflow_outputs_base(self):
        """WorkflowOutputsBase requires dataset_size."""
        outputs = WorkflowOutputsBase(dataset_size=100)
        assert outputs.dataset_size == 100

    def test_workflow_report_base(self):
        """WorkflowReportBase requires summary."""
        report = WorkflowReportBase(summary="Test")
        assert report.summary == "Test"

    def test_inheritance_data_cleaning_parameters(self):
        """DataCleaningParameters inherits from WorkflowParametersBase."""
        assert issubclass(DataCleaningParameters, WorkflowParametersBase)

    def test_inheritance_data_cleaning_raw_outputs(self):
        """DataCleaningRawOutputs inherits from WorkflowOutputsBase."""
        assert issubclass(DataCleaningRawOutputs, WorkflowOutputsBase)

    def test_inheritance_data_cleaning_report(self):
        """DataCleaningReport inherits from WorkflowReportBase."""
        assert issubclass(DataCleaningReport, WorkflowReportBase)


class TestModelConfig:
    """Test ModelConfig schema."""

    def test_model_config_valid(self):
        """ModelConfig accepts valid fields."""
        model = ModelConfig(
            name="resnet50",
            path="./resnet50-v2-7.onnx",
            embedding_layer="resnetv24_flatten0_reshape0",
        )
        assert model.name == "resnet50"
        assert model.path == "./resnet50-v2-7.onnx"
        assert model.embedding_layer == "resnetv24_flatten0_reshape0"

    def test_model_config_missing_field_raises(self):
        """ModelConfig requires all fields."""
        with pytest.raises(ValidationError, match="embedding_layer"):
            ModelConfig(name="test", path="./test.onnx")  # type: ignore[call-arg]

    def test_workflow_config_without_models(self, tmp_path: Path):
        """WorkflowConfig works without models section."""
        config_file = tmp_path / "params.yaml"
        config_file.write_text(
            "data_cleaning:\n"
            "  outlier_method: iqr\n"
            "  outlier_use_dimension: true\n"
            "  outlier_use_pixel: true\n"
            "  outlier_use_visual: true\n"
            "  outlier_threshold: null\n"
        )

        config = load_config(config_file)
        assert config.models is None

    def test_workflow_config_with_models(self, tmp_path: Path):
        """WorkflowConfig loads models list."""
        config_file = tmp_path / "params.yaml"
        config_file.write_text(
            "data_cleaning:\n"
            "  outlier_method: iqr\n"
            "  outlier_use_dimension: true\n"
            "  outlier_use_pixel: true\n"
            "  outlier_use_visual: true\n"
            "  outlier_threshold: null\n"
            "models:\n"
            "  - name: resnet50\n"
            "    path: ./resnet50.onnx\n"
            "    embedding_layer: flatten0\n"
            "  - name: mobilenet\n"
            "    path: ./mobilenet.onnx\n"
            "    embedding_layer: flatten1\n"
        )

        config = load_config(config_file)
        assert config.models is not None
        assert len(config.models) == 2
        assert config.models[0].name == "resnet50"
        assert config.models[1].name == "mobilenet"
