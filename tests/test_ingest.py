"""Tests for ingest package (parameters, outputs, base classes)."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from dataeval_app.ingest import (
    DataCleaningOutputs,
    DataCleaningParameters,
    DataCleaningRawOutputs,
    DataCleaningReport,
    Reportable,
    WorkflowOutputsBase,
    WorkflowParametersBase,
    WorkflowReportBase,
    export_params_schema,
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
        """Loads params from YAML file with all required fields."""
        params_file = tmp_path / "params.yaml"
        params_file.write_text(
            "outlier_method: iqr\n"
            "outlier_use_dimension: true\n"
            "outlier_use_pixel: false\n"
            "outlier_use_visual: true\n"
            "outlier_threshold: 2.5\n"
            "mode: preparatory\n"
        )

        params = load_params(params_file)
        assert params.outlier_method == "iqr"
        assert params.outlier_use_dimension is True
        assert params.outlier_use_pixel is False
        assert params.outlier_threshold == 2.5
        assert params.mode == "preparatory"


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
        """export_params_schema creates valid JSON."""
        import json

        schema_path = tmp_path / "schema.json"
        export_params_schema(schema_path)
        content = schema_path.read_text()
        schema = json.loads(content)
        assert "properties" in schema
        assert "outlier_method" in schema["properties"]


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
