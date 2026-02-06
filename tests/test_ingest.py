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
        assert config.data_cleaning is not None
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


class TestLoadConfigFolder:
    """Test load_config_folder function."""

    def test_load_from_folder(self, tmp_path: Path):
        """Load config from folder with multiple YAML files."""
        from dataeval_app.ingest import load_config_folder

        # Create test YAML files
        (tmp_path / "00-base.yaml").write_text(
            "data_cleaning:\n"
            "  outlier_method: iqr\n"
            "  outlier_use_dimension: true\n"
            "  outlier_use_pixel: true\n"
            "  outlier_use_visual: true\n"
            "  outlier_threshold: null\n"
        )
        (tmp_path / "01-models.yaml").write_text(
            "models:\n  - name: resnet50\n    path: ./resnet50.onnx\n    embedding_layer: flatten0\n"
        )

        config = load_config_folder(tmp_path)
        assert config.data_cleaning is not None
        assert config.data_cleaning.outlier_method == "iqr"
        assert config.models is not None
        assert len(config.models) == 1
        assert config.models[0].name == "resnet50"

    def test_load_from_folder_not_directory(self, tmp_path: Path):
        """Raises ValueError if path is not a directory."""
        from dataeval_app.ingest import load_config_folder

        file_path = tmp_path / "file.yaml"
        file_path.write_text("key: value\n")

        with pytest.raises(ValueError, match="not a directory"):
            load_config_folder(file_path)


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


class TestP1SchemaClasses:
    """Test P1 schema classes (DatasetConfig, PreprocessorConfig, SelectionConfig, TaskConfig)."""

    def test_split_config_defaults(self):
        """SplitConfig has correct defaults."""
        from dataeval_app.ingest import SplitConfig

        split = SplitConfig()
        assert split.num_folds is None
        assert split.stratify is False
        assert split.split_on is None
        assert split.test_frac is None
        assert split.val_frac is None

    def test_split_config_custom(self):
        """SplitConfig accepts custom values."""
        from dataeval_app.ingest import SplitConfig

        split = SplitConfig(
            num_folds=5,
            stratify=True,
            split_on=["category_id"],
            test_frac=0.2,
            val_frac=0.1,
        )
        assert split.num_folds == 5
        assert split.stratify is True
        assert split.split_on == ["category_id"]
        assert split.test_frac == 0.2
        assert split.val_frac == 0.1

    def test_dataset_config_with_list_splits(self):
        """DatasetConfig with list of split names."""
        from dataeval_app.ingest import DatasetConfig

        dataset = DatasetConfig(
            name="cppe5",
            format="huggingface",
            path="./cppe5",
            splits=["train", "test", "validation"],
        )
        assert dataset.name == "cppe5"
        assert dataset.format == "huggingface"
        assert dataset.path == "./cppe5"
        assert dataset.splits == ["train", "test", "validation"]
        assert dataset.metadata_auto_bin_method is None
        assert dataset.metadata_ignore == []
        assert dataset.metadata_continuous_factor_bins is None

    def test_dataset_config_with_split_config(self):
        """DatasetConfig with SplitConfig object."""
        from dataeval_app.ingest import DatasetConfig, SplitConfig

        split = SplitConfig(num_folds=5, stratify=True)
        dataset = DatasetConfig(
            name="retail",
            format="coco",
            path="./retail",
            splits=split,
            metadata_auto_bin_method="clusters",
            metadata_ignore=["id", "source"],
            metadata_continuous_factor_bins={"age": [0.0, 18.0, 65.0, 100.0]},
        )
        assert dataset.splits == split
        assert dataset.metadata_auto_bin_method == "clusters"
        assert dataset.metadata_ignore == ["id", "source"]
        assert dataset.metadata_continuous_factor_bins == {"age": [0.0, 18.0, 65.0, 100.0]}

    def test_dataset_config_missing_required_raises(self):
        """DatasetConfig requires name, format, path, splits."""
        from dataeval_app.ingest import DatasetConfig

        with pytest.raises(ValidationError, match="name"):
            DatasetConfig(format="coco", path="./data", splits=["train"])  # type: ignore[call-arg]

    def test_dataset_config_invalid_format_raises(self):
        """DatasetConfig rejects invalid format."""
        from dataeval_app.ingest import DatasetConfig

        with pytest.raises(ValidationError, match="format"):
            DatasetConfig(
                name="test",
                format="invalid",  # type: ignore[arg-type]
                path="./data",
                splits=["train"],
            )

    def test_preprocessor_config_valid(self):
        """PreprocessorConfig with valid steps."""
        from dataeval_app.ingest import PreprocessorConfig
        from dataeval_app.utility import PreprocessingStep

        config = PreprocessorConfig(
            name="resnet50_preprocessor",
            steps=[
                PreprocessingStep(step="Resize", params={"size": [224, 224]}),
                PreprocessingStep(step="ToTensor"),
            ],
        )
        assert config.name == "resnet50_preprocessor"
        assert len(config.steps) == 2
        assert config.steps[0].step == "Resize"
        assert config.steps[1].step == "ToTensor"

    def test_selection_step_valid(self):
        """SelectionStep with type and params."""
        from dataeval_app.ingest import SelectionStep

        step = SelectionStep(type="Limit", params={"size": 10000})
        assert step.type == "Limit"
        assert step.params == {"size": 10000}

    def test_selection_step_default_params(self):
        """SelectionStep params defaults to empty dict."""
        from dataeval_app.ingest import SelectionStep

        step = SelectionStep(type="ClassBalance")
        assert step.type == "ClassBalance"
        assert step.params == {}

    def test_selection_config_valid(self):
        """SelectionConfig with named pipeline."""
        from dataeval_app.ingest import SelectionConfig, SelectionStep

        config = SelectionConfig(
            name="training_subset",
            steps=[
                SelectionStep(type="Limit", params={"size": 10000}),
                SelectionStep(type="ClassFilter", params={"classes": [0, 1, 2]}),
            ],
        )
        assert config.name == "training_subset"
        assert len(config.steps) == 2
        assert config.steps[0].type == "Limit"
        assert config.steps[1].type == "ClassFilter"

    def test_task_config_valid(self):
        """TaskConfig with all optional fields."""
        from dataeval_app.ingest import TaskConfig

        task = TaskConfig(
            name="data_cleaning",
            dataset="cppe5",
            model="resnet50",
            preprocessor="resnet50_preprocessor",
            selection="training_subset",
            params={"outlier_method": "modzscore"},
            output_format="json",
        )
        assert task.name == "data_cleaning"
        assert task.dataset == "cppe5"
        assert task.model == "resnet50"
        assert task.preprocessor == "resnet50_preprocessor"
        assert task.selection == "training_subset"
        assert task.params == {"outlier_method": "modzscore"}
        assert task.output_format == "json"

    def test_task_config_minimal(self):
        """TaskConfig with only required fields."""
        from dataeval_app.ingest import TaskConfig

        task = TaskConfig(name="minimal", dataset="test")
        assert task.name == "minimal"
        assert task.dataset == "test"
        assert task.model is None
        assert task.preprocessor is None
        assert task.selection is None
        assert task.params == {}
        assert task.output_format == "json"

    def test_task_config_invalid_output_format_raises(self):
        """TaskConfig rejects invalid output_format."""
        from dataeval_app.ingest import TaskConfig

        with pytest.raises(ValidationError, match="output_format"):
            TaskConfig(
                name="test",
                dataset="test",
                output_format="invalid",  # type: ignore[arg-type]
            )

    def test_workflow_config_with_all_p1_schemas(self, tmp_path: Path):
        """WorkflowConfig loads all P1 schema sections."""
        from dataeval_app.ingest import load_config_folder

        # Create config with all P1 sections
        (tmp_path / "config.yaml").write_text(
            "datasets:\n"
            "  - name: cppe5\n"
            "    format: huggingface\n"
            "    path: ./cppe5\n"
            "    splits:\n"
            "      - train\n"
            "      - test\n"
            "preprocessors:\n"
            "  - name: basic\n"
            "    steps:\n"
            "      - step: ToTensor\n"
            "selections:\n"
            "  - name: subset\n"
            "    steps:\n"
            "      - type: Limit\n"
            "        params:\n"
            "          size: 1000\n"
            "tasks:\n"
            "  - name: clean\n"
            "    dataset: cppe5\n"
            "    preprocessor: basic\n"
            "    selection: subset\n"
        )

        config = load_config_folder(tmp_path)

        # Verify datasets
        assert config.datasets is not None
        assert len(config.datasets) == 1
        assert config.datasets[0].name == "cppe5"
        assert config.datasets[0].splits == ["train", "test"]

        # Verify preprocessors
        assert config.preprocessors is not None
        assert len(config.preprocessors) == 1
        assert config.preprocessors[0].name == "basic"

        # Verify selections
        assert config.selections is not None
        assert len(config.selections) == 1
        assert config.selections[0].name == "subset"

        # Verify tasks
        assert config.tasks is not None
        assert len(config.tasks) == 1
        assert config.tasks[0].name == "clean"
        assert config.tasks[0].dataset == "cppe5"
