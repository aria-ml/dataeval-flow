"""Tests for config/workflow parameters, outputs, and base classes."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from dataeval_app.config import (
    ModelConfig,
    WorkflowConfig,
    export_params_schema,
    load_config,
    load_config_folder,
)
from dataeval_app.workflow.base import (
    Reportable,
    WorkflowOutputsBase,
    WorkflowParametersBase,
    WorkflowReportBase,
)
from dataeval_app.workflows.cleaning import (
    DataCleaningOutputs,
    DataCleaningParameters,
    DataCleaningRawOutputs,
    DataCleaningReport,
    load_params,
)

# Valid required parameters for reuse in tests
VALID_REQUIRED_PARAMS = {
    "outlier_method": "modzscore",
    "outlier_flags": ["dimension", "pixel", "visual"],
}


class TestDataCleaningParameters:
    """Test DataCleaningParameters schema."""

    def test_required_fields_missing(self):
        """Missing required fields raise ValidationError (CR-4.14-G-1)."""
        with pytest.raises(ValidationError, match="outlier_method"):
            DataCleaningParameters()  # type: ignore[call-arg]

    def test_required_fields_partial(self):
        """Partial required fields raise ValidationError."""
        with pytest.raises(ValidationError, match="outlier_flags"):
            DataCleaningParameters(outlier_method="iqr")  # type: ignore[call-arg]

    def test_required_fields_complete(self):
        """All required fields provided succeeds."""
        params = DataCleaningParameters(**VALID_REQUIRED_PARAMS)
        assert params.outlier_method == "modzscore"
        assert params.outlier_flags == ["dimension", "pixel", "visual"]

    def test_optional_defaults(self):
        """Optional fields have safe defaults."""
        params = DataCleaningParameters(**VALID_REQUIRED_PARAMS)
        assert params.mode == "advisory"
        assert params.outlier_threshold is None

    def test_custom_values(self):
        """Parameters accept custom values."""
        params = DataCleaningParameters(
            outlier_method="iqr",
            outlier_threshold=2.5,
            outlier_flags=["pixel", "visual"],
            mode="preparatory",
        )
        assert params.outlier_method == "iqr"
        assert params.outlier_threshold == 2.5
        assert params.outlier_flags == ["pixel", "visual"]
        assert params.mode == "preparatory"

    def test_invalid_outlier_method(self):
        """Invalid outlier_method raises ValidationError."""
        with pytest.raises(ValidationError, match="outlier_method"):
            DataCleaningParameters(
                outlier_method="invalid",  # type: ignore[arg-type]
                outlier_flags=["dimension"],
                outlier_threshold=None,
            )

    def test_negative_threshold_rejected(self):
        """Negative outlier_threshold raises ValidationError."""
        with pytest.raises(ValidationError, match="outlier_threshold"):
            DataCleaningParameters(
                outlier_method="modzscore",
                outlier_flags=["dimension"],
                outlier_threshold=-1.0,
            )

    def test_empty_outlier_flags_rejected(self):
        """Empty outlier_flags list raises ValidationError."""
        with pytest.raises(ValidationError, match="outlier_flags"):
            DataCleaningParameters(
                outlier_method="modzscore",
                outlier_flags=[],
                outlier_threshold=None,
            )

    def test_invalid_duplicate_flag_rejected(self):
        """Invalid duplicate_flags value raises ValidationError."""
        with pytest.raises(ValidationError, match="duplicate_flags"):
            DataCleaningParameters(
                outlier_method="modzscore",
                outlier_flags=["dimension"],
                duplicate_flags=["invalid_hash"],  # type: ignore[list-item]
            )

    def test_valid_duplicate_flags_accepted(self):
        """Valid duplicate_flags values are accepted."""
        params = DataCleaningParameters(
            outlier_method="modzscore",
            outlier_flags=["dimension"],
            duplicate_flags=["hash_basic", "hash_d4"],
        )
        assert params.duplicate_flags == ["hash_basic", "hash_d4"]

    def test_invalid_mode_rejected(self):
        """Invalid mode raises ValidationError."""
        with pytest.raises(ValidationError, match="mode"):
            DataCleaningParameters(
                outlier_method="modzscore",
                outlier_flags=["dimension"],
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
            "  outlier_flags:\n"
            "    - dimension\n"
            "    - visual\n"
            "  outlier_threshold: 2.5\n"
            "  mode: preparatory\n"
        )

        params = load_params(params_file)
        assert params.outlier_method == "iqr"
        assert params.outlier_flags == ["dimension", "visual"]
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
            "  outlier_flags:\n"
            "    - dimension\n"
            "    - pixel\n"
            "    - visual\n"
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
            "  outlier_flags:\n"
            "    - dimension\n"
            "    - pixel\n"
            "    - visual\n"
            "  outlier_threshold: 'high'\n"  # string not number
        )

        with pytest.raises(ValidationError, match="outlier_threshold"):
            load_params(config_file)

    def test_invalid_outlier_flag_value(self, tmp_path: Path):
        """Invalid outlier flag value raises ValidationError."""
        config_file = tmp_path / "params.yaml"
        config_file.write_text(
            "data_cleaning:\n  outlier_method: iqr\n  outlier_flags:\n    - invalid_flag\n  outlier_threshold: null\n"
        )

        with pytest.raises(ValidationError, match="outlier_flags"):
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
            "  outlier_flags:\n"
            "    - dimension\n"
            "    - pixel\n"
            "    - visual\n"
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
        assert raw.duplicates == {"items": {}, "targets": {}}
        assert raw.img_outliers == {"issues": [], "count": 0}
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
        # Create test YAML files
        (tmp_path / "00-base.yaml").write_text(
            "data_cleaning:\n"
            "  outlier_method: iqr\n"
            "  outlier_flags:\n"
            "    - dimension\n"
            "    - pixel\n"
            "    - visual\n"
            "  outlier_threshold: null\n"
        )
        (tmp_path / "01-models.yaml").write_text(
            "models:\n"
            "  - name: resnet50\n"
            "    extractor:\n"
            "      type: onnx\n"
            "      model_path: ./resnet50.onnx\n"
            "      output_name: flatten0\n"
        )

        config = load_config_folder(tmp_path)
        assert config.data_cleaning is not None
        assert config.data_cleaning.outlier_method == "iqr"
        assert config.models is not None
        assert len(config.models) == 1
        assert config.models[0].name == "resnet50"

    def test_load_from_folder_not_directory(self, tmp_path: Path):
        """Raises ValueError if path is not a directory."""
        file_path = tmp_path / "file.yaml"
        file_path.write_text("key: value\n")

        with pytest.raises(ValueError, match="not a directory"):
            load_config_folder(file_path)


class TestModelConfig:
    """Test ModelConfig schema."""

    def test_model_config_valid_onnx(self):
        """ModelConfig accepts valid OnnxExtractorConfig."""
        from dataeval_app.config.models import OnnxExtractorConfig

        model = ModelConfig(
            name="resnet50",
            extractor=OnnxExtractorConfig(
                model_path="./resnet50-v2-7.onnx",
                output_name="resnetv24_flatten0_reshape0",
            ),
        )
        assert model.name == "resnet50"
        assert model.extractor.type == "onnx"
        assert model.extractor.model_path == "./resnet50-v2-7.onnx"

    def test_model_config_valid_flatten(self):
        """ModelConfig accepts FlattenExtractorConfig."""
        from dataeval_app.config.models import FlattenExtractorConfig

        model = ModelConfig(name="flat", extractor=FlattenExtractorConfig())
        assert model.extractor.type == "flatten"

    def test_model_config_valid_bovw(self):
        """ModelConfig accepts BoVWExtractorConfig."""
        from dataeval_app.config.models import BoVWExtractorConfig

        model = ModelConfig(name="bovw", extractor=BoVWExtractorConfig(vocab_size=1024))
        assert model.extractor.type == "bovw"
        assert model.extractor.vocab_size == 1024

    def test_model_config_valid_bovw_default(self):
        """BoVWExtractorConfig has default vocab_size."""
        from dataeval_app.config.models import BoVWExtractorConfig

        config = BoVWExtractorConfig()
        assert config.vocab_size == 2048

    def test_bovw_vocab_size_out_of_range_raises(self):
        """BoVWExtractorConfig rejects vocab_size outside 256-4096."""
        from dataeval_app.config.models import BoVWExtractorConfig

        with pytest.raises(ValidationError, match="vocab_size"):
            BoVWExtractorConfig(vocab_size=1)
        with pytest.raises(ValidationError, match="vocab_size"):
            BoVWExtractorConfig(vocab_size=10000)

    def test_model_config_valid_torch(self):
        """ModelConfig accepts TorchExtractorConfig."""
        from dataeval_app.config.models import TorchExtractorConfig

        model = ModelConfig(
            name="resnet_torch",
            extractor=TorchExtractorConfig(
                model_path="./resnet.pt",
                layer_name="layer4",
                device="cpu",
            ),
        )
        assert model.extractor.type == "torch"
        assert model.extractor.model_path == "./resnet.pt"
        assert model.extractor.layer_name == "layer4"

    def test_model_config_valid_uncertainty(self):
        """ModelConfig accepts UncertaintyExtractorConfig."""
        from dataeval_app.config.models import UncertaintyExtractorConfig

        model = ModelConfig(
            name="classifier",
            extractor=UncertaintyExtractorConfig(
                model_path="./classifier.pt",
                preds_type="logits",
                batch_size=64,
            ),
        )
        assert model.extractor.type == "uncertainty"
        assert model.extractor.preds_type == "logits"
        assert model.extractor.batch_size == 64

    def test_extractor_discriminated_union_from_yaml(self, tmp_path: Path):
        """YAML with type field selects correct extractor config via discriminated union."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "models:\n"
            "  - name: onnx_model\n"
            "    extractor:\n"
            "      type: onnx\n"
            "      model_path: ./model.onnx\n"
            "  - name: bovw_model\n"
            "    extractor:\n"
            "      type: bovw\n"
            "      vocab_size: 512\n"
            "  - name: flat_model\n"
            "    extractor:\n"
            "      type: flatten\n"
        )

        config = load_config(config_file)
        assert config.models is not None
        assert len(config.models) == 3

        from dataeval_app.config.models import (
            BoVWExtractorConfig,
            FlattenExtractorConfig,
            OnnxExtractorConfig,
        )

        assert isinstance(config.models[0].extractor, OnnxExtractorConfig)
        assert isinstance(config.models[1].extractor, BoVWExtractorConfig)
        assert config.models[1].extractor.vocab_size == 512
        assert isinstance(config.models[2].extractor, FlattenExtractorConfig)

    def test_extractor_invalid_type_raises(self):
        """Invalid extractor type raises ValidationError."""
        with pytest.raises(ValidationError, match="extractor"):
            ModelConfig(
                name="bad",
                extractor={"type": "invalid", "model_path": "./x"},  # type: ignore[arg-type]
            )

    def test_model_config_missing_extractor_raises(self):
        """ModelConfig requires extractor field."""
        with pytest.raises(ValidationError, match="extractor"):
            ModelConfig(name="test")  # type: ignore[call-arg]

    def test_workflow_config_without_models(self, tmp_path: Path):
        """WorkflowConfig works without models section."""
        config_file = tmp_path / "params.yaml"
        config_file.write_text(
            "data_cleaning:\n"
            "  outlier_method: iqr\n"
            "  outlier_flags:\n"
            "    - dimension\n"
            "    - pixel\n"
            "    - visual\n"
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
            "  outlier_flags:\n"
            "    - dimension\n"
            "    - pixel\n"
            "    - visual\n"
            "  outlier_threshold: null\n"
            "models:\n"
            "  - name: resnet50\n"
            "    extractor:\n"
            "      type: onnx\n"
            "      model_path: ./resnet50.onnx\n"
            "      output_name: flatten0\n"
            "  - name: mobilenet\n"
            "    extractor:\n"
            "      type: onnx\n"
            "      model_path: ./mobilenet.onnx\n"
            "      output_name: flatten1\n"
        )

        config = load_config(config_file)
        assert config.models is not None
        assert len(config.models) == 2
        assert config.models[0].name == "resnet50"
        assert config.models[1].name == "mobilenet"


class TestP1SchemaClasses:
    """Test P1 schema classes (DatasetConfig, PreprocessorConfig, SelectionConfig, TaskConfig)."""

    def test_dataset_config_with_split(self):
        """DatasetConfig with a split name."""
        from dataeval_app.config.schemas import DatasetConfig

        dataset = DatasetConfig(
            name="cppe5",
            format="huggingface",
            path="./cppe5",
            split="train",
        )
        assert dataset.name == "cppe5"
        assert dataset.format == "huggingface"
        assert dataset.path == "./cppe5"
        assert dataset.split == "train"

    def test_dataset_config_with_none_split(self):
        """DatasetConfig with split=None (single-split dataset)."""
        from dataeval_app.config.schemas import DatasetConfig

        dataset = DatasetConfig(
            name="retail",
            format="coco",
            path="./retail",
            split=None,
        )
        assert dataset.split is None

    def test_dataset_config_missing_required_raises(self):
        """DatasetConfig requires name, format, path, split."""
        from dataeval_app.config.schemas import DatasetConfig

        with pytest.raises(ValidationError, match="name"):
            DatasetConfig(format="coco", path="./data", split="train")  # type: ignore[call-arg]

    def test_dataset_config_invalid_format_raises(self):
        """DatasetConfig rejects invalid format."""
        from dataeval_app.config.schemas import DatasetConfig

        with pytest.raises(ValidationError, match="format"):
            DatasetConfig(
                name="test",
                format="invalid",  # type: ignore[arg-type]
                path="./data",
                split="train",
            )

    def test_dataset_config_rejects_labels_dir_for_coco(self):
        """DatasetConfig rejects labels_dir for COCO format."""
        from dataeval_app.config.schemas import DatasetConfig

        with pytest.raises(ValidationError, match="labels_dir"):
            DatasetConfig(name="ds", format="coco", path="./data", labels_dir="labels")

    def test_dataset_config_rejects_annotations_file_for_yolo(self):
        """DatasetConfig rejects annotations_file for YOLO format."""
        from dataeval_app.config.schemas import DatasetConfig

        with pytest.raises(ValidationError, match="annotations_file"):
            DatasetConfig(name="ds", format="yolo", path="./data", annotations_file="ann.json")

    def test_dataset_config_rejects_split_for_image_folder(self):
        """DatasetConfig rejects split for image_folder format."""
        from dataeval_app.config.schemas import DatasetConfig

        with pytest.raises(ValidationError, match="split"):
            DatasetConfig(name="ds", format="image_folder", path="./data", split="train")

    def test_dataset_config_rejects_recursive_for_huggingface(self):
        """DatasetConfig rejects recursive for huggingface format."""
        from dataeval_app.config.schemas import DatasetConfig

        with pytest.raises(ValidationError, match="recursive"):
            DatasetConfig(name="ds", format="huggingface", path="./data", recursive=True)

    def test_preprocessor_config_valid(self):
        """PreprocessorConfig with valid steps."""
        from dataeval_app.config.schemas import PreprocessorConfig
        from dataeval_app.preprocessing import PreprocessingStep

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
        from dataeval_app.config.schemas import SelectionStep

        step = SelectionStep(type="Limit", params={"size": 10000})
        assert step.type == "Limit"
        assert step.params == {"size": 10000}

    def test_selection_step_default_params(self):
        """SelectionStep params defaults to empty dict."""
        from dataeval_app.config.schemas import SelectionStep

        step = SelectionStep(type="ClassBalance")
        assert step.type == "ClassBalance"
        assert step.params == {}

    def test_selection_config_valid(self):
        """SelectionConfig with named pipeline."""
        from dataeval_app.config.schemas import SelectionConfig, SelectionStep

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
        from dataeval_app.config.schemas import TaskConfig

        task = TaskConfig(
            name="data_cleaning",
            workflow="data-cleaning",
            datasets="cppe5",
            models="resnet50",
            preprocessors="resnet50_preprocessor",
            selections="training_subset",
            params={"outlier_method": "modzscore"},
            output_format="json",
            batch_size=64,
        )
        assert task.name == "data_cleaning"
        assert task.workflow == "data-cleaning"
        assert task.datasets == "cppe5"
        assert task.models == "resnet50"
        assert task.preprocessors == "resnet50_preprocessor"
        assert task.selections == "training_subset"
        assert task.params == {"outlier_method": "modzscore"}
        assert task.output_format == "json"
        assert task.batch_size == 64

    def test_task_config_with_model_no_batch_size_raises(self):
        """TaskConfig with model requiring batch_size but no batch_size raises ValidationError."""
        from dataeval_app.config.schemas import TaskConfig

        with pytest.raises(ValidationError, match="batch_size"):
            TaskConfig(
                name="test_task",
                workflow="data-cleaning",
                datasets="cppe5",
                models="bovw",
            )

    def test_task_config_with_datasets_list(self):
        """TaskConfig accepts datasets as a list."""
        from dataeval_app.config.schemas import TaskConfig

        task = TaskConfig(
            name="multi",
            workflow="data-cleaning",
            datasets=["ds_a", "ds_b"],
        )
        assert task.datasets == ["ds_a", "ds_b"]

    def test_task_config_with_model_mapping(self):
        """TaskConfig accepts model as a per-dataset mapping."""
        from dataeval_app.config.schemas import TaskConfig

        task = TaskConfig(
            name="mapped",
            workflow="data-cleaning",
            datasets=["ds_a", "ds_b"],
            models={"ds_a": "m1", "ds_b": "m2"},
            batch_size=64,
        )
        assert task.models == {"ds_a": "m1", "ds_b": "m2"}

    def test_task_config_minimal(self):
        """TaskConfig with only required fields."""
        from dataeval_app.config.schemas import TaskConfig

        task = TaskConfig(name="minimal", workflow="data-cleaning", datasets="test")
        assert task.name == "minimal"
        assert task.workflow == "data-cleaning"
        assert task.datasets == "test"
        assert task.models is None
        assert task.preprocessors is None
        assert task.selections is None
        assert task.params == {}
        assert task.output_format == "json"

    def test_task_config_invalid_output_format_raises(self):
        """TaskConfig rejects invalid output_format."""
        from dataeval_app.config.schemas import TaskConfig

        with pytest.raises(ValidationError, match="output_format"):
            TaskConfig(
                name="test",
                workflow="data-cleaning",
                datasets="test",
                output_format="invalid",  # type: ignore[arg-type]
            )

    def test_workflow_config_with_all_p1_schemas(self, tmp_path: Path):
        """WorkflowConfig loads all P1 schema sections."""
        # Create config with all P1 sections
        (tmp_path / "config.yaml").write_text(
            "datasets:\n"
            "  - name: cppe5\n"
            "    format: huggingface\n"
            "    path: ./cppe5\n"
            "    split: train\n"
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
            "    workflow: data-cleaning\n"
            "    datasets: cppe5\n"
            "    preprocessors: basic\n"
            "    selections: subset\n"
        )

        config = load_config_folder(tmp_path)

        # Verify datasets
        assert config.datasets is not None
        assert len(config.datasets) == 1
        assert config.datasets[0].name == "cppe5"
        assert config.datasets[0].split == "train"

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
        assert config.tasks[0].datasets == "cppe5"


class TestLoggingConfig:
    """Test LoggingConfig parsing via WorkflowConfig."""

    def test_workflow_config_with_logging(self, tmp_path: Path):
        config_file = tmp_path / "params.yaml"
        config_file.write_text("logging:\n  app_level: INFO\n  lib_level: DEBUG\n")

        config = load_config(config_file)
        assert config.logging is not None
        assert config.logging.app_level == "INFO"
        assert config.logging.lib_level == "DEBUG"

    def test_workflow_config_logging_defaults(self, tmp_path: Path):
        config_file = tmp_path / "params.yaml"
        config_file.write_text("logging: {}\n")

        config = load_config(config_file)
        assert config.logging is not None
        assert config.logging.app_level == "DEBUG"
        assert config.logging.lib_level == "WARNING"

    def test_workflow_config_logging_invalid_level(self, tmp_path: Path):
        config_file = tmp_path / "params.yaml"
        config_file.write_text("logging:\n  app_level: TRACE\n")

        with pytest.raises(ValidationError):
            load_config(config_file)
