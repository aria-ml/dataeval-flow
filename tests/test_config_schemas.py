"""Tests for config/workflow parameters, outputs, and base classes."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from dataeval_flow.config import (
    BoVWExtractorConfig,
    CocoDatasetConfig,
    DataCleaningWorkflowConfig,
    FlattenExtractorConfig,
    HuggingFaceDatasetConfig,
    ImageFolderDatasetConfig,
    OnnxExtractorConfig,
    PipelineConfig,
    SourceConfig,
    TorchExtractorConfig,
    UncertaintyExtractorConfig,
    YoloDatasetConfig,
    export_params_schema,
    load_config,
    load_config_folder,
)
from dataeval_flow.workflow.base import Reportable, WorkflowOutputsBase, WorkflowParametersBase, WorkflowReportBase
from dataeval_flow.workflows.cleaning import (
    DataCleaningOutputs,
    DataCleaningParameters,
    DataCleaningRawOutputs,
    DataCleaningReport,
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
            outlier_method="iqr", outlier_threshold=2.5, outlier_flags=["pixel", "visual"], mode="preparatory"
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
            DataCleaningParameters(outlier_method="modzscore", outlier_flags=["dimension"], outlier_threshold=-1.0)

    def test_empty_outlier_flags_rejected(self):
        """Empty outlier_flags list raises ValidationError."""
        with pytest.raises(ValidationError, match="outlier_flags"):
            DataCleaningParameters(outlier_method="modzscore", outlier_flags=[], outlier_threshold=None)

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
            outlier_method="modzscore", outlier_flags=["dimension"], duplicate_flags=["hash_basic", "hash_d4"]
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


class TestYAMLValidationEdgeCases:
    """Test YAML validation edge cases for user error scenarios."""

    def test_unknown_field_name_raises(self):
        """Unknown field name (typo) shows as missing required field."""
        # Using wrong field name simulates a typo - Pydantic ignores unknown fields
        # but requires all declared fields, so missing 'outlier_method' raises
        with pytest.raises(ValidationError, match="outlier_method"):
            DataCleaningParameters(
                outler_method="iqr",  # type: ignore[call-arg]  # wrong field name
                outlier_flags=["dimension", "pixel", "visual"],
            )

    def test_wrong_type_string_instead_of_number(self):
        """outlier_threshold as string raises ValidationError."""
        with pytest.raises(ValidationError, match="outlier_threshold"):
            DataCleaningParameters(
                outlier_method="iqr",
                outlier_flags=["dimension", "pixel", "visual"],
                outlier_threshold="high",  # type: ignore[arg-type]  # string not number
            )

    def test_invalid_outlier_flag_value(self):
        """Invalid outlier flag value raises ValidationError."""
        with pytest.raises(ValidationError, match="outlier_flags"):
            DataCleaningParameters(
                outlier_method="iqr",
                outlier_flags=["invalid_flag"],  # type: ignore[list-item]
            )


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
        assert "tasks" in schema["properties"]


class TestUnifiedConfig:
    """Test unified config loading."""

    def test_load_config_unified(self, tmp_path: Path):
        """load_config loads task-centric config."""
        config_file = tmp_path / "params.yaml"
        config_file.write_text(
            "datasets:\n"
            "  - name: train\n"
            "    format: image_folder\n"
            "    path: ./data/train\n"
            "sources:\n"
            "  - name: train_src\n"
            "    dataset: train\n"
            "workflows:\n"
            "  - name: iqr_clean\n"
            "    type: data-cleaning\n"
            "    outlier_method: iqr\n"
            "    outlier_flags:\n"
            "      - dimension\n"
            "      - pixel\n"
            "      - visual\n"
            "tasks:\n"
            "  - name: clean_train\n"
            "    workflow: iqr_clean\n"
            "    sources: train_src\n"
        )

        config = load_config(config_file)
        assert isinstance(config, PipelineConfig)
        assert config.tasks is not None
        assert len(config.tasks) == 1
        assert config.tasks[0].workflow == "iqr_clean"
        assert config.workflows is not None
        wf = config.workflows[0]
        assert isinstance(wf, DataCleaningWorkflowConfig)
        assert wf.outlier_method == "iqr"

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
        item = Reportable(report_type="table", title="Test", data={"key": "value"})
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
        finding = Reportable(report_type="key_value", title="Finding", data={"count": 10}, description="Test finding")
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
        (tmp_path / "00-extractors.yaml").write_text(
            "extractors:\n"
            "  - name: resnet50\n"
            "    model: onnx\n"
            "    model_path: ./resnet50.onnx\n"
            "    output_name: flatten0\n"
        )
        (tmp_path / "01-datasets.yaml").write_text(
            "datasets:\n  - name: train\n    format: image_folder\n    path: ./data/train\n"
        )

        config = load_config_folder(tmp_path)
        assert config.extractors is not None
        assert len(config.extractors) == 1
        assert config.extractors[0].name == "resnet50"
        assert config.datasets is not None
        assert len(config.datasets) == 1

    def test_load_from_folder_not_directory(self, tmp_path: Path):
        """Raises ValueError if path is not a directory."""
        file_path = tmp_path / "file.yaml"
        file_path.write_text("key: value\n")

        with pytest.raises(ValueError, match="not a directory"):
            load_config_folder(file_path)


class TestSourceConfig:
    """Test SourceConfig schema."""

    def test_source_config_valid(self):
        """SourceConfig with dataset and selection."""
        src = SourceConfig(name="train_src", dataset="train_ds", selection="first_5k")
        assert src.name == "train_src"
        assert src.dataset == "train_ds"
        assert src.selection == "first_5k"

    def test_source_config_no_selection(self):
        """SourceConfig without selection defaults to None."""
        src = SourceConfig(name="train_src", dataset="train_ds")
        assert src.selection is None

    def test_source_config_missing_dataset_raises(self):
        """SourceConfig requires dataset field."""
        with pytest.raises(ValidationError, match="dataset"):
            SourceConfig(name="src")  # type: ignore[call-arg]


class TestExtractorConfig:
    """Test ExtractorConfig schema."""

    def test_extractor_config_valid_onnx(self):
        """ExtractorConfig with onnx model and preprocessor."""
        ext = OnnxExtractorConfig(
            name="resnet_ext",
            model_path="./resnet50.onnx",
            output_name="flatten0",
            preprocessor="resnet_preprocess",
            batch_size=64,
        )
        assert ext.name == "resnet_ext"
        assert ext.model == "onnx"
        assert ext.model_path == "./resnet50.onnx"
        assert ext.output_name == "flatten0"
        assert ext.preprocessor == "resnet_preprocess"
        assert ext.batch_size == 64

    def test_extractor_config_valid_flatten(self):
        """ExtractorConfig with flatten model type."""
        ext = FlattenExtractorConfig(name="flat_ext")
        assert ext.model == "flatten"
        assert ext.preprocessor is None
        assert ext.batch_size is None

    def test_extractor_config_valid_bovw(self):
        """ExtractorConfig with bovw model type and vocab_size."""
        ext = BoVWExtractorConfig(name="bovw_ext", vocab_size=1024)
        assert ext.model == "bovw"
        assert ext.vocab_size == 1024

    def test_extractor_config_bovw_vocab_size_out_of_range_raises(self):
        """ExtractorConfig rejects bovw vocab_size outside 256-4096."""
        with pytest.raises(ValidationError, match="vocab_size"):
            BoVWExtractorConfig(name="bovw_ext", vocab_size=1)
        with pytest.raises(ValidationError, match="vocab_size"):
            BoVWExtractorConfig(name="bovw_ext", vocab_size=10000)

    def test_extractor_config_valid_torch(self):
        """ExtractorConfig with torch model type."""
        ext = TorchExtractorConfig(name="torch_ext", model_path="./resnet.pt", layer_name="layer4", device="cpu")
        assert ext.model == "torch"
        assert ext.model_path == "./resnet.pt"
        assert ext.layer_name == "layer4"
        assert ext.device == "cpu"

    def test_extractor_config_valid_uncertainty(self):
        """ExtractorConfig with uncertainty model type."""
        ext = UncertaintyExtractorConfig(name="unc_ext", model_path="./classifier.pt", preds_type="logits")
        assert ext.model == "uncertainty"
        assert ext.preds_type == "logits"

    def test_extractor_config_minimal(self):
        """ExtractorConfig with only required fields."""
        ext = FlattenExtractorConfig(name="ext")
        assert ext.preprocessor is None
        assert ext.batch_size is None
        assert ext.model == "flatten"

    def test_extractor_config_onnx_flatten_default(self):
        """ExtractorConfig onnx flatten defaults to True."""
        ext = OnnxExtractorConfig(name="ext", model_path="./m.onnx")
        assert ext.flatten is True

    def test_extractor_config_torch_use_output_default(self):
        """ExtractorConfig torch use_output defaults to True."""
        ext = TorchExtractorConfig(name="ext", model_path="./m.pt")
        assert ext.use_output is True

    def test_extractor_discriminated_union_from_yaml(self, tmp_path: Path):
        """YAML with extractors section loads correct extractor configs."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "extractors:\n"
            "  - name: onnx_ext\n"
            "    model: onnx\n"
            "    model_path: ./model.onnx\n"
            "    batch_size: 64\n"
            "  - name: bovw_ext\n"
            "    model: bovw\n"
            "    vocab_size: 512\n"
            "  - name: flat_ext\n"
            "    model: flatten\n"
        )

        config = load_config(config_file)
        assert config.extractors is not None
        assert len(config.extractors) == 3

        assert config.extractors[0].model == "onnx"
        assert config.extractors[0].model_path == "./model.onnx"
        assert config.extractors[0].batch_size == 64
        assert config.extractors[1].model == "bovw"
        assert config.extractors[1].vocab_size == 512
        assert config.extractors[2].model == "flatten"

    def test_pipeline_config_without_extractors(self, tmp_path: Path):
        """PipelineConfig works without extractors section."""
        config_file = tmp_path / "params.yaml"
        config_file.write_text("datasets:\n  - name: train\n    format: image_folder\n    path: ./data/train\n")

        config = load_config(config_file)
        assert config.extractors is None

    def test_pipeline_config_with_extractors(self, tmp_path: Path):
        """PipelineConfig loads extractors list."""
        config_file = tmp_path / "params.yaml"
        config_file.write_text(
            "extractors:\n"
            "  - name: resnet50\n"
            "    model: onnx\n"
            "    model_path: ./resnet50.onnx\n"
            "    output_name: flatten0\n"
            "  - name: mobilenet\n"
            "    model: onnx\n"
            "    model_path: ./mobilenet.onnx\n"
            "    output_name: flatten1\n"
        )

        config = load_config(config_file)
        assert config.extractors is not None
        assert len(config.extractors) == 2
        assert config.extractors[0].name == "resnet50"
        assert config.extractors[1].name == "mobilenet"


class TestP1SchemaClasses:
    """Test P1 schema classes (DatasetConfig, PreprocessorConfig, SelectionConfig, TaskConfig)."""

    def test_dataset_config_with_split(self):
        """DatasetConfig with a split name."""

        dataset = HuggingFaceDatasetConfig(name="cppe5", path="./cppe5", split="train")
        assert dataset.name == "cppe5"
        assert dataset.format == "huggingface"
        assert dataset.path == "./cppe5"
        assert dataset.split == "train"

    def test_dataset_config_with_none_split(self):
        """HuggingFaceDatasetConfig with split=None (single-split dataset)."""
        dataset = HuggingFaceDatasetConfig(name="retail", path="./retail", split=None)
        assert dataset.split is None

    def test_dataset_config_missing_required_raises(self):
        """DatasetConfig requires name and path."""
        with pytest.raises(ValidationError, match="name"):
            CocoDatasetConfig(path="./data")  # type: ignore[call-arg]

    def test_dataset_config_rejects_labels_dir_for_coco(self):
        """CocoDatasetConfig has no labels_dir field."""
        with pytest.raises(ValidationError):
            CocoDatasetConfig(name="ds", path="./data", labels_dir="labels")  # type: ignore[call-arg]

    def test_dataset_config_rejects_annotations_file_for_yolo(self):
        """YoloDatasetConfig has no annotations_file field."""
        with pytest.raises(ValidationError):
            YoloDatasetConfig(name="ds", path="./data", annotations_file="ann.json")  # type: ignore[call-arg]

    def test_dataset_config_rejects_split_for_image_folder(self):
        """ImageFolderDatasetConfig has no split field."""
        with pytest.raises(ValidationError):
            ImageFolderDatasetConfig(name="ds", path="./data", split="train")  # type: ignore[call-arg]

    def test_dataset_config_rejects_recursive_for_huggingface(self):
        """HuggingFaceDatasetConfig has no recursive field."""
        with pytest.raises(ValidationError):
            HuggingFaceDatasetConfig(name="ds", path="./data", recursive=True)  # type: ignore[call-arg]

    def test_preprocessor_config_valid(self):
        """PreprocessorConfig with valid steps."""
        from dataeval_flow.config import PreprocessorConfig
        from dataeval_flow.preprocessing import PreprocessingStep

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
        from dataeval_flow.config import SelectionStep

        step = SelectionStep(type="Limit", params={"size": 10000})
        assert step.type == "Limit"
        assert step.params == {"size": 10000}

    def test_selection_step_default_params(self):
        """SelectionStep params defaults to empty dict."""
        from dataeval_flow.config import SelectionStep

        step = SelectionStep(type="ClassBalance")
        assert step.type == "ClassBalance"
        assert step.params == {}

    def test_selection_config_valid(self):
        """SelectionConfig with named pipeline."""
        from dataeval_flow.config import SelectionConfig, SelectionStep

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

    def test_selection_step_indices_range_shorthand(self):
        """SelectionStep expands indices range dict into a list."""
        from dataeval_flow.config import SelectionStep

        step = SelectionStep(type="Indices", params={"indices": {"start": 500, "stop": 505}})
        assert step.params["indices"] == [500, 501, 502, 503, 504]

    def test_selection_step_indices_range_with_step(self):
        """SelectionStep expands indices range dict with custom step."""
        from dataeval_flow.config import SelectionStep

        step = SelectionStep(type="Indices", params={"indices": {"start": 0, "stop": 10, "step": 3}})
        assert step.params["indices"] == [0, 3, 6, 9]

    def test_selection_step_indices_list_unchanged(self):
        """SelectionStep leaves an explicit indices list untouched."""
        from dataeval_flow.config import SelectionStep

        step = SelectionStep(type="Indices", params={"indices": [1, 5, 42]})
        assert step.params["indices"] == [1, 5, 42]

    def test_selection_step_indices_range_rejects_invalid_keys(self):
        """SelectionStep rejects unknown keys like 'end' in the range shorthand."""
        from dataeval_flow.config import SelectionStep

        with pytest.raises(ValidationError, match="Invalid keys"):
            SelectionStep(type="Indices", params={"indices": {"start": 500, "end": 550}})

    def test_selection_step_indices_range_requires_start_and_stop(self):
        """SelectionStep requires both 'start' and 'stop' in range shorthand."""
        from dataeval_flow.config import SelectionStep

        with pytest.raises(ValidationError, match="requires both"):
            SelectionStep(type="Indices", params={"indices": {"start": 500}})

    def test_selection_step_non_indices_params_unchanged(self):
        """SelectionStep does not touch params without an indices key."""
        from dataeval_flow.config import SelectionStep

        step = SelectionStep(type="Limit", params={"size": 100})
        assert step.params == {"size": 100}

    def test_selection_step_indices_range_too_large(self):
        """SelectionStep rejects range shorthand exceeding 1M elements (line 83)."""
        from dataeval_flow.config import SelectionStep

        with pytest.raises(ValidationError, match="max 1,000,000"):
            SelectionStep(type="Indices", params={"indices": {"start": 0, "stop": 2_000_000}})

    def test_selection_step_rejects_non_dict_input(self):
        """SelectionStep.model_validate rejects non-dict, non-instance input."""
        from dataeval_flow.config import SelectionStep

        with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
            SelectionStep.model_validate("not a dict")

    def test_task_config_valid(self):
        """TaskConfig with sources and extractor references."""
        from dataeval_flow.config import TaskConfig

        task = TaskConfig(
            name="data_cleaning",
            workflow="clean_zscore_stats",
            sources="train_src",
            extractor="resnet_ext",
        )
        assert task.name == "data_cleaning"
        assert task.workflow == "clean_zscore_stats"
        assert task.sources == "train_src"
        assert task.extractor == "resnet_ext"

    def test_task_config_with_sources_list(self):
        """TaskConfig accepts sources as a list."""
        from dataeval_flow.config import TaskConfig

        task = TaskConfig(name="multi", workflow="data-cleaning", sources=["src_a", "src_b"])
        assert task.sources == ["src_a", "src_b"]

    def test_task_config_minimal(self):
        """TaskConfig with only required fields."""
        from dataeval_flow.config import TaskConfig

        task = TaskConfig(name="minimal", workflow="data-cleaning", sources="test_src")
        assert task.name == "minimal"
        assert task.workflow == "data-cleaning"
        assert task.sources == "test_src"
        assert task.extractor is None

    def test_workflow_config_with_all_p1_schemas(self, tmp_path: Path):
        """PipelineConfig loads all P1 schema sections."""
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
            "sources:\n"
            "  - name: cppe5_src\n"
            "    dataset: cppe5\n"
            "    selection: subset\n"
            "extractors:\n"
            "  - name: resnet_ext\n"
            "    model: onnx\n"
            "    model_path: ./resnet50.onnx\n"
            "    output_name: flatten0\n"
            "    batch_size: 64\n"
            "tasks:\n"
            "  - name: clean\n"
            "    workflow: data-cleaning\n"
            "    sources: cppe5_src\n"
            "    extractor: resnet_ext\n"
        )

        config = load_config_folder(tmp_path)

        # Verify datasets
        assert config.datasets is not None
        assert len(config.datasets) == 1
        assert isinstance(config.datasets[0], HuggingFaceDatasetConfig)
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

        # Verify sources
        assert config.sources is not None
        assert len(config.sources) == 1
        assert config.sources[0].name == "cppe5_src"
        assert config.sources[0].dataset == "cppe5"
        assert config.sources[0].selection == "subset"

        # Verify extractors
        assert config.extractors is not None
        assert len(config.extractors) == 1
        assert config.extractors[0].name == "resnet_ext"
        assert config.extractors[0].model == "onnx"
        assert config.extractors[0].model_path == "./resnet50.onnx"
        assert config.extractors[0].batch_size == 64

        # Verify tasks
        assert config.tasks is not None
        assert len(config.tasks) == 1
        assert config.tasks[0].name == "clean"
        assert config.tasks[0].sources == "cppe5_src"
        assert config.tasks[0].extractor == "resnet_ext"


class TestDriftMonitoringTaskConfig:
    """Test DriftMonitoringTaskConfig schema."""

    def test_single_source_raises(self):
        """DriftMonitoringTaskConfig requires at least 2 sources."""
        from dataeval_flow.config import DriftMonitoringTaskConfig

        with pytest.raises(ValidationError, match="at least 2 sources"):
            DriftMonitoringTaskConfig(name="test", workflow="drift-instance", sources="single_src")

    def test_single_source_in_list_raises(self):
        """DriftMonitoringTaskConfig rejects a list with only one source."""
        from dataeval_flow.config import DriftMonitoringTaskConfig

        with pytest.raises(ValidationError, match="at least 2 sources"):
            DriftMonitoringTaskConfig(name="test", workflow="drift-instance", sources=["only_one"])

    def test_valid_drift_task_config(self):
        """DriftMonitoringTaskConfig accepts valid config with 2 sources."""
        from dataeval_flow.config import DriftMonitoringTaskConfig

        task = DriftMonitoringTaskConfig(name="drift", workflow="drift-instance", sources=["ref_src", "test_src"])
        assert task.name == "drift"
        assert len(task.sources) == 2


class TestLoggingConfig:
    """Test LoggingConfig parsing via PipelineConfig."""

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


class TestWorkflowConfig:
    """Test WorkflowConfig schema (workflow instances)."""

    def test_cleaning_workflow_config_basic(self):
        """DataCleaningWorkflowConfig stores name, type, and flat params."""
        wc = DataCleaningWorkflowConfig(
            name="aggressive_clean", outlier_method="zscore", outlier_flags=["dimension", "pixel"]
        )
        assert wc.name == "aggressive_clean"
        assert wc.type == "data-cleaning"
        assert wc.outlier_method == "zscore"
        assert wc.outlier_flags == ["dimension", "pixel"]

    def test_cleaning_workflow_config_requires_fields(self):
        """DataCleaningWorkflowConfig requires outlier_method and outlier_flags."""
        with pytest.raises(ValidationError):
            DataCleaningWorkflowConfig(name="empty")  # type: ignore[call-arg]

    def test_drift_workflow_config_basic(self):
        """DriftMonitoringWorkflowConfig stores name, type, and flat params."""
        from dataeval_flow.config import DriftMonitoringWorkflowConfig
        from dataeval_flow.workflows.drift.params import DriftDetectorKNeighbors

        wc = DriftMonitoringWorkflowConfig(name="knn_drift", detectors=[DriftDetectorKNeighbors(k=10)])
        assert wc.name == "knn_drift"
        assert wc.type == "drift-monitoring"
        assert len(wc.detectors) == 1

    def test_workflow_config_in_pipeline_config(self, tmp_path: Path):
        """PipelineConfig loads named workflows list from flat YAML."""
        config_file = tmp_path / "params.yaml"
        config_file.write_text(
            "workflows:\n"
            "  - name: standard_clean\n"
            "    type: data-cleaning\n"
            "    outlier_method: adaptive\n"
            "    outlier_flags: [dimension, pixel, visual]\n"
            "  - name: strict_clean\n"
            "    type: data-cleaning\n"
            "    outlier_method: zscore\n"
            "    outlier_flags: [dimension]\n"
        )

        config = load_config(config_file)
        assert config.workflows is not None
        assert len(config.workflows) == 2
        assert config.workflows[0].name == "standard_clean"
        assert config.workflows[1].name == "strict_clean"
        wf = config.workflows[0]
        assert isinstance(wf, DataCleaningWorkflowConfig)
        assert wf.outlier_method == "adaptive"

    def test_full_config_with_workflows(self, tmp_path: Path):
        """Full config with workflows referenced by tasks."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "datasets:\n"
            "  - name: train\n"
            "    format: image_folder\n"
            "    path: ./data/train\n"
            "sources:\n"
            "  - name: train_src\n"
            "    dataset: train\n"
            "workflows:\n"
            "  - name: standard_clean\n"
            "    type: data-cleaning\n"
            "    outlier_method: adaptive\n"
            "    outlier_flags: [dimension, pixel, visual]\n"
            "tasks:\n"
            "  - name: clean_train\n"
            "    workflow: standard_clean\n"
            "    sources: train_src\n"
        )

        config = load_config(config_file)
        assert config.workflows is not None
        assert config.tasks is not None
        assert config.tasks[0].workflow == "standard_clean"

    def test_invalid_params_rejected(self):
        """DataCleaningWorkflowConfig rejects invalid field values."""
        with pytest.raises(ValidationError):
            DataCleaningWorkflowConfig(
                name="bad",
                outlier_method="not_a_real_method",  # type: ignore[call-arg]
                outlier_flags=["dimension"],
            )


class TestResolveWorkflow:
    """Test orchestrator _resolve_workflow function."""

    def test_resolve_workflow_by_name(self):
        """Workflow is resolved from config.workflows by name."""
        from dataeval_flow.workflow.orchestrator import _resolve_workflow

        config = PipelineConfig(
            workflows=[
                DataCleaningWorkflowConfig(
                    name="standard_clean", outlier_method="adaptive", outlier_flags=["dimension", "pixel"]
                ),
            ]
        )
        result = _resolve_workflow("standard_clean", config)
        assert result.name == "standard_clean"
        assert result.type == "data-cleaning"
        assert isinstance(result, DataCleaningWorkflowConfig)
        assert result.outlier_method == "adaptive"

    def test_resolve_workflow_not_found(self):
        """Unknown workflow name raises ValueError."""
        from dataeval_flow.workflow.orchestrator import _resolve_workflow

        config = PipelineConfig(
            workflows=[
                DataCleaningWorkflowConfig(name="existing", outlier_method="zscore", outlier_flags=["dimension"])
            ]
        )
        with pytest.raises(ValueError, match="Unknown workflow: 'nonexistent'"):
            _resolve_workflow("nonexistent", config)

    def test_resolve_workflow_no_workflows_section(self):
        """Workflow reference with no workflows section raises ValueError."""
        from dataeval_flow.workflow.orchestrator import _resolve_workflow

        config = PipelineConfig()
        with pytest.raises(ValueError, match="No workflow configs defined"):
            _resolve_workflow("standard_clean", config)


# ---------------------------------------------------------------------------
# PipelineConfig — duplicate name validation
# ---------------------------------------------------------------------------


class TestPipelineConfigDuplicateNames:
    """Test PipelineConfig rejects duplicate names within sections."""

    def test_duplicate_dataset_name_raises(self):
        with pytest.raises(ValidationError, match="Duplicate name 'ds' in datasets"):
            PipelineConfig(
                datasets=[
                    ImageFolderDatasetConfig(name="ds", path="./a"),
                    ImageFolderDatasetConfig(name="ds", path="./b"),
                ]
            )

    def test_duplicate_source_name_raises(self):
        with pytest.raises(ValidationError, match="Duplicate name 'src' in sources"):
            PipelineConfig(
                sources=[
                    SourceConfig(name="src", dataset="ds_a"),
                    SourceConfig(name="src", dataset="ds_b"),
                ]
            )

    def test_duplicate_extractor_name_raises(self):
        with pytest.raises(ValidationError, match="Duplicate name 'ext' in extractors"):
            PipelineConfig(
                extractors=[
                    FlattenExtractorConfig(name="ext"),
                    FlattenExtractorConfig(name="ext"),
                ]
            )

    def test_duplicate_workflow_name_raises(self):
        with pytest.raises(ValidationError, match="Duplicate name 'wf' in workflows"):
            PipelineConfig(
                workflows=[
                    DataCleaningWorkflowConfig(name="wf", outlier_method="zscore", outlier_flags=["dimension"]),
                    DataCleaningWorkflowConfig(name="wf", outlier_method="iqr", outlier_flags=["pixel"]),
                ]
            )

    def test_unique_names_pass(self):
        config = PipelineConfig(
            sources=[
                SourceConfig(name="src_a", dataset="ds_a"),
                SourceConfig(name="src_b", dataset="ds_b"),
            ]
        )
        assert config.sources is not None
        assert len(config.sources) == 2


# ---------------------------------------------------------------------------
# ExtractorConfig — irrelevant field validation
# ---------------------------------------------------------------------------


class TestExtractorConfigFieldValidation:
    """Test typed extractor configs reject fields from other model types."""

    def test_bovw_rejects_output_name(self):
        with pytest.raises(ValidationError):
            BoVWExtractorConfig(name="ext", vocab_size=512, output_name="layer4")  # type: ignore[call-arg]

    def test_bovw_rejects_model_path(self):
        with pytest.raises(ValidationError):
            BoVWExtractorConfig(name="ext", vocab_size=512, model_path="/m.onnx")  # type: ignore[call-arg]

    def test_flatten_rejects_vocab_size(self):
        with pytest.raises(ValidationError):
            FlattenExtractorConfig(name="ext", vocab_size=1024)  # type: ignore[call-arg]

    def test_flatten_rejects_model_path(self):
        with pytest.raises(ValidationError):
            FlattenExtractorConfig(name="ext", model_path="/m.onnx")  # type: ignore[call-arg]

    def test_onnx_rejects_vocab_size(self):
        with pytest.raises(ValidationError):
            OnnxExtractorConfig(name="ext", model_path="/m.onnx", vocab_size=512)  # type: ignore[call-arg]

    def test_onnx_rejects_preds_type(self):
        with pytest.raises(ValidationError):
            OnnxExtractorConfig(name="ext", model_path="/m.onnx", preds_type="logits")  # type: ignore[call-arg]

    def test_onnx_allows_relevant_fields(self):
        ext = OnnxExtractorConfig(name="ext", model_path="/m.onnx", output_name="out", flatten=False)
        assert ext.model_path == "/m.onnx"
        assert ext.output_name == "out"
        assert ext.flatten is False

    def test_uncertainty_rejects_vocab_size(self):
        with pytest.raises(ValidationError):
            UncertaintyExtractorConfig(name="ext", model_path="/m.pt", preds_type="logits", vocab_size=512)  # type: ignore[call-arg]

    def test_torch_allows_relevant_fields(self):
        ext = TorchExtractorConfig(name="ext", model_path="/m.pt", layer_name="layer4", use_output=False, device="cpu")
        assert ext.layer_name == "layer4"
        assert ext.use_output is False
