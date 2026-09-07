"""Tests for workflow orchestrator — _run_single_task, _resolve_by_name."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dataeval_flow.config import (
    CocoDatasetConfig,
    DataCleaningWorkflowConfig,
    DataCoverageWorkflowConfig,
    HuggingFaceDatasetConfig,
    ImageFolderDatasetConfig,
    OnnxExtractorConfig,
    SourceConfig,
    TaskConfig,
    YoloDatasetConfig,
)
from dataeval_flow.config.schemas import ResultMetadata
from dataeval_flow.workflow.orchestrator import (
    _relativize_paths,
    _resolve_by_name,
    _resolve_extractor_paths,
    _run_single_task,
    run_task,
    run_tasks,
    select_tasks,
)

pytestmark = pytest.mark.required

# Shared workflow instance used across tests
_CLEAN_INSTANCE = DataCleaningWorkflowConfig(
    name="clean", outlier_method="zscore", outlier_flags=["dimension", "pixel"]
)

# ---------------------------------------------------------------------------
# _resolve_by_name
# ---------------------------------------------------------------------------


class _Named:
    """Simple named object for testing _resolve_by_name."""

    def __init__(self, name: str) -> None:
        self.name = name


class TestResolveByName:
    def test_finds_existing_item(self):
        items = [_Named("a"), _Named("b"), _Named("c")]
        result = _resolve_by_name(items, "b", "test")
        assert result.name == "b"

    def test_raises_on_missing_name(self):
        items = [_Named("a"), _Named("b")]
        with pytest.raises(ValueError, match="Unknown test: 'z'"):
            _resolve_by_name(items, "z", "test")

    def test_raises_on_none_list(self):
        with pytest.raises(ValueError, match="No dataset configs defined"):
            _resolve_by_name(None, "x", "dataset")

    def test_error_shows_available_names(self):
        items = [_Named("foo"), _Named("bar")]
        with pytest.raises(ValueError, match="Available: \\['foo', 'bar'\\]"):
            _resolve_by_name(items, "baz", "thing")


# ---------------------------------------------------------------------------
# _run_single_task (integration with mocks)
# ---------------------------------------------------------------------------


class TestRunTask:
    """Tests for _run_single_task().

    Note: _run_single_task() uses lazy imports inside the function body, so we
    patch at the source module level (e.g. dataeval_flow.dataset.load_dataset)
    rather than on orchestrator.
    """

    def _build_config_and_task(self) -> tuple[MagicMock, TaskConfig]:
        """Build minimal config + task for _run_single_task testing."""

        ds_config = HuggingFaceDatasetConfig(name="test_ds", path="./test", split="train", task="image_classification")
        source = SourceConfig(name="src_test", dataset="test_ds")
        task_config = TaskConfig(name="test_task", workflow="clean", sources="src_test")

        config = MagicMock()
        config.datasets = [ds_config]
        config.sources = [source]
        config.extractors = None
        config.preprocessors = None
        config.selections = None
        config.workflows = [_CLEAN_INSTANCE]

        return config, task_config

    def _mock_workflow(self, params_schema: Any = None) -> MagicMock:
        """Build a mock workflow that returns a mock result."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_workflow = MagicMock()
        mock_workflow.params_schema = params_schema
        mock_workflow.execute.return_value = mock_result
        return mock_workflow

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_task_basic(self, mock_load_ds: MagicMock):
        """_run_single_task resolves config, runs workflow, returns result."""
        config, task = self._build_config_and_task()
        mock_load_ds.return_value = MagicMock()

        mock_wf = self._mock_workflow()

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            result = _run_single_task(task, config)

        assert result.success
        mock_load_ds.assert_called_once()
        mock_wf.execute.assert_called_once()

    @patch("dataeval_flow.dataset.load_dataset")
    @patch("dataeval_flow.preprocessing.build_preprocessing")
    def test_run_task_with_preprocessor(self, mock_build_pre: MagicMock, mock_load_ds: MagicMock):
        """_run_single_task resolves preprocessor via extractor config."""
        from dataeval_flow.config import PreprocessorConfig
        from dataeval_flow.preprocessing import PreprocessingStep

        config, _ = self._build_config_and_task()
        config.preprocessors = [
            PreprocessorConfig(name="basic", steps=[PreprocessingStep(step="ToTensor")]),
        ]
        config.extractors = [
            OnnxExtractorConfig(name="ext1", model_path="./model.onnx", preprocessor="basic", batch_size=64),
        ]

        task = TaskConfig(name="test_task", workflow="clean", sources="src_test", extractor="ext1")

        mock_load_ds.return_value = MagicMock()
        mock_build_pre.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            result = _run_single_task(task, config)

        assert result.success
        mock_build_pre.assert_called_once()

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_task_with_extractor(self, mock_load_ds: MagicMock):
        """_run_single_task resolves extractor when task references one."""
        config, _ = self._build_config_and_task()
        config.extractors = [
            OnnxExtractorConfig(name="ext1", model_path="./model.onnx", output_name="layer4", batch_size=64),
        ]

        task = TaskConfig(name="test_task", workflow="clean", sources="src_test", extractor="ext1")

        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            result = _run_single_task(task, config)

        assert result.success
        # Verify extractor config was passed into DatasetContext
        context = mock_wf.execute.call_args[0][0]
        dc = context.dataset_contexts["src_test"]
        assert dc.extractor is not None
        # _resolve_extractor_paths joins relative path against data_dir (default ".")
        assert dc.extractor.model_path == "model.onnx"
        assert dc.extractor.output_name == "layer4"

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_task_with_view(self, mock_load_ds: MagicMock):
        """_run_single_task resolves view config from source."""
        from dataeval_flow.config import ViewConfig, ViewOperation

        ds_config = HuggingFaceDatasetConfig(name="test_ds", path="./test", split="train", task="image_classification")
        source = SourceConfig(name="src_test", dataset="test_ds", view="sub")

        task = TaskConfig(name="test_task", workflow="clean", sources="src_test")

        config = MagicMock()
        config.datasets = [ds_config]
        config.sources = [source]
        config.extractors = None
        config.preprocessors = None
        config.views = [
            ViewConfig(name="sub", operations=[ViewOperation(type="Limit", params={"size": 100})]),
        ]
        config.workflows = [_CLEAN_INSTANCE]

        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            result = _run_single_task(task, config)

        assert result.success
        # Verify selection steps were passed into DatasetContext
        context = mock_wf.execute.call_args[0][0]
        dc = context.dataset_contexts["src_test"]
        assert len(dc.view_operations) == 1
        assert dc.view_operations[0].type == "Limit"
        assert dc.view_operations[0].params == {"size": 100}

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_task_validates_params(self, mock_load_ds: MagicMock):
        """_run_single_task validates workflow instance params against workflow.params_schema."""
        from dataeval_flow.workflows.cleaning.params import DataCleaningParameters

        config, task = self._build_config_and_task()
        mock_load_ds.return_value = MagicMock()

        mock_wf = self._mock_workflow(params_schema=DataCleaningParameters)

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            result = _run_single_task(task, config)

        assert result.success
        # Verify the instance params were validated against the schema
        mock_wf.execute.assert_called_once()
        call_args = mock_wf.execute.call_args
        params = call_args[0][1]
        assert isinstance(params, DataCleaningParameters)
        assert params.outlier_method == "zscore"

    def test_run_task_raises_on_missing_source(self):
        """_run_single_task raises ValueError when source not found."""
        config = MagicMock()
        config.sources = []
        config.extractors = None
        config.workflows = [_CLEAN_INSTANCE]
        task = TaskConfig(name="t", workflow="clean", sources="nonexistent")

        with pytest.raises(ValueError, match="Unknown source"):
            _run_single_task(task, config)

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_task_raises_on_missing_extractor(self, mock_load_ds: MagicMock):
        """_run_single_task raises ValueError when extractor not found."""
        config, _ = self._build_config_and_task()
        config.extractors = []  # Empty list — no extractors defined

        task = TaskConfig(name="t", workflow="clean", sources="src_test", extractor="nonexistent")
        mock_load_ds.return_value = MagicMock()

        with pytest.raises(ValueError, match="Unknown extractor"):
            _run_single_task(task, config)

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_task_passes_format_and_image_folder_params(self, mock_load_ds: MagicMock):
        """_run_single_task passes dataset_format, recursive, and infer_labels to load_dataset."""

        ds_config = ImageFolderDatasetConfig(name="photos", path="data/photos", recursive=True, infer_labels=True)
        source = SourceConfig(name="src_photos", dataset="photos")
        task = TaskConfig(name="t", workflow="clean", sources="src_photos")

        config = MagicMock()
        config.datasets = [ds_config]
        config.sources = [source]
        config.extractors = None
        config.preprocessors = None
        config.selections = None
        config.workflows = [_CLEAN_INSTANCE]

        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            _run_single_task(task, config)

        mock_load_ds.assert_called_once_with(
            Path("data/photos"),
            dataset_format="image_folder",
            recursive=True,
            infer_labels=True,
        )

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_task_passes_coco_params(self, mock_load_ds: MagicMock):
        """_run_single_task passes COCO-specific config fields to load_dataset."""

        ds_config = CocoDatasetConfig(
            name="coco_ds",
            path="data/coco",
            annotations_file="instances.json",
            images_dir="train2017",
        )
        source = SourceConfig(name="src_coco", dataset="coco_ds")
        task = TaskConfig(name="t", workflow="clean", sources="src_coco")

        config = MagicMock()
        config.datasets = [ds_config]
        config.sources = [source]
        config.extractors = None
        config.preprocessors = None
        config.selections = None
        config.workflows = [_CLEAN_INSTANCE]

        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            _run_single_task(task, config)

        mock_load_ds.assert_called_once_with(
            Path("data/coco"),
            dataset_format="coco",
            annotations_file="instances.json",
            images_dir="train2017",
        )

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_task_passes_yolo_params(self, mock_load_ds: MagicMock):
        """_run_single_task passes YOLO-specific config fields to load_dataset."""

        ds_config = YoloDatasetConfig(name="yolo_ds", path="data/yolo", split="val", ann_dir="annotations")
        source = SourceConfig(name="src_yolo", dataset="yolo_ds")
        task = TaskConfig(name="t", workflow="clean", sources="src_yolo")

        config = MagicMock()
        config.datasets = [ds_config]
        config.sources = [source]
        config.extractors = None
        config.preprocessors = None
        config.selections = None
        config.workflows = [_CLEAN_INSTANCE]

        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            _run_single_task(task, config)

        mock_load_ds.assert_called_once_with(
            Path("data/yolo"),
            dataset_format="yolo",
            split="val",
            yaml_file=None,
            ann_dir="annotations",
        )

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_task_coco_sets_label_source(self, mock_load_ds: MagicMock):
        """_run_single_task sets label_source='annotations' for COCO datasets."""

        ds_config = CocoDatasetConfig(name="coco_ds", path="data/coco")
        source = SourceConfig(name="src_coco", dataset="coco_ds")
        task = TaskConfig(name="t", workflow="clean", sources="src_coco")

        config = MagicMock()
        config.datasets = [ds_config]
        config.sources = [source]
        config.extractors = None
        config.preprocessors = None
        config.selections = None
        config.workflows = [_CLEAN_INSTANCE]

        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            _run_single_task(task, config)

        context = mock_wf.execute.call_args[0][0]
        dc = context.dataset_contexts["src_coco"]
        assert dc.label_source == "annotations"

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_task_yolo_sets_label_source(self, mock_load_ds: MagicMock):
        """_run_single_task sets label_source='annotations' for YOLO datasets."""

        ds_config = YoloDatasetConfig(name="yolo_ds", path="data/yolo")
        source = SourceConfig(name="src_yolo", dataset="yolo_ds")
        task = TaskConfig(name="t", workflow="clean", sources="src_yolo")

        config = MagicMock()
        config.datasets = [ds_config]
        config.sources = [source]
        config.extractors = None
        config.preprocessors = None
        config.selections = None
        config.workflows = [_CLEAN_INSTANCE]

        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            _run_single_task(task, config)

        context = mock_wf.execute.call_args[0][0]
        dc = context.dataset_contexts["src_yolo"]
        assert dc.label_source == "annotations"


# ---------------------------------------------------------------------------
# _infer_label_source unit tests
# ---------------------------------------------------------------------------


class TestLabelSourceResolution:
    """Tests for label_source resolution via _LABEL_SOURCE and resolve_dataset."""

    def test_label_source_table_huggingface(self) -> None:
        from dataeval_flow.dataset import _LABEL_SOURCE

        assert _LABEL_SOURCE["huggingface"] == "huggingface"

    def test_label_source_table_coco(self) -> None:
        from dataeval_flow.dataset import _LABEL_SOURCE

        assert _LABEL_SOURCE["coco"] == "annotations"

    def test_label_source_table_yolo(self) -> None:
        from dataeval_flow.dataset import _LABEL_SOURCE

        assert _LABEL_SOURCE["yolo"] == "annotations"

    def test_label_source_table_image_folder_absent(self) -> None:
        from dataeval_flow.dataset import _LABEL_SOURCE

        assert "image_folder" not in _LABEL_SOURCE

    def test_protocol_config_returns_protocol(self) -> None:
        from dataeval_flow.config import DatasetProtocolConfig
        from dataeval_flow.dataset import resolve_dataset

        cfg = DatasetProtocolConfig(name="ds", dataset=[1, 2, 3])
        resolved = resolve_dataset(cfg)
        assert resolved.label_source == "protocol"

    def test_coco_rejects_infer_labels(self) -> None:
        """Schema rejects infer_labels=True for COCO (it's image_folder-only)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="infer_labels"):
            CocoDatasetConfig(name="ds", path="data/ds", infer_labels=True)  # type: ignore[arg-type]

    def test_yolo_rejects_infer_labels(self) -> None:
        """Schema rejects infer_labels=True for YOLO (it's image_folder-only)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="infer_labels"):
            YoloDatasetConfig(name="ds", path="data/ds", infer_labels=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Workflow discovery (replaces WorkflowRegistry)
# ---------------------------------------------------------------------------


class TestWorkflowDiscovery:
    def test_get_workflow_returns_registered(self):
        from dataeval_flow.workflow import WorkflowProtocol, get_workflow

        wf = get_workflow("data-cleaning")
        assert isinstance(wf, WorkflowProtocol)

    def test_get_workflow_unknown_raises(self):
        from dataeval_flow.workflow import get_workflow

        with pytest.raises(ValueError, match="Unknown workflow: 'nope'"):
            get_workflow("nope")

    def test_list_workflows(self):
        from dataeval_flow.workflow import list_workflows

        workflows = list_workflows()
        names = [w["name"] for w in workflows]
        assert "data-cleaning" in names


# ---------------------------------------------------------------------------
# _run_single_task — multi-source
# ---------------------------------------------------------------------------


class TestRunTaskMultiSource:
    """Tests for multi-source _run_single_task behaviour."""

    def _make_config(self, ds_names: list[str]) -> MagicMock:
        datasets = [
            HuggingFaceDatasetConfig(name=n, path=f"./{n}", split="train", task="image_classification")
            for n in ds_names
        ]
        sources = [SourceConfig(name=f"src_{n}", dataset=n) for n in ds_names]
        config = MagicMock()
        config.datasets = datasets
        config.sources = sources
        config.extractors = None
        config.preprocessors = None
        config.selections = None
        config.workflows = [_CLEAN_INSTANCE]
        return config

    def _mock_workflow(self) -> MagicMock:
        mock_result = MagicMock()
        mock_result.success = True
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = mock_result
        return mock_wf

    @patch("dataeval_flow.dataset.load_dataset")
    def test_sources_string_single(self, mock_load_ds: MagicMock):
        """sources as a plain string works (single source)."""
        config = self._make_config(["ds"])
        task = TaskConfig(name="t", workflow="clean", sources="src_ds")
        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            result = _run_single_task(task, config)

        assert result.success
        mock_load_ds.assert_called_once()

    @patch("dataeval_flow.dataset.load_dataset")
    def test_sources_list_loads_multiple(self, mock_load_ds: MagicMock):
        """sources as a list loads each dataset."""
        config = self._make_config(["ds_a", "ds_b"])
        task = TaskConfig(name="t", workflow="clean", sources=["src_ds_a", "src_ds_b"])
        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            result = _run_single_task(task, config)

        assert result.success
        assert mock_load_ds.call_count == 2

    @patch("dataeval_flow.dataset.load_dataset")
    def test_shared_extractor_applies_to_all(self, mock_load_ds: MagicMock):
        """An extractor specified on the task is shared across all sources."""
        config = self._make_config(["ds_a", "ds_b"])
        config.extractors = [
            OnnxExtractorConfig(name="ext1", model_path="./m.onnx", output_name="out", batch_size=64),
        ]
        task = TaskConfig(name="t", workflow="clean", sources=["src_ds_a", "src_ds_b"], extractor="ext1")
        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            result = _run_single_task(task, config)

        assert result.success
        context = mock_wf.execute.call_args[0][0]
        # Both datasets should have the same extractor
        for dc in context.dataset_contexts.values():
            assert dc.extractor is not None
            assert dc.extractor.model_path == "m.onnx"

    @patch("dataeval_flow.dataset.load_dataset")
    def test_no_extractor_gives_none(self, mock_load_ds: MagicMock):
        """When no extractor is specified, datasets get None extractor."""
        config = self._make_config(["ds_a", "ds_b"])
        task = TaskConfig(name="t", workflow="clean", sources=["src_ds_a", "src_ds_b"])
        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            result = _run_single_task(task, config)

        assert result.success
        context = mock_wf.execute.call_args[0][0]
        for dc in context.dataset_contexts.values():
            assert dc.extractor is None

    @patch("dataeval_flow.dataset.load_dataset")
    def test_single_source_context_fields(self, mock_load_ds: MagicMock):
        """Single-source WorkflowContext populates dataset_contexts correctly."""
        config = self._make_config(["ds"])
        task = TaskConfig(name="t", workflow="clean", sources="src_ds")
        mock_dataset = MagicMock()
        mock_load_ds.return_value = mock_dataset
        mock_wf = self._mock_workflow()

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            _run_single_task(task, config)

        context = mock_wf.execute.call_args[0][0]
        # dataset_contexts should have exactly one entry with the dataset object
        assert len(context.dataset_contexts) == 1
        assert "src_ds" in context.dataset_contexts
        assert context.dataset_contexts["src_ds"].dataset is mock_dataset

    @patch("dataeval_flow.dataset.load_dataset")
    def test_multi_source_metadata_has_comma_joined_id(self, mock_load_ds: MagicMock):
        """_run_single_task populates metadata.dataset_id with comma-joined names for multi-source."""
        config = self._make_config(["ds_a", "ds_b"])
        task = TaskConfig(name="t", workflow="clean", sources=["src_ds_a", "src_ds_b"])
        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            result = _run_single_task(task, config)

        assert result.metadata.dataset_id == "ds_a,ds_b"


# ---------------------------------------------------------------------------
# DriftMonitoringTaskConfig validation
# ---------------------------------------------------------------------------


class TestDriftMonitoringTaskConfig:
    """Tests for DriftMonitoringTaskConfig source validation."""

    def test_requires_at_least_two_sources(self):
        from dataeval_flow.config import DriftMonitoringTaskConfig

        with pytest.raises(ValueError, match="at least 2 sources"):
            DriftMonitoringTaskConfig(name="drift_task", workflow="drift", sources="single_source")

    def test_accepts_two_sources(self):
        from dataeval_flow.config import DriftMonitoringTaskConfig

        task = DriftMonitoringTaskConfig(name="drift_task", workflow="drift", sources=["src_ref", "src_test"])
        assert task.sources == ["src_ref", "src_test"]


# ---------------------------------------------------------------------------
# run_tasks — enabled filtering, task name filtering, all-disabled
# ---------------------------------------------------------------------------


class TestRunTasks:
    """Tests for run_tasks() top-level function."""

    def _build_pipeline_config(self) -> MagicMock:
        ds = HuggingFaceDatasetConfig(name="ds", path="./ds", split="train", task="image_classification")
        source = SourceConfig(name="src", dataset="ds")

        config = MagicMock()
        config.datasets = [ds]
        config.sources = [source]
        config.extractors = None
        config.preprocessors = None
        config.selections = None
        config.workflows = [_CLEAN_INSTANCE]
        config.tasks = [
            TaskConfig(name="task_a", workflow="clean", sources="src"),
            TaskConfig(name="task_b", workflow="clean", sources="src"),
            TaskConfig(name="task_disabled", workflow="clean", sources="src", enabled=False),
        ]
        return config

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_all_enabled_tasks(self, mock_load_ds: MagicMock):
        """run_tasks(config) runs only enabled tasks."""
        config = self._build_pipeline_config()
        mock_load_ds.return_value = MagicMock()
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = MagicMock(success=True)

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            results = run_tasks(config)

        assert len(results) == 2  # task_disabled skipped

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_tasks_by_name_string(self, mock_load_ds: MagicMock):
        """run_tasks(config, 'task_b') runs only task_b."""
        config = self._build_pipeline_config()
        mock_load_ds.return_value = MagicMock()
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = MagicMock(success=True)

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            results = run_tasks(config, "task_b")

        assert len(results) == 1

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_tasks_by_name_list(self, mock_load_ds: MagicMock):
        """run_tasks(config, ['task_a', 'task_b']) runs both in order."""
        config = self._build_pipeline_config()
        mock_load_ds.return_value = MagicMock()
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = MagicMock(success=True)

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            results = run_tasks(config, ["task_a", "task_b"])

        assert len(results) == 2

    def test_run_tasks_all_disabled_raises(self):
        """run_tasks raises ValueError when all tasks are disabled."""
        config = MagicMock()
        config.tasks = [
            TaskConfig(name="t1", workflow="clean", sources="src", enabled=False),
            TaskConfig(name="t2", workflow="clean", sources="src", enabled=False),
        ]

        with pytest.raises(ValueError, match="All tasks are disabled"):
            run_tasks(config)

    def test_run_tasks_no_tasks_raises(self):
        """run_tasks raises ValueError when no tasks defined."""
        config = MagicMock()
        config.tasks = None

        with pytest.raises(ValueError, match="No tasks defined"):
            run_tasks(config)

        config.tasks = []
        with pytest.raises(ValueError, match="No tasks defined"):
            run_tasks(config)

    def test_run_tasks_unknown_name_raises(self):
        """run_tasks raises ValueError for unknown task name."""
        config = MagicMock()
        config.tasks = [TaskConfig(name="t1", workflow="clean", sources="src")]

        with pytest.raises(ValueError, match="Unknown task: 'nonexistent'"):
            run_tasks(config, "nonexistent")


# ---------------------------------------------------------------------------
# select_tasks — the selection run_tasks executes, exposed for result pairing
# ---------------------------------------------------------------------------


class TestSelectTasks:
    """Tests for select_tasks(), which callers use to pair results back to tasks."""

    def _config(self) -> MagicMock:
        config = MagicMock()
        config.tasks = [
            TaskConfig(name="task_a", workflow="clean", sources="src"),
            TaskConfig(name="task_b", workflow="clean", sources="src", enabled=False),
            TaskConfig(name="task_c", workflow="clean", sources="src"),
        ]
        return config

    def test_default_selects_enabled_in_config_order(self):
        assert [t.name for t in select_tasks(self._config())] == ["task_a", "task_c"]

    def test_naming_a_task_overrides_disabled(self):
        """An explicit request outranks the config's enabled default."""
        assert [t.name for t in select_tasks(self._config(), "task_b")] == ["task_b"]

    def test_named_list_keeps_the_given_order(self):
        selected = select_tasks(self._config(), ["task_c", "task_a"])
        assert [t.name for t in selected] == ["task_c", "task_a"]

    def test_matches_what_run_tasks_executes(self, monkeypatch: pytest.MonkeyPatch):
        """The pairing contract: one selected task per result, in the same order."""
        config = self._config()
        executed: list[str] = []

        def _fake(task, cfg, data_dir=None, cache_dir=None):  # noqa: ARG001
            executed.append(task.name)
            return MagicMock(success=True)

        monkeypatch.setattr("dataeval_flow.workflow.orchestrator._run_single_task", _fake)
        results = run_tasks(config)

        assert executed == [t.name for t in select_tasks(config)]
        assert len(results) == len(select_tasks(config))

    def test_all_disabled_raises(self):
        config = MagicMock()
        config.tasks = [TaskConfig(name="t1", workflow="clean", sources="src", enabled=False)]
        with pytest.raises(ValueError, match="All tasks are disabled"):
            select_tasks(config)

    def test_no_tasks_raises(self):
        config = MagicMock()
        config.tasks = []
        with pytest.raises(ValueError, match="No tasks defined"):
            select_tasks(config)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown task: 'nope'"):
            select_tasks(self._config(), "nope")


# ---------------------------------------------------------------------------
# Same-dataset, different-source keying
# ---------------------------------------------------------------------------


class TestSourceNameKeying:
    """Tests that dataset_contexts uses source name, not dataset name, as key."""

    @patch("dataeval_flow.dataset.load_dataset")
    def test_two_sources_same_dataset_different_keys(self, mock_load_ds: MagicMock):
        """Two sources referencing the same dataset get distinct context entries."""
        from dataeval_flow.config import ViewConfig, ViewOperation

        ds = HuggingFaceDatasetConfig(name="cifar", path="./cifar", split="train", task="image_classification")
        src_full = SourceConfig(name="cifar_full", dataset="cifar")
        src_sub = SourceConfig(name="cifar_sub", dataset="cifar", view="first_5k")
        view = ViewConfig(name="first_5k", operations=[ViewOperation(type="Limit", params={"size": 5000})])

        task = TaskConfig(name="t", workflow="clean", sources=["cifar_full", "cifar_sub"])

        config = MagicMock()
        config.datasets = [ds]
        config.sources = [src_full, src_sub]
        config.extractors = None
        config.preprocessors = None
        config.views = [view]
        config.workflows = [_CLEAN_INSTANCE]

        mock_load_ds.return_value = MagicMock()
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = MagicMock(success=True)

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            _run_single_task(task, config)

        context = mock_wf.execute.call_args[0][0]
        assert "cifar_full" in context.dataset_contexts
        assert "cifar_sub" in context.dataset_contexts
        assert len(context.dataset_contexts) == 2

        # The sub source should have selection steps, the full should not
        assert context.dataset_contexts["cifar_full"].view_operations is None
        assert context.dataset_contexts["cifar_sub"].view_operations is not None


# ---------------------------------------------------------------------------
# _resolve_extractor_paths
# ---------------------------------------------------------------------------


class TestResolveExtractorPaths:
    def test_resolves_relative_model_path(self, tmp_path):
        extractor = OnnxExtractorConfig(name="ext", model_path="models/model.onnx", batch_size=32)
        resolved = _resolve_extractor_paths(extractor, data_dir=tmp_path)
        assert resolved.model_path == str(tmp_path / "models" / "model.onnx")
        assert resolved is not extractor

    def test_absolute_model_path_rejected(self, tmp_path):
        """Absolute model_path is rejected at schema validation time."""
        from pydantic import ValidationError

        abs_path = str(tmp_path / "model.onnx")
        with pytest.raises(ValidationError, match="relative"):
            OnnxExtractorConfig(name="ext", model_path=abs_path, batch_size=32)

    def test_no_model_path_returns_same(self, tmp_path):
        cfg = MagicMock(spec=[])  # no model_path attribute
        result = _resolve_extractor_paths(cfg, data_dir=tmp_path)
        assert result is cfg


# ---------------------------------------------------------------------------
# run_tasks — skipped disabled tasks
# ---------------------------------------------------------------------------


class TestRunTasksDisabledSkip:
    @patch("dataeval_flow.dataset.load_dataset")
    def test_skipped_disabled_tasks_logged(self, mock_load_ds, caplog):
        import logging

        config = MagicMock()
        config.datasets = [HuggingFaceDatasetConfig(name="ds", path="./ds", split="train", task="image_classification")]
        config.sources = [SourceConfig(name="src", dataset="ds")]
        config.extractors = None
        config.preprocessors = None
        config.selections = None
        config.workflows = [_CLEAN_INSTANCE]
        config.tasks = [
            TaskConfig(name="enabled_task", workflow="clean", sources="src"),
            TaskConfig(name="disabled_task", workflow="clean", sources="src", enabled=False),
        ]

        mock_load_ds.return_value = MagicMock()
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = MagicMock(success=True)

        with (
            patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf),
            caplog.at_level(logging.INFO, logger="dataeval_flow.workflow.orchestrator"),
        ):
            run_tasks(config)

        assert any("Skipping" in r.message and "disabled" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# run_task — public wrapper
# ---------------------------------------------------------------------------


class TestRunTaskWrapper:
    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_task_delegates(self, mock_load_ds):
        ds = HuggingFaceDatasetConfig(name="ds", path="./ds", split="train", task="image_classification")
        source = SourceConfig(name="src", dataset="ds")
        task = TaskConfig(name="my_task", workflow="clean", sources="src")

        config = MagicMock()
        config.datasets = [ds]
        config.sources = [source]
        config.extractors = None
        config.preprocessors = None
        config.selections = None
        config.workflows = [_CLEAN_INSTANCE]

        mock_load_ds.return_value = MagicMock()
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = MagicMock(success=True)

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            result = run_task(task, config)

        assert result.success
        mock_wf.execute.assert_called_once()

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_task_logs_task_header(self, mock_load_ds, caplog):
        import logging

        task = TaskConfig(name="my_task", workflow="clean", sources="src")

        config = MagicMock()
        config.datasets = [HuggingFaceDatasetConfig(name="ds", path="./ds", split="train", task="image_classification")]
        config.sources = [SourceConfig(name="src", dataset="ds")]
        config.extractors = None
        config.preprocessors = None
        config.selections = None
        config.workflows = [_CLEAN_INSTANCE]

        mock_load_ds.return_value = MagicMock()
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = MagicMock(success=True)

        with (
            patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf),
            caplog.at_level(logging.INFO, logger="dataeval_flow.workflow.orchestrator"),
        ):
            run_task(task, config)

        assert any("my_task" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# cache_dir logging + label_source annotation
# ---------------------------------------------------------------------------


class TestCacheDirAndLabelSource:
    @patch("dataeval_flow.dataset.load_dataset")
    def test_cache_dir_logs_info(self, mock_load_ds, caplog, tmp_path):
        import logging

        task = TaskConfig(name="t", workflow="clean", sources="src")
        config = MagicMock()
        config.datasets = [HuggingFaceDatasetConfig(name="ds", path="./ds", split="train", task="image_classification")]
        config.sources = [SourceConfig(name="src", dataset="ds")]
        config.extractors = None
        config.preprocessors = None
        config.selections = None
        config.workflows = [_CLEAN_INSTANCE]

        mock_load_ds.return_value = MagicMock()
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = MagicMock(success=True)

        with (
            patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf),
            caplog.at_level(logging.INFO, logger="dataeval_flow.workflow.orchestrator"),
        ):
            _run_single_task(task, config, cache_dir=tmp_path / "cache")

        assert any("Cache enabled" in r.message for r in caplog.records)

    @patch("dataeval_flow.dataset.load_dataset")
    def test_label_source_propagated(self, mock_load_ds):
        ds = YoloDatasetConfig(name="yolo_ds", path="data/yolo")
        source = SourceConfig(name="src", dataset="yolo_ds")
        task = TaskConfig(name="t", workflow="clean", sources="src")

        config = MagicMock()
        config.datasets = [ds]
        config.sources = [source]
        config.extractors = None
        config.preprocessors = None
        config.selections = None
        config.workflows = [_CLEAN_INSTANCE]

        mock_load_ds.return_value = MagicMock()
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = MagicMock(success=True)

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            result = _run_single_task(task, config)

        assert result.metadata.label_source == "annotations"


# ---------------------------------------------------------------------------
# _build_resolved_config — direct unit tests
# ---------------------------------------------------------------------------


class TestBuildResolvedConfig:
    """Direct tests for _build_resolved_config branches, over a single-operand source."""

    @staticmethod
    def _single(dataset_config: Any, view_config: Any = None, dataset: str = "ds"):
        """One ResolvedSource wrapping one operand, for exercising _operand_entry."""
        from dataeval_flow.sources import ResolvedSource, SourceOperand

        operand = SourceOperand(
            source=SourceConfig(name="src", dataset=dataset),
            dataset_config=dataset_config,
            view_config=view_config,
            raw=MagicMock(),
            label_source=None,
            cache_key="k",
        )
        return ResolvedSource(
            name="src",
            operands=(operand,),
            dataset=operand.raw,
            view_config=view_config,
            cache_name="src",
            cache_key="k",
        )

    def test_non_serializable_dataset_protocol_config(self):
        """A non-serializable dataset (DatasetProtocolConfig) writes a protocol entry."""
        from dataeval_flow.config import DatasetProtocolConfig
        from dataeval_flow.workflow.orchestrator import _build_resolved_config

        runtime_ds = MagicMock()
        runtime_ds.metadata = {"id": "my-dataset-id"}
        ds_cfg = DatasetProtocolConfig(name="proto_ds", dataset=runtime_ds)

        cfg = _build_resolved_config(
            resolved_sources=[self._single(ds_cfg, dataset="proto_ds")],
            workflow_instance=None,
            extractor_cfg=None,
            pipeline_config=None,
        )

        ds_config = cfg["sources"][0]["dataset_config"]
        assert ds_config["dataset"]["type"] == "protocol"
        assert ds_config["dataset"]["id"] == "my-dataset-id"

    def test_non_serializable_dataset_none_runtime_obj(self):
        """A non-serializable dataset with no runtime object records 'unknown'."""
        from dataeval_flow.config import DatasetProtocolConfig
        from dataeval_flow.workflow.orchestrator import _build_resolved_config

        ds_cfg = DatasetProtocolConfig(name="proto_ds", dataset=None)

        cfg = _build_resolved_config(
            resolved_sources=[self._single(ds_cfg, dataset="proto_ds")],
            workflow_instance=None,
            extractor_cfg=None,
            pipeline_config=None,
        )

        ds_config = cfg["sources"][0]["dataset_config"]
        assert ds_config["dataset"]["class"] == "unknown"

    def test_serializable_dataset_uses_model_dump(self):
        """A serializable dataset config is dumped as-is."""
        ds_cfg = ImageFolderDatasetConfig(name="photos", path="./data")
        from dataeval_flow.workflow.orchestrator import _build_resolved_config

        cfg = _build_resolved_config(
            resolved_sources=[self._single(ds_cfg, dataset="photos")],
            workflow_instance=None,
            extractor_cfg=None,
            pipeline_config=None,
        )

        assert cfg["sources"][0]["dataset_config"]["name"] == "photos"

    def test_source_with_view_writes_view_config(self):
        """An operand's view is resolved inline as `view` and `view_config`."""
        from dataeval_flow.config import ViewConfig, ViewOperation
        from dataeval_flow.workflow.orchestrator import _build_resolved_config

        ds_cfg = ImageFolderDatasetConfig(name="ds", path="./data")
        view = ViewConfig(name="sub", operations=[ViewOperation(type="Limit", params={"size": 100})])

        cfg = _build_resolved_config(
            resolved_sources=[self._single(ds_cfg, view_config=view)],
            workflow_instance=None,
            extractor_cfg=None,
            pipeline_config=None,
        )

        entry = cfg["sources"][0]
        assert entry["view"] == "sub"
        assert entry["view_config"]["name"] == "sub"

    def test_source_with_no_view_omits_view_keys(self):
        """An operand naming no view writes no `view` or `view_config` key."""
        from dataeval_flow.workflow.orchestrator import _build_resolved_config

        ds_cfg = ImageFolderDatasetConfig(name="ds", path="./data")
        cfg = _build_resolved_config(
            resolved_sources=[self._single(ds_cfg)],
            workflow_instance=None,
            extractor_cfg=None,
            pipeline_config=None,
        )

        entry = cfg["sources"][0]
        assert "view" not in entry
        assert "view_config" not in entry

    def test_workflow_instance_included(self):
        """Workflow instance is included when not None."""
        from dataeval_flow.workflow.orchestrator import _build_resolved_config

        ds_cfg = ImageFolderDatasetConfig(name="ds", path="./data")
        cfg = _build_resolved_config(
            resolved_sources=[self._single(ds_cfg)],
            workflow_instance=_CLEAN_INSTANCE,
            extractor_cfg=None,
            pipeline_config=None,
        )

        assert "workflow" in cfg
        assert cfg["workflow"]["name"] == "clean"
        assert cfg["workflow"]["type"] == "data-cleaning"

    def test_extractor_included(self):
        """Extractor config is included when not None."""
        from dataeval_flow.workflow.orchestrator import _build_resolved_config

        ds_cfg = ImageFolderDatasetConfig(name="ds", path="./data")
        ext = OnnxExtractorConfig(name="ext", model_path="./m.onnx", batch_size=32)
        cfg = _build_resolved_config(
            resolved_sources=[self._single(ds_cfg)],
            workflow_instance=None,
            extractor_cfg=ext,
            pipeline_config=None,
        )

        assert "extractor" in cfg
        assert cfg["extractor"]["name"] == "ext"

    def test_no_workflow_no_extractor(self):
        """No workflow or extractor omits those keys."""
        from dataeval_flow.workflow.orchestrator import _build_resolved_config

        ds_cfg = ImageFolderDatasetConfig(name="ds", path="./data")
        cfg = _build_resolved_config(
            resolved_sources=[self._single(ds_cfg)],
            workflow_instance=None,
            extractor_cfg=None,
            pipeline_config=None,
        )

        assert "workflow" not in cfg
        assert "extractor" not in cfg

    def test_merge_recurses_into_operands(self):
        """A merged source's entry lists `merge` operands and keeps its own view separate."""
        from dataeval_flow.config import ViewConfig, ViewOperation
        from dataeval_flow.sources import ResolvedSource, SourceOperand
        from dataeval_flow.workflow.orchestrator import _build_resolved_config

        ds_a = ImageFolderDatasetConfig(name="ds_a", path="./a")
        ds_b = ImageFolderDatasetConfig(name="ds_b", path="./b")
        own_view = ViewConfig(name="head", operations=[ViewOperation(type="Limit", params={"size": 3})])
        operand_a = SourceOperand(
            source=SourceConfig(name="a", dataset="ds_a"),
            dataset_config=ds_a,
            view_config=None,
            raw=MagicMock(),
            label_source=None,
            cache_key="ka",
        )
        operand_b = SourceOperand(
            source=SourceConfig(name="b", dataset="ds_b"),
            dataset_config=ds_b,
            view_config=None,
            raw=MagicMock(),
            label_source=None,
            cache_key="kb",
        )
        resolved = ResolvedSource(
            name="merged",
            operands=(operand_a, operand_b),
            dataset=MagicMock(),
            view_config=own_view,
            cache_name="merged",
            cache_key="merge:ka|kb",
        )

        cfg = _build_resolved_config(
            resolved_sources=[resolved],
            workflow_instance=None,
            extractor_cfg=None,
            pipeline_config=None,
        )

        entry = cfg["sources"][0]
        assert entry["name"] == "merged"
        assert [op["dataset"] for op in entry["merge"]] == ["ds_a", "ds_b"]
        assert entry["view"] == "head"
        assert "dataset" not in entry


# ---------------------------------------------------------------------------
# _populate_result_metadata — label_source falsy branch
# ---------------------------------------------------------------------------


class TestPopulateResultMetadataLabelSource:
    """label_source is left unset when no operand of any source reports one."""

    def test_no_label_source_skips_annotation(self):
        from dataeval_flow.config import SourceConfig
        from dataeval_flow.sources import ResolvedSource, SourceOperand
        from dataeval_flow.workflow import WorkflowResult
        from dataeval_flow.workflow.orchestrator import _populate_result_metadata

        result = WorkflowResult(name="t", success=True, data=MagicMock(), metadata=ResultMetadata())
        operand = SourceOperand(
            source=SourceConfig(name="src", dataset="ds"),
            dataset_config=MagicMock(),
            view_config=None,
            raw=MagicMock(),
            label_source=None,
            cache_key="k",
        )
        resolved = ResolvedSource(
            name="src",
            operands=(operand,),
            dataset=operand.raw,
            view_config=None,
            cache_name="ds",
            cache_key="k",
        )

        _populate_result_metadata(
            result=result,
            resolved_sources=[resolved],
            extractor_cfg=None,
            elapsed=1.0,
        )

        assert result.metadata.label_source is None


# ---------------------------------------------------------------------------
# run_tasks — all enabled (no skipped tasks, line 353->355)
# ---------------------------------------------------------------------------


class TestRunTasksAllEnabled:
    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_tasks_none_disabled(self, mock_load_ds: MagicMock):
        """run_tasks with all tasks enabled skips the 'Skipping' log (line 353->355)."""
        config = MagicMock()
        config.datasets = [HuggingFaceDatasetConfig(name="ds", path="./ds", split="train", task="image_classification")]
        config.sources = [SourceConfig(name="src", dataset="ds")]
        config.extractors = None
        config.preprocessors = None
        config.selections = None
        config.workflows = [_CLEAN_INSTANCE]
        config.tasks = [
            TaskConfig(name="t1", workflow="clean", sources="src"),
            TaskConfig(name="t2", workflow="clean", sources="src"),
        ]

        mock_load_ds.return_value = MagicMock()
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = MagicMock(success=True)

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            results = run_tasks(config)

        assert len(results) == 2


# ===========================================================================
# _relativize_paths
# ===========================================================================


class TestRelativizePaths:
    def test_none_root_returns_unchanged(self):
        obj = {"path": "/some/absolute/path"}
        assert _relativize_paths(obj, None) is obj

    def test_dict_paths_relativized(self, tmp_path):
        child = tmp_path / "sub" / "file.txt"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.touch()
        result = _relativize_paths({"file": str(child)}, tmp_path)
        assert result["file"] == "sub/file.txt"

    def test_list_paths_relativized(self, tmp_path):
        child = tmp_path / "a.txt"
        child.touch()
        result = _relativize_paths([str(child)], tmp_path)
        assert result == ["a.txt"]

    def test_non_child_path_unchanged(self, tmp_path):
        result = _relativize_paths("/totally/different/path", tmp_path)
        assert result == "/totally/different/path"

    def test_non_path_string_unchanged(self, tmp_path):
        assert _relativize_paths("hello", tmp_path) == "hello"

    def test_non_string_passthrough(self, tmp_path):
        assert _relativize_paths(42, tmp_path) == 42


# ===========================================================================
# Resolved dataset backfill
# ===========================================================================


class _TinyDataset:
    """Minimal MAITE-shaped classification dataset."""

    def __init__(self, size: int = 4) -> None:
        self._size = size
        self.metadata: dict[str, Any] = {"id": "tiny", "index2label": {0: "a", 1: "b"}}

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        import numpy as np

        onehot = np.zeros(2, dtype=np.float32)
        onehot[index % 2] = 1.0
        return np.zeros((3, 4, 4), dtype=np.uint8), onehot, {"id": index}


def _real_result(*, success: bool, **kwargs: Any):
    """A genuine WorkflowResult — MagicMock would mask the None we care about."""
    from dataeval_flow.workflow import WorkflowResult
    from dataeval_flow.workflows.cleaning.outputs import (
        DataCleaningMetadata,
        DataCleaningOutputs,
        DataCleaningRawOutputs,
        DataCleaningReport,
    )

    return WorkflowResult(
        name="data-cleaning",
        success=success,
        data=DataCleaningOutputs(
            raw=DataCleaningRawOutputs(dataset_size=0),
            report=DataCleaningReport(summary="", findings=[]),
        ),
        metadata=DataCleaningMetadata(),
        **kwargs,
    )


class TestResolvedDatasetBackfill:
    """Regression: a result must carry the dataset it ran on, failure included.

    Workflows attach ``dataset`` on their success path only, so a failed run
    used to come back with ``result.dataset is None`` — leaving callers (and
    notebooks doing ``assert result.dataset is not None``) with no handle on
    the inputs that produced the failure.
    """

    def _config(self, ds_names: list[str], views: Any = None) -> MagicMock:
        config = MagicMock()
        config.datasets = [
            HuggingFaceDatasetConfig(name=n, path=f"./{n}", split="train", task="image_classification")
            for n in ds_names
        ]
        config.sources = [SourceConfig(name=f"src_{n}", dataset=n) for n in ds_names]
        config.extractors = None
        config.preprocessors = None
        config.views = views
        config.workflows = [_CLEAN_INSTANCE]
        return config

    def _workflow(self, result: Any) -> MagicMock:
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = result
        return mock_wf

    @patch("dataeval_flow.dataset.load_dataset")
    def test_failed_result_carries_dataset(self, mock_load_ds: MagicMock):
        dataset = _TinyDataset()
        mock_load_ds.return_value = dataset
        config = self._config(["ds"])
        task = TaskConfig(name="t", workflow="clean", sources="src_ds")

        with patch("dataeval_flow.workflow.get_workflow", return_value=self._workflow(_real_result(success=False))):
            result = _run_single_task(task, config)

        assert not result.success
        assert result.dataset is dataset

    @patch("dataeval_flow.dataset.load_dataset")
    def test_backfilled_dataset_is_post_view(self, mock_load_ds: MagicMock):
        """The backfill reapplies the source's view, as the workflow would."""
        from dataeval_flow.config import ViewConfig, ViewOperation

        mock_load_ds.return_value = _TinyDataset(size=4)
        config = self._config(
            ["ds"], views=[ViewConfig(name="lim", operations=[ViewOperation(type="Limit", params={"size": 2})])]
        )
        config.sources = [SourceConfig(name="src_ds", dataset="ds", view="lim")]
        task = TaskConfig(name="t", workflow="clean", sources="src_ds")

        with patch("dataeval_flow.workflow.get_workflow", return_value=self._workflow(_real_result(success=False))):
            result = _run_single_task(task, config)

        assert result.dataset is not None
        assert len(result.dataset) == 2

    @patch("dataeval_flow.dataset.load_dataset")
    def test_workflow_supplied_dataset_is_not_replaced(self, mock_load_ds: MagicMock):
        """A successful workflow's own post-selection dataset wins."""
        mock_load_ds.return_value = _TinyDataset()
        own = _TinyDataset(size=1)
        config = self._config(["ds"])
        task = TaskConfig(name="t", workflow="clean", sources="src_ds")

        workflow = self._workflow(_real_result(success=True, dataset=own))
        with patch("dataeval_flow.workflow.get_workflow", return_value=workflow):
            result = _run_single_task(task, config)

        assert result.dataset is own

    @patch("dataeval_flow.dataset.load_dataset")
    def test_multi_source_failure_backfills_sources(self, mock_load_ds: MagicMock):
        """Multi-source workflows report per-source datasets, not a single one."""
        datasets = [_TinyDataset(), _TinyDataset()]
        mock_load_ds.side_effect = datasets
        config = self._config(["ds_a", "ds_b"])
        task = TaskConfig(name="t", workflow="clean", sources=["src_ds_a", "src_ds_b"])

        with patch("dataeval_flow.workflow.get_workflow", return_value=self._workflow(_real_result(success=False))):
            result = _run_single_task(task, config)

        assert result.dataset is None
        assert result.sources is not None
        assert list(result.sources) == ["src_ds_a", "src_ds_b"]
        assert result.sources["src_ds_a"] is datasets[0]


class TestValueRangeReachesTheRun:
    """One value per dataset, so the injection pass and a workflow's own cannot disagree."""

    def test_stamped_onto_the_resolved_policy(self):
        from dataeval_flow.policy import ResolvedPolicy
        from dataeval_flow.workflow.orchestrator import _apply_dataset_value_range

        policy = ResolvedPolicy()
        stamped = _apply_dataset_value_range(policy, [(0.0, 1.0), (0.0, 1.0)], "clean")
        assert stamped
        assert stamped.value_range == (0.0, 1.0)

    def test_absent_leaves_the_policy_alone(self):
        from dataeval_flow.policy import ResolvedPolicy
        from dataeval_flow.workflow.orchestrator import _apply_dataset_value_range

        policy = ResolvedPolicy()
        stamped = _apply_dataset_value_range(policy, [None, None], "clean")
        assert stamped
        assert stamped.value_range is None

    def test_disagreeing_datasets_are_refused_before_the_data_is_read(self):
        import pytest

        from dataeval_flow.policy import ResolvedPolicy
        from dataeval_flow.workflow.orchestrator import _apply_dataset_value_range

        with pytest.raises(ValueError, match="value_range") as exc:
            _apply_dataset_value_range(ResolvedPolicy(), [(0.0, 1.0), (0.0, 255.0)], "clean")
        message = str(exc.value)
        assert "0.0" in message
        assert "255.0" in message
        assert "clean" in message

    def test_a_declared_range_beside_undeclared_datasets_still_applies(self):
        from dataeval_flow.policy import ResolvedPolicy
        from dataeval_flow.workflow.orchestrator import _apply_dataset_value_range

        stamped = _apply_dataset_value_range(ResolvedPolicy(), [(0.0, 1.0), None], "clean")
        assert stamped
        assert stamped.value_range == (0.0, 1.0)

    @patch("dataeval_flow.dataset.load_dataset")
    def test_declared_on_the_dataset_reaches_the_stamped_policy(self, mock_load_ds: MagicMock):
        """The real seam: ds_config.value_range -> DatasetContext -> the run's metadata_policy."""
        ds = ImageFolderDatasetConfig(name="images", path="data/images", value_range=(0.0, 1.0))
        source = SourceConfig(name="src", dataset="images")
        task = TaskConfig(name="t", workflow="clean", sources="src")

        config = MagicMock()
        config.datasets = [ds]
        config.sources = [source]
        config.extractors = None
        config.preprocessors = None
        config.selections = None
        config.workflows = [_CLEAN_INSTANCE]

        mock_load_ds.return_value = MagicMock()
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = MagicMock(success=True)

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            _run_single_task(task, config)

        context = mock_wf.execute.call_args[0][0]
        dc = context.dataset_contexts["src"]
        assert dc.value_range == (0.0, 1.0)
        assert context.metadata_policy.value_range == (0.0, 1.0)


class TestOntologyReachesTheContext:
    """The real seam: config.ontologies -> _resolve_ontology -> WorkflowContext.ontology."""

    @patch("dataeval_flow.dataset.load_dataset")
    def test_a_named_pool_entry_reaches_the_context(self, mock_load_ds: MagicMock):
        """A workflow naming a pool entry gets a resolved ontology whose source is that name."""
        from dataeval_flow.config.schemas import OntologyConfig

        ds = ImageFolderDatasetConfig(name="images", path="data/images")
        source = SourceConfig(name="src", dataset="images")
        task = TaskConfig(name="t", workflow="coverage", sources="src")
        coverage_instance = DataCoverageWorkflowConfig(name="coverage", ontology="animals")

        config = MagicMock()
        config.datasets = [ds]
        config.sources = [source]
        config.extractors = None
        config.preprocessors = None
        config.selections = None
        config.workflows = [coverage_instance]
        config.ontologies = [
            OntologyConfig(
                name="animals",
                concepts=[  # type: ignore[arg-type]
                    {"id": "animal", "label": "animal"},
                    {"id": "cat", "label": "cat", "parents": ["animal"]},
                ],
            )
        ]

        mock_load_ds.return_value = MagicMock()
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = MagicMock(success=True)

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            _run_single_task(task, config)

        context = mock_wf.execute.call_args[0][0]
        assert context.ontology is not None
        assert context.ontology.error is None
        assert context.ontology.source == "animals"


# ---------------------------------------------------------------------------
# _run_single_task — merged sources
# ---------------------------------------------------------------------------


@pytest.mark.required
class TestMergedSourceTask:
    """A task naming a merged source reads one corpus."""

    def test_workflow_receives_one_merged_context(self):
        from dataeval_flow.config import TaskConfig
        from dataeval_flow.workflow.orchestrator import _run_single_task
        from tests.test_sources import _merge_config

        config = _merge_config()
        config.workflows = [_CLEAN_INSTANCE]
        config.tasks = [TaskConfig(name="t", workflow="clean", sources="merged")]

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.metadata = ResultMetadata()
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = mock_result

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            _run_single_task(config.tasks[0], config)

        context = mock_wf.execute.call_args[0][0]
        assert list(context.dataset_contexts) == ["merged"]
        assert len(context.dataset_contexts["merged"].dataset) == 4

    def test_dataset_id_names_every_operand(self):
        from dataeval_flow.config import TaskConfig
        from dataeval_flow.workflow.orchestrator import _run_single_task
        from tests.test_sources import _merge_config

        config = _merge_config()
        config.workflows = [_CLEAN_INSTANCE]
        config.tasks = [TaskConfig(name="t", workflow="clean", sources="merged")]

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.metadata = ResultMetadata()
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = mock_result

        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            result = _run_single_task(config.tasks[0], config)

        assert result.metadata.dataset_id == "ds_a,ds_b"


# ---------------------------------------------------------------------------
# _populate_result_metadata / _build_resolved_config — merged source envelope
# ---------------------------------------------------------------------------


def _envelope_config():
    """The merge fixture, wired to one task that reads the merged source."""
    from dataeval_flow.config import TaskConfig
    from tests.test_sources import _merge_config

    config = _merge_config()
    config.workflows = [_CLEAN_INSTANCE]
    config.tasks = [TaskConfig(name="t", workflow="clean", sources="merged")]
    return config


def _run_envelope(config):
    """Run the config's one task against a stub workflow and return the result."""
    from dataeval_flow.workflow.orchestrator import _run_single_task

    mock_result = MagicMock()
    mock_result.success = True
    mock_result.metadata = ResultMetadata()
    mock_wf = MagicMock()
    mock_wf.params_schema = None
    mock_wf.execute.return_value = mock_result
    with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
        return _run_single_task(config.tasks[0], config)


@pytest.mark.required
class TestMergedEnvelope:
    """A merged source is fully recorded in the result envelope."""

    def test_dataset_id_names_every_operand(self):
        assert _run_envelope(_envelope_config()).metadata.dataset_id == "ds_a,ds_b"

    def test_selection_id_names_the_conform_views(self):
        assert _run_envelope(_envelope_config()).metadata.selection_id == "conform_a,conform_b"

    def test_source_description_spells_the_merge(self):
        meta = _run_envelope(_envelope_config()).metadata
        assert list(meta.source_descriptions) == ["merged (merge: ds_a[conform_a] + ds_b[conform_b])"]

    def test_merged_source_records_its_own_view(self):
        """A merged source's own view lands after the operands' in `selection_id` and as a suffix."""
        from dataeval_flow.config import SourceConfig, ViewConfig, ViewOperation

        config = _envelope_config()
        assert config.views is not None
        assert config.sources is not None
        config.views.append(  # type: ignore[reportAttributeAccessIssue]
            ViewConfig(name="head", operations=[ViewOperation(type="Limit", params={"size": 3})])
        )
        config.sources[2] = SourceConfig(name="merged", merge=["a", "b"], view="head")  # type: ignore[reportIndexIssue]

        meta = _run_envelope(config).metadata
        assert meta.selection_id == "conform_a,conform_b,head"
        assert list(meta.source_descriptions) == ["merged (merge: ds_a[conform_a] + ds_b[conform_b])[head]"]

    def test_resolved_config_recurses_into_operands(self):
        cfg = _run_envelope(_envelope_config()).metadata.resolved_config
        entry = cfg["sources"][0]
        assert entry["name"] == "merged"
        assert [op["dataset"] for op in entry["merge"]] == ["ds_a", "ds_b"]
        # The Relabel that defines the label space is recorded verbatim.
        relabel = entry["merge"][0]["view_config"]["operations"][0]
        assert relabel["type"] == "Relabel"
        assert relabel["params"]["class_remap"] == {"car": "Car"}
        assert relabel["params"]["target"] == ["Person", "Car", "Truck"]

    def test_single_source_entry_is_unchanged(self):
        config = _envelope_config()
        assert config.tasks is not None
        config.tasks[0].sources = "a"
        entry = _run_envelope(config).metadata.resolved_config["sources"][0]
        assert entry["dataset"] == "ds_a"
        assert sorted(entry) == ["dataset", "dataset_config", "name", "view", "view_config"]
        assert "merge" not in entry

    def test_label_source_is_scalar_when_operands_agree(self):
        assert _run_envelope(_envelope_config()).metadata.label_source == "protocol"


@pytest.mark.required
class TestLabelSpaceRecords:
    """Each conformed operand records the vocabulary its labels were read under."""

    def test_one_record_per_conformed_operand(self):
        meta = _run_envelope(_envelope_config()).metadata
        assert [r.source for r in meta.label_space] == ["a", "b"]
        assert [r.class_remap for r in meta.label_space] == [{"car": "Car"}, {"lorry": "Truck"}]
        assert all(list(r.target) == ["Person", "Car", "Truck"] for r in meta.label_space)

    def test_digests_differ_per_operand(self):
        meta = _run_envelope(_envelope_config()).metadata
        assert meta.label_space[0].digest != meta.label_space[1].digest

    def test_scalar_digest_is_unset_when_operands_disagree(self):
        meta = _run_envelope(_envelope_config()).metadata
        assert meta.label_space_digest is None

    def test_scalar_digest_is_set_for_one_conformed_source(self):
        config = _envelope_config()
        assert config.tasks is not None
        config.tasks[0].sources = "a"
        meta = _run_envelope(config).metadata
        assert len(meta.label_space) == 1
        assert meta.label_space_digest == meta.label_space[0].digest

    def test_no_relabel_records_nothing(self):
        config = _envelope_config()
        assert config.sources is not None
        assert config.tasks is not None
        config.sources[0] = SourceConfig(name="a", dataset="ds_a")  # type: ignore[reportIndexIssue]
        config.tasks[0].sources = "a"
        meta = _run_envelope(config).metadata
        assert list(meta.label_space) == []
        assert meta.label_space_digest is None

    def test_target_defaults_to_the_remap_values_in_first_seen_order(self):
        """Relabel derives the vocabulary from the remap when `target` is omitted."""
        from dataeval_flow.sources import label_space_records

        config = _envelope_config()
        assert config.views is not None
        config.views[0].operations[0].params = {"class_remap": {"car": "Car", "van": "Car", "x": "Bus"}}
        from dataeval_flow.sources import resolve_source

        records = label_space_records([resolve_source("a", config)], None)
        assert list(records[0].target) == ["Car", "Bus"]

    def test_digest_matches_the_coverage_audit(self):
        """A run conformed by an audit's stanza carries the audit's own digest."""
        from dataeval_flow.label_space import label_space_digest, ontology_digest

        config = _envelope_config()
        assert config.tasks is not None
        config.tasks[0].sources = "a"
        meta = _run_envelope(config).metadata
        assert meta.label_space[0].digest == label_space_digest(
            ontology=ontology_digest([]),
            class_remap={"car": "Car"},
            target=["Person", "Car", "Truck"],
        )
        assert meta.label_space[0].ontology is None

    def test_a_workflow_stamped_digest_survives(self):
        """A workflow that already stamped its own digest — the coverage audit does — keeps it."""
        config = _envelope_config()
        assert config.tasks is not None
        config.tasks[0].sources = "a"

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.metadata = ResultMetadata(label_space_digest="already-stamped")
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = mock_result
        with patch("dataeval_flow.workflow.get_workflow", return_value=mock_wf):
            result = _run_single_task(config.tasks[0], config)

        assert result.metadata.label_space_digest == "already-stamped"

    def test_merged_sources_own_relabel_is_recorded_too(self):
        """A merged source's own view can carry a further Relabel, coarsening the shared target."""
        from dataeval_flow.config import SourceConfig, ViewConfig, ViewOperation
        from dataeval_flow.sources import label_space_records, resolve_source
        from tests.test_sources import _merge_config

        config = _merge_config()
        assert config.views is not None
        assert config.sources is not None
        config.views.append(  # type: ignore[reportAttributeAccessIssue]
            ViewConfig(
                name="collapse",
                operations=[
                    ViewOperation(
                        type="Relabel",
                        params={
                            "class_remap": {"Car": "Vehicle", "Truck": "Vehicle"},
                            "target": ["Person", "Vehicle"],
                        },
                    )
                ],
            )
        )
        config.sources[2] = SourceConfig(name="merged", merge=["a", "b"], view="collapse")  # type: ignore[reportIndexIssue]

        records = label_space_records([resolve_source("merged", config)], None)
        assert [r.source for r in records] == ["a", "b", "merged"]
        assert records[-1].class_remap == {"Car": "Vehicle", "Truck": "Vehicle"}
        assert list(records[-1].target) == ["Person", "Vehicle"]

    def test_a_plain_sources_view_is_not_recorded_twice(self):
        """A non-merged source's view_config is the same object as its one operand's."""
        from dataeval_flow.sources import label_space_records, resolve_source

        config = _envelope_config()
        records = label_space_records([resolve_source("a", config)], None)
        assert [r.source for r in records] == ["a"]

    def test_failed_ontology_leaves_the_label_and_digest_unset(self):
        """A failed load leaves `source` set and `ontology` None — recording the label would
        claim a vocabulary nothing was conformed to."""
        from dataeval_flow.sources import label_space_records, resolve_source
        from dataeval_flow.workflow import ResolvedOntology

        config = _envelope_config()
        ontology = ResolvedOntology(ontology=None, source="vehicles", error="ontology file not found")
        records = label_space_records([resolve_source("a", config)], ontology)
        assert records[0].ontology is None
        assert records[0].ontology_digest is None

    def test_explicit_empty_target_is_recorded_as_empty(self):
        """An explicit `target: []` is a real, empty vocabulary — not an omitted one."""
        from dataeval_flow.sources import label_space_records, resolve_source

        config = _envelope_config()
        assert config.views is not None
        config.views[0].operations[0].params = {"class_remap": {"car": "Car"}, "target": []}
        records = label_space_records([resolve_source("a", config)], None)
        assert list(records[0].target) == []

    def test_mapping_target_sorts_numerically_not_lexicographically(self):
        """A JSON config's mapping keys are strings; "10" must not sort before "2"."""
        from dataeval_flow.sources import label_space_records, resolve_source

        config = _envelope_config()
        assert config.views is not None
        config.views[0].operations[0].params = {
            "class_remap": {"car": "Car"},
            "target": {"0": "Person", "1": "Car", "2": "Truck", "10": "X"},
        }
        records = label_space_records([resolve_source("a", config)], None)
        target = list(records[0].target)
        assert target[:3] == ["Person", "Car", "Truck"]
        assert target[10] == "X"

    def test_mapping_target_pads_gaps_so_sparse_cannot_collide_with_dense(self):
        """A sparse mapping's gaps become "", so it cannot hash the same as a dense one."""
        from dataeval_flow.sources import label_space_records, resolve_source

        config = _envelope_config()
        assert config.views is not None
        config.views[0].operations[0].params = {
            "class_remap": {"car": "Car"},
            "target": {"0": "Person", "5": "Car"},
        }
        records = label_space_records([resolve_source("a", config)], None)
        assert list(records[0].target) == ["Person", "", "", "", "", "Car"]


@pytest.mark.required
class TestAuditToRunJoin:
    """A run conformed by an audit's stanza carries that audit's digest."""

    def test_analysis_digest_matches_the_audit_that_justified_it(self):
        """Run the audit, conform a source by what it emitted, and compare digests."""
        from dataeval import Ontology

        from dataeval_flow.config import ViewConfig, ViewOperation
        from dataeval_flow.sources import label_space_records, resolve_source
        from dataeval_flow.workflow import ResolvedOntology
        from dataeval_flow.workflows.coverage.ontology import _alignment

        ontology = Ontology.from_hierarchy({"Vehicle": ["Car", "Truck"], "Person": []})
        alignment = _alignment(ontology, ["car", "van"])

        # What a user pastes out of the audit's report and into a view.
        config = _envelope_config()
        assert config.views is not None
        assert config.sources is not None
        config.views[0] = ViewConfig(  # type: ignore[reportIndexIssue]
            name="conform_a",
            operations=[
                ViewOperation(
                    type="Relabel",
                    params={
                        "class_remap": alignment.paste_remap,
                        "target": alignment.target_vocabulary,
                    },
                )
            ],
        )

        records = label_space_records(
            [resolve_source("a", config)],
            ResolvedOntology(ontology=ontology, source="vehicles"),
        )
        assert records[0].digest == alignment.label_space_digest
        assert records[0].ontology == "vehicles"

    def test_any_workflow_can_declare_an_ontology(self):
        """Not only data-coverage. A conformed run of any type must be able to join."""
        from dataeval_flow.config import DataAnalysisWorkflowConfig

        instance = DataAnalysisWorkflowConfig(
            name="a", outlier_method="zscore", outlier_flags=["dimension"], ontology="vehicles"
        )
        assert instance.ontology == "vehicles"


@pytest.mark.required
class TestRelabelTarget:
    """`_relabel_target` in isolation — the vocabulary-derivation rules `Relabel` itself applies."""

    def test_sequence_is_taken_as_given(self):
        from dataeval_flow.sources import _relabel_target

        assert _relabel_target(["Person", "Car"], {}) == ["Person", "Car"]

    def test_explicit_empty_sequence_stays_empty(self):
        from dataeval_flow.sources import _relabel_target

        assert _relabel_target([], {"car": "Car"}) == []

    def test_omitted_derives_from_the_remap_values_first_seen_first(self):
        from dataeval_flow.sources import _relabel_target

        assert _relabel_target(None, {"car": "Car", "van": "Car", "x": "Bus"}) == ["Car", "Bus"]

    def test_dense_mapping_is_ordered_by_index(self):
        from dataeval_flow.sources import _relabel_target

        assert _relabel_target({1: "Car", 0: "Person"}, {}) == ["Person", "Car"]

    def test_mapping_keys_sort_numerically(self):
        from dataeval_flow.sources import _relabel_target

        assert _relabel_target({"10": "X", "2": "Y", "0": "Z"}, {}) == ["Z", "", "Y", "", "", "", "", "", "", "", "X"]

    def test_sparse_mapping_pads_with_empty_string(self):
        from dataeval_flow.sources import _relabel_target

        assert _relabel_target({0: "Person", 5: "Car"}, {}) == ["Person", "", "", "", "", "Car"]


@pytest.mark.required
class TestValueRangeOf:
    """_value_range_of reports a range only where every operand declares the same one."""

    @staticmethod
    def _resolved(*ranges: tuple[float, float] | None):
        from dataeval_flow.sources import ResolvedSource, SourceOperand

        operands = []
        for value_range in ranges:
            dataset_config = MagicMock()
            dataset_config.value_range = value_range
            operands.append(
                SourceOperand(
                    source=SourceConfig(name="s", dataset="ds"),
                    dataset_config=dataset_config,
                    view_config=None,
                    raw=MagicMock(),
                    label_source=None,
                    cache_key="k",
                )
            )
        return ResolvedSource(
            name="s",
            operands=tuple(operands),
            dataset=MagicMock(),
            view_config=None,
            cache_name="s",
            cache_key="k",
        )

    def test_one_operand_reports_its_own_range(self):
        from dataeval_flow.workflow.orchestrator import _value_range_of

        assert _value_range_of(self._resolved((0.0, 1.0))) == (0.0, 1.0)

    def test_agreeing_operands_report_the_shared_range(self):
        from dataeval_flow.workflow.orchestrator import _value_range_of

        assert _value_range_of(self._resolved((0.0, 1.0), (0.0, 1.0))) == (0.0, 1.0)

    def test_undeclared_operands_report_none(self):
        from dataeval_flow.workflow.orchestrator import _value_range_of

        assert _value_range_of(self._resolved(None, None)) is None

    def test_disagreeing_operands_are_refused(self):
        from dataeval_flow.workflow.orchestrator import _value_range_of

        with pytest.raises(ValueError, match=r"merges datasets declaring different"):
            _value_range_of(self._resolved((0.0, 1.0), (0.0, 255.0)))


@pytest.mark.required
class TestLabelSourceOf:
    """_label_source_of reports one provenance only where every operand shares it."""

    def test_agreeing_operands_report_the_shared_value(self):
        from dataeval_flow.workflow.orchestrator import _label_source_of

        assert _label_source_of(["protocol", "protocol"]) == "protocol"

    def test_all_unknown_reports_none(self):
        from dataeval_flow.workflow.orchestrator import _label_source_of

        assert _label_source_of([None, None]) is None

    def test_differing_operands_report_each(self):
        from dataeval_flow.workflow.orchestrator import _label_source_of

        assert _label_source_of(["protocol", "filepath"]) == ["protocol", "filepath"]

    def test_one_unknown_operand_is_named_unknown(self):
        from dataeval_flow.workflow.orchestrator import _label_source_of

        assert _label_source_of(["protocol", None]) == ["protocol", "unknown"]


def _resolved_source_with(*, channel_groups):
    """A ResolvedSource whose operands declare the given `channel_groups`, in order.

    Carries `source.name` as well as `dataset_config`: Task 4 refactors
    `_channel_groups_of` onto a shared helper that reads the operand's source name.
    """
    from types import SimpleNamespace
    from typing import cast

    from dataeval_flow.sources import ResolvedSource

    operands = tuple(
        SimpleNamespace(
            source=SimpleNamespace(name=f"leaf{i}"),
            dataset_config=SimpleNamespace(channel_groups=groups),
        )
        for i, groups in enumerate(channel_groups)
    )
    return cast(ResolvedSource, SimpleNamespace(name="src", operands=operands))


@pytest.mark.required
class TestChannelGroupsOnContext:
    """A merged source's operands must agree about what its bands are."""

    def test_carries_the_declared_groups(self):
        from dataeval_flow.workflow.orchestrator import _channel_groups_of

        resolved = _resolved_source_with(channel_groups=[{"rgb": [0, 1, 2], "ir": 3}])
        assert _channel_groups_of(resolved) == {"rgb": (0, 1, 2), "ir": (3,)}

    def test_none_when_nothing_declares_groups(self):
        from dataeval_flow.workflow.orchestrator import _channel_groups_of

        assert _channel_groups_of(_resolved_source_with(channel_groups=[None, None])) is None

    def test_refuses_operands_defining_one_name_differently(self):
        from dataeval_flow.workflow.orchestrator import _channel_groups_of

        resolved = _resolved_source_with(channel_groups=[{"rgb": [0, 1, 2]}, {"rgb": [0, 1]}])
        with pytest.raises(ValueError, match="different bands for channel group 'rgb'"):
            _channel_groups_of(resolved)

    def test_unions_groups_the_operands_do_not_share(self):
        from dataeval_flow.workflow.orchestrator import _channel_groups_of

        resolved = _resolved_source_with(channel_groups=[{"rgb": [0, 1, 2]}, {"ir": 3}])
        assert _channel_groups_of(resolved) == {"rgb": (0, 1, 2), "ir": (3,)}


@pytest.mark.required
class TestResolveStatsPolicy:
    """A named stats policy is resolved before the dataset is walked."""

    def _contexts(self, *groups):
        from dataeval_flow.workflow import DatasetContext

        return {
            f"s{i}": DatasetContext(name=f"s{i}", dataset=[], channel_groups=g)  # type: ignore[arg-type]
            for i, g in enumerate(groups)
        }

    def test_unions_groups_across_the_datasets_a_workflow_reads(self):
        from dataeval_flow.workflow.orchestrator import _channel_groups_for

        merged = _channel_groups_for(self._contexts({"rgb": (0, 1, 2)}, {"ir": (3,)}))
        assert merged == {"rgb": (0, 1, 2), "ir": (3,)}

    def test_refuses_two_datasets_defining_one_group_differently(self):
        from dataeval_flow.workflow.orchestrator import _channel_groups_for

        with pytest.raises(ValueError, match="different bands for channel group 'rgb'"):
            _channel_groups_for(self._contexts({"rgb": (0, 1, 2)}, {"rgb": (0, 1)}))

    def test_none_for_a_workflow_that_computes_no_statistics(self):
        from dataeval_flow.config import PipelineConfig
        from dataeval_flow.workflow.orchestrator import _resolve_stats_policy
        from dataeval_flow.workflows.splitting.params import DataSplittingParameters

        instance = DataSplittingParameters(name="s", type="data-splitting")  # type: ignore[call-arg]
        assert _resolve_stats_policy(instance, PipelineConfig(), {}) is None
