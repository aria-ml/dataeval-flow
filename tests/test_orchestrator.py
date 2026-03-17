"""Tests for workflow orchestrator — _run_single_task, _resolve_by_name."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dataeval_flow.config import (
    CocoDatasetConfig,
    DataCleaningWorkflowConfig,
    HuggingFaceDatasetConfig,
    ImageFolderDatasetConfig,
    OnnxExtractorConfig,
    SourceConfig,
    TaskConfig,
    YoloDatasetConfig,
)
from dataeval_flow.workflow.orchestrator import _resolve_by_name, _run_single_task, run_tasks

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

        ds_config = HuggingFaceDatasetConfig(name="test_ds", path="./test", split="train")
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
            OnnxExtractorConfig(name="ext1", model_path="/model.onnx", preprocessor="basic", batch_size=64),
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
            OnnxExtractorConfig(name="ext1", model_path="/model.onnx", output_name="layer4", batch_size=64),
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
        assert dc.extractor.model_path == "/model.onnx"
        assert dc.extractor.output_name == "layer4"

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_task_with_selection(self, mock_load_ds: MagicMock):
        """_run_single_task resolves selection config from source."""
        from dataeval_flow.config import SelectionConfig, SelectionStep

        ds_config = HuggingFaceDatasetConfig(name="test_ds", path="./test", split="train")
        source = SourceConfig(name="src_test", dataset="test_ds", selection="sub")

        task = TaskConfig(name="test_task", workflow="clean", sources="src_test")

        config = MagicMock()
        config.datasets = [ds_config]
        config.sources = [source]
        config.extractors = None
        config.preprocessors = None
        config.selections = [
            SelectionConfig(name="sub", steps=[SelectionStep(type="Limit", params={"size": 100})]),
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
        assert len(dc.selection_steps) == 1
        assert dc.selection_steps[0].type == "Limit"
        assert dc.selection_steps[0].params == {"size": 100}

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
        from pathlib import Path

        ds_config = ImageFolderDatasetConfig(name="photos", path="/data/photos", recursive=True, infer_labels=True)
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
            Path("/data/photos"),
            dataset_format="image_folder",
            recursive=True,
            infer_labels=True,
        )

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_task_passes_coco_params(self, mock_load_ds: MagicMock):
        """_run_single_task passes COCO-specific config fields to load_dataset."""
        from pathlib import Path

        ds_config = CocoDatasetConfig(
            name="coco_ds",
            path="/data/coco",
            annotations_file="instances.json",
            images_dir="train2017",
            classes_file="classes.txt",
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
            Path("/data/coco"),
            dataset_format="coco",
            annotations_file="instances.json",
            images_dir="train2017",
            classes_file="classes.txt",
        )

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_task_passes_yolo_params(self, mock_load_ds: MagicMock):
        """_run_single_task passes YOLO-specific config fields to load_dataset."""
        from pathlib import Path

        ds_config = YoloDatasetConfig(
            name="yolo_ds", path="/data/yolo", images_dir="imgs", labels_dir="lbls", classes_file="cls.txt"
        )
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
            Path("/data/yolo"),
            dataset_format="yolo",
            images_dir="imgs",
            labels_dir="lbls",
            classes_file="cls.txt",
        )

    @patch("dataeval_flow.dataset.load_dataset")
    def test_run_task_coco_sets_label_source(self, mock_load_ds: MagicMock):
        """_run_single_task sets label_source='annotations' for COCO datasets."""

        ds_config = CocoDatasetConfig(name="coco_ds", path="/data/coco")
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

        ds_config = YoloDatasetConfig(name="yolo_ds", path="/data/yolo")
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
            CocoDatasetConfig(name="ds", path="/data/ds", infer_labels=True)  # type: ignore[arg-type]

    def test_yolo_rejects_infer_labels(self) -> None:
        """Schema rejects infer_labels=True for YOLO (it's image_folder-only)."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="infer_labels"):
            YoloDatasetConfig(name="ds", path="/data/ds", infer_labels=True)  # type: ignore[arg-type]


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
        datasets = [HuggingFaceDatasetConfig(name=n, path=f"./{n}", split="train") for n in ds_names]
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
            OnnxExtractorConfig(name="ext1", model_path="/m.onnx", output_name="out", batch_size=64),
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
            assert dc.extractor.model_path == "/m.onnx"

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
        ds = HuggingFaceDatasetConfig(name="ds", path="./ds", split="train")
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
# Same-dataset, different-source keying
# ---------------------------------------------------------------------------


class TestSourceNameKeying:
    """Tests that dataset_contexts uses source name, not dataset name, as key."""

    @patch("dataeval_flow.dataset.load_dataset")
    def test_two_sources_same_dataset_different_keys(self, mock_load_ds: MagicMock):
        """Two sources referencing the same dataset get distinct context entries."""
        from dataeval_flow.config import SelectionConfig, SelectionStep

        ds = HuggingFaceDatasetConfig(name="cifar", path="./cifar", split="train")
        src_full = SourceConfig(name="cifar_full", dataset="cifar")
        src_sub = SourceConfig(name="cifar_sub", dataset="cifar", selection="first_5k")
        sel = SelectionConfig(name="first_5k", steps=[SelectionStep(type="Limit", params={"size": 5000})])

        task = TaskConfig(name="t", workflow="clean", sources=["cifar_full", "cifar_sub"])

        config = MagicMock()
        config.datasets = [ds]
        config.sources = [src_full, src_sub]
        config.extractors = None
        config.preprocessors = None
        config.selections = [sel]
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
        assert context.dataset_contexts["cifar_full"].selection_steps is None
        assert context.dataset_contexts["cifar_sub"].selection_steps is not None
