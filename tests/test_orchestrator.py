"""Tests for workflow orchestrator — run_task, _resolve_by_name, _write_output."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from dataeval_app.workflow.orchestrator import _resolve_by_name, _write_output, run_task

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
# _write_output
# ---------------------------------------------------------------------------


def _make_task(name: str = "test_task", output_format: str = "json", dataset: str = "ds") -> MagicMock:
    task = MagicMock()
    task.name = name
    task.output_format = output_format
    task.dataset = dataset
    return task


def _make_result(summary: str = "done", findings: list | None = None) -> MagicMock:
    result = MagicMock()
    report = MagicMock()
    report.summary = summary
    report.findings = findings or []
    result.data = MagicMock()
    result.data.report = report
    result.data.model_dump.return_value = {"raw": {}, "report": {"summary": summary}}
    return result


class TestWriteOutput:
    def test_terminal_format_prints(self, capsys: pytest.CaptureFixture):
        task = _make_task(output_format="terminal")
        result = _make_result(summary="All good")
        _write_output(result, task)
        out = capsys.readouterr().out
        assert "test_task" in out
        assert "All good" in out

    def test_terminal_format_prints_findings(self, capsys: pytest.CaptureFixture):
        """Terminal output includes finding titles and descriptions."""
        task = _make_task(output_format="terminal")
        finding = MagicMock()
        finding.title = "Image Outliers"
        finding.description = "5 images flagged"
        result = _make_result(summary="Done", findings=[finding])
        _write_output(result, task)
        out = capsys.readouterr().out
        assert "Image Outliers" in out
        assert "5 images flagged" in out

    def test_terminal_format_finding_no_description(self, capsys: pytest.CaptureFixture):
        """Terminal output handles findings with description=None."""
        task = _make_task(output_format="terminal")
        finding = MagicMock()
        finding.title = "Duplicates"
        finding.description = None
        result = _make_result(summary="Done", findings=[finding])
        _write_output(result, task)
        out = capsys.readouterr().out
        assert "Duplicates" in out
        # description=None should be skipped, not printed
        assert "None" not in out

    def test_json_format_creates_file(self, tmp_path: Path):
        task = _make_task(output_format="json")
        result = _make_result()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        _write_output(result, task, output_dir=output_dir)

        results_file = output_dir / task.name / "results.json"
        assert results_file.exists()
        data = json.loads(results_file.read_text())
        assert "raw" in data

    def test_yaml_format_creates_file(self, tmp_path: Path):
        task = _make_task(output_format="yaml")
        result = _make_result()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        _write_output(result, task, output_dir=output_dir)

        results_file = output_dir / task.name / "results.yaml"
        assert results_file.exists()
        data = yaml.safe_load(results_file.read_text())
        assert "raw" in data

    def test_fallback_to_terminal_when_no_output_dir(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        """When output_dir doesn't exist, fall back to terminal."""
        task = _make_task(output_format="json")
        result = _make_result(summary="Fallback")
        nonexistent = tmp_path / "does_not_exist"
        _write_output(result, task, output_dir=nonexistent)
        out = capsys.readouterr().out
        assert "Fallback" in out

    def test_json_writes_metadata(self, tmp_path: Path):
        """JSON output also writes JATIC metadata.json."""
        task = _make_task(output_format="json")
        result = _make_result()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        _write_output(result, task, output_dir=output_dir)

        metadata_file = output_dir / task.name / "metadata.json"
        assert metadata_file.exists()

    def test_yaml_writes_metadata(self, tmp_path: Path):
        """YAML output also writes JATIC metadata.json."""
        task = _make_task(output_format="yaml")
        result = _make_result()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        _write_output(result, task, output_dir=output_dir)

        metadata_file = output_dir / task.name / "metadata.json"
        assert metadata_file.exists()


class TestRunTaskWriteOutputGuard:
    """Verify run_task only calls _write_output when result.success is True."""

    @patch("dataeval_app.dataset.load_dataset")
    @patch("dataeval_app.workflow.orchestrator._write_output")
    def test_write_output_skipped_on_failure(self, mock_write: MagicMock, mock_load_ds: MagicMock):
        """Failed workflow result should NOT trigger _write_output."""
        mock_load_ds.return_value = MagicMock()

        # Workflow that returns a failed result
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.errors = ["simulated failure"]
        mock_workflow = MagicMock()
        mock_workflow.params_schema = None
        mock_workflow.execute.return_value = mock_result

        from dataeval_app.config.schemas.dataset import DatasetConfig
        from dataeval_app.config.schemas.task import TaskConfig

        ds = DatasetConfig(name="ds", format="huggingface", path="./x", splits=["train"])
        task = TaskConfig(name="t", workflow="data-cleaning", dataset="ds")
        config = MagicMock()
        config.datasets = [ds]
        config.preprocessors = None
        config.models = None
        config.selections = None

        with patch("dataeval_app.workflow.get_workflow", return_value=mock_workflow):
            result = run_task(task, config)

        assert not result.success
        mock_write.assert_not_called()

    @patch("dataeval_app.dataset.load_dataset")
    @patch("dataeval_app.workflow.orchestrator._write_output")
    def test_write_output_called_on_success(self, mock_write: MagicMock, mock_load_ds: MagicMock):
        """Successful workflow result should trigger _write_output."""
        mock_load_ds.return_value = MagicMock()

        mock_result = MagicMock()
        mock_result.success = True
        mock_workflow = MagicMock()
        mock_workflow.params_schema = None
        mock_workflow.execute.return_value = mock_result

        from dataeval_app.config.schemas.dataset import DatasetConfig
        from dataeval_app.config.schemas.task import TaskConfig

        ds = DatasetConfig(name="ds", format="huggingface", path="./x", splits=["train"])
        task = TaskConfig(name="t", workflow="data-cleaning", dataset="ds")
        config = MagicMock()
        config.datasets = [ds]
        config.preprocessors = None
        config.models = None
        config.selections = None

        with patch("dataeval_app.workflow.get_workflow", return_value=mock_workflow):
            result = run_task(task, config)

        assert result.success
        mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# run_task (integration with mocks)
# ---------------------------------------------------------------------------


class TestRunTask:
    """Tests for run_task().

    Note: run_task() uses lazy imports inside the function body, so we
    patch at the source module level (e.g. dataeval_app.dataset.load_dataset)
    rather than on orchestrator.
    """

    def _build_config_and_task(self) -> tuple[MagicMock, Any]:
        """Build minimal config + task for run_task testing."""
        from dataeval_app.config.schemas.dataset import DatasetConfig
        from dataeval_app.config.schemas.task import TaskConfig

        ds_config = DatasetConfig(
            name="test_ds",
            format="huggingface",
            path="./test",
            splits=["train"],
        )
        task_config = TaskConfig(
            name="test_task",
            workflow="data-cleaning",
            dataset="test_ds",
            params={
                "outlier_method": "zscore",
                "outlier_flags": ["dimension", "pixel"],
                "outlier_threshold": None,
            },
        )

        config = MagicMock()
        config.datasets = [ds_config]
        config.preprocessors = None
        config.models = None
        config.selections = None

        return config, task_config

    def _mock_workflow(self, params_schema: Any = None) -> MagicMock:
        """Build a mock workflow that returns a mock result."""
        mock_result = MagicMock()
        mock_result.success = True
        mock_workflow = MagicMock()
        mock_workflow.params_schema = params_schema
        mock_workflow.execute.return_value = mock_result
        return mock_workflow

    @patch("dataeval_app.dataset.load_dataset")
    def test_run_task_basic(self, mock_load_ds: MagicMock):
        """run_task resolves config, runs workflow, returns result."""
        config, task = self._build_config_and_task()
        mock_load_ds.return_value = MagicMock()

        mock_wf = self._mock_workflow()

        with (
            patch("dataeval_app.workflow.get_workflow", return_value=mock_wf),
            patch("dataeval_app.workflow.orchestrator._write_output"),
        ):
            result = run_task(task, config)

        assert result.success
        mock_load_ds.assert_called_once()
        mock_wf.execute.assert_called_once()

    @patch("dataeval_app.dataset.load_dataset")
    @patch("dataeval_app.preprocessing.build_preprocessing")
    def test_run_task_with_preprocessor(self, mock_build_pre: MagicMock, mock_load_ds: MagicMock):
        """run_task resolves preprocessor when task references one."""
        from dataeval_app.config.schemas.preprocessor import PreprocessorConfig
        from dataeval_app.config.schemas.task import TaskConfig
        from dataeval_app.preprocessing import PreprocessingStep

        config, _ = self._build_config_and_task()
        config.preprocessors = [
            PreprocessorConfig(name="basic", steps=[PreprocessingStep(step="ToTensor")]),
        ]

        task = TaskConfig(
            name="test_task",
            workflow="data-cleaning",
            dataset="test_ds",
            preprocessor="basic",
            params={
                "outlier_method": "zscore",
                "outlier_flags": ["dimension", "pixel"],
                "outlier_threshold": None,
            },
        )

        mock_load_ds.return_value = MagicMock()
        mock_build_pre.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with (
            patch("dataeval_app.workflow.get_workflow", return_value=mock_wf),
            patch("dataeval_app.workflow.orchestrator._write_output"),
        ):
            result = run_task(task, config)

        assert result.success
        mock_build_pre.assert_called_once()

    @patch("dataeval_app.dataset.load_dataset")
    def test_run_task_with_model(self, mock_load_ds: MagicMock):
        """run_task resolves model when task references one."""
        from dataeval_app.config.models import ModelConfig
        from dataeval_app.config.schemas.task import TaskConfig

        config, _ = self._build_config_and_task()
        from dataeval_app.config.models import OnnxExtractorConfig

        config.models = [
            ModelConfig(
                name="resnet",
                extractor=OnnxExtractorConfig(model_path="/model.onnx", output_name="layer4"),
            )
        ]

        task = TaskConfig(
            name="test_task",
            workflow="data-cleaning",
            dataset="test_ds",
            model="resnet",
            params={
                "outlier_method": "zscore",
                "outlier_flags": ["dimension", "pixel"],
                "outlier_threshold": None,
            },
        )

        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with (
            patch("dataeval_app.workflow.get_workflow", return_value=mock_wf),
            patch("dataeval_app.workflow.orchestrator._write_output"),
        ):
            result = run_task(task, config)

        assert result.success
        # Verify extractor config was passed into WorkflowContext
        context = mock_wf.execute.call_args[0][0]
        assert context.extractor is not None
        assert context.extractor.model_path == "/model.onnx"
        assert context.extractor.output_name == "layer4"

    @patch("dataeval_app.dataset.load_dataset")
    def test_run_task_with_selection(self, mock_load_ds: MagicMock):
        """run_task resolves selection config."""
        from dataeval_app.config.schemas.selection import SelectionConfig, SelectionStep
        from dataeval_app.config.schemas.task import TaskConfig

        config, _ = self._build_config_and_task()
        config.selections = [
            SelectionConfig(name="sub", steps=[SelectionStep(type="Limit", params={"size": 100})]),
        ]

        task = TaskConfig(
            name="test_task",
            workflow="data-cleaning",
            dataset="test_ds",
            selection="sub",
            params={
                "outlier_method": "zscore",
                "outlier_flags": ["dimension", "pixel"],
                "outlier_threshold": None,
            },
        )

        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with (
            patch("dataeval_app.workflow.get_workflow", return_value=mock_wf),
            patch("dataeval_app.workflow.orchestrator._write_output"),
        ):
            result = run_task(task, config)

        assert result.success
        # Verify selection steps were passed into WorkflowContext as SelectionStep objects
        context = mock_wf.execute.call_args[0][0]
        assert len(context.selection_steps) == 1
        assert context.selection_steps[0].type == "Limit"
        assert context.selection_steps[0].params == {"size": 100}

    @patch("dataeval_app.dataset.load_dataset")
    def test_run_task_validates_params(self, mock_load_ds: MagicMock):
        """run_task validates params against workflow.params_schema."""
        from dataeval_app.workflows.cleaning.params import DataCleaningParameters

        config, task = self._build_config_and_task()
        mock_load_ds.return_value = MagicMock()

        mock_wf = self._mock_workflow(params_schema=DataCleaningParameters)

        with (
            patch("dataeval_app.workflow.get_workflow", return_value=mock_wf),
            patch("dataeval_app.workflow.orchestrator._write_output"),
        ):
            result = run_task(task, config)

        assert result.success
        # Verify the params were validated
        mock_wf.execute.assert_called_once()
        call_args = mock_wf.execute.call_args
        params = call_args[0][1]
        assert isinstance(params, DataCleaningParameters)
        assert params.outlier_method == "zscore"

    def test_run_task_raises_on_missing_dataset(self):
        """run_task raises ValueError when dataset not found."""
        from dataeval_app.config.schemas.task import TaskConfig

        config = MagicMock()
        config.datasets = []
        task = TaskConfig(name="t", workflow="data-cleaning", dataset="nonexistent")

        with pytest.raises(ValueError, match="Unknown dataset"):
            run_task(task, config)

    @patch("dataeval_app.dataset.load_dataset")
    def test_run_task_raises_on_missing_preprocessor(self, mock_load_ds: MagicMock):
        """run_task raises ValueError when preprocessor not found."""
        from dataeval_app.config.schemas.task import TaskConfig

        config, _ = self._build_config_and_task()
        config.preprocessors = []  # Empty list — no preprocessors defined

        task = TaskConfig(
            name="t",
            workflow="data-cleaning",
            dataset="test_ds",
            preprocessor="nonexistent",
        )
        mock_load_ds.return_value = MagicMock()

        with pytest.raises(ValueError, match="Unknown preprocessor"):
            run_task(task, config)

    @patch("dataeval_app.dataset.load_dataset")
    def test_run_task_raises_on_missing_model(self, mock_load_ds: MagicMock):
        """run_task raises ValueError when model not found."""
        from dataeval_app.config.schemas.task import TaskConfig

        config, _ = self._build_config_and_task()
        config.models = []  # Empty list — no models defined

        task = TaskConfig(
            name="t",
            workflow="data-cleaning",
            dataset="test_ds",
            model="nonexistent",
        )
        mock_load_ds.return_value = MagicMock()

        with pytest.raises(ValueError, match="Unknown model"):
            run_task(task, config)

    @patch("dataeval_app.dataset.load_dataset")
    def test_run_task_raises_on_missing_selection(self, mock_load_ds: MagicMock):
        """run_task raises ValueError when selection not found."""
        from dataeval_app.config.schemas.task import TaskConfig

        config, _ = self._build_config_and_task()
        config.selections = []  # Empty list — no selections defined

        task = TaskConfig(
            name="t",
            workflow="data-cleaning",
            dataset="test_ds",
            selection="nonexistent",
        )
        mock_load_ds.return_value = MagicMock()

        with pytest.raises(ValueError, match="Unknown selection"):
            run_task(task, config)


# ---------------------------------------------------------------------------
# Workflow discovery (replaces WorkflowRegistry)
# ---------------------------------------------------------------------------


class TestWorkflowDiscovery:
    def test_get_workflow_returns_registered(self):
        from dataeval_app.workflow import WorkflowProtocol, get_workflow

        wf = get_workflow("data-cleaning")
        assert isinstance(wf, WorkflowProtocol)

    def test_get_workflow_unknown_raises(self):
        from dataeval_app.workflow import get_workflow

        with pytest.raises(ValueError, match="Unknown workflow: 'nope'"):
            get_workflow("nope")

    def test_list_workflows(self):
        from dataeval_app.workflow import list_workflows

        workflows = list_workflows()
        names = [w["name"] for w in workflows]
        assert "data-cleaning" in names
