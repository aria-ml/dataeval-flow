"""Tests for workflow orchestrator — run_task, _resolve_by_name, _write_output."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from dataeval_app.workflow.orchestrator import (
    _resolve_by_name,
    _resolve_optional_mapping,
    _validate_mapping_keys,
    _write_output,
    run_task,
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
# _write_output
# ---------------------------------------------------------------------------


def _make_task(name: str = "test_task", output_format: str = "json", dataset: str = "ds") -> MagicMock:
    task = MagicMock()
    task.name = name
    task.output_format = output_format
    task.datasets = dataset
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

        ds = DatasetConfig(name="ds", format="huggingface", path="./x", split="train")
        task = TaskConfig(name="t", workflow="data-cleaning", datasets="ds")
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

        ds = DatasetConfig(name="ds", format="huggingface", path="./x", split="train")
        task = TaskConfig(name="t", workflow="data-cleaning", datasets="ds")
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
            split="train",
        )
        task_config = TaskConfig(
            name="test_task",
            workflow="data-cleaning",
            datasets="test_ds",
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
            datasets="test_ds",
            preprocessors="basic",
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
            datasets="test_ds",
            models="resnet",
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
        # Verify extractor config was passed into DatasetContext
        context = mock_wf.execute.call_args[0][0]
        dc = context.dataset_contexts["test_ds"]
        assert dc.extractor is not None
        assert dc.extractor.model_path == "/model.onnx"
        assert dc.extractor.output_name == "layer4"

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
            datasets="test_ds",
            selections="sub",
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
        # Verify selection steps were passed into DatasetContext
        context = mock_wf.execute.call_args[0][0]
        dc = context.dataset_contexts["test_ds"]
        assert len(dc.selection_steps) == 1
        assert dc.selection_steps[0].type == "Limit"
        assert dc.selection_steps[0].params == {"size": 100}

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
        task = TaskConfig(name="t", workflow="data-cleaning", datasets="nonexistent")

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
            datasets="test_ds",
            preprocessors="nonexistent",
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
            datasets="test_ds",
            models="nonexistent",
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
            datasets="test_ds",
            selections="nonexistent",
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


# ---------------------------------------------------------------------------
# _resolve_optional_mapping
# ---------------------------------------------------------------------------


class TestResolveOptionalMapping:
    """Tests for _resolve_optional_mapping helper."""

    def test_none_returns_none(self):
        result = _resolve_optional_mapping(None, "ds", [_Named("a")], "model")
        assert result is None

    def test_string_resolves_shared(self):
        items = [_Named("shared_model")]
        result = _resolve_optional_mapping("shared_model", "ds", items, "model")
        assert result is not None
        assert result.name == "shared_model"

    def test_string_raises_on_missing(self):
        items = [_Named("a")]
        with pytest.raises(ValueError, match="Unknown model"):
            _resolve_optional_mapping("missing", "ds", items, "model")

    def test_mapping_resolves_by_dataset_key(self):
        items = [_Named("m1"), _Named("m2")]
        mapping = {"ds_a": "m1", "ds_b": "m2"}
        result = _resolve_optional_mapping(mapping, "ds_a", items, "model")
        assert result is not None
        assert result.name == "m1"

    def test_mapping_missing_key_returns_none(self):
        items = [_Named("m1")]
        mapping = {"ds_a": "m1"}
        result = _resolve_optional_mapping(mapping, "ds_b", items, "model")
        assert result is None

    def test_mapping_bad_name_raises(self):
        items = [_Named("m1")]
        mapping = {"ds_a": "nonexistent"}
        with pytest.raises(ValueError, match="Unknown model"):
            _resolve_optional_mapping(mapping, "ds_a", items, "model")


# ---------------------------------------------------------------------------
# _validate_mapping_keys
# ---------------------------------------------------------------------------


class TestValidateMappingKeys:
    """Tests for _validate_mapping_keys helper."""

    def test_valid_keys_pass(self):
        _validate_mapping_keys({"ds_a": "m1", "ds_b": "m2"}, ["ds_a", "ds_b"], "model")

    def test_extra_keys_raise(self):
        with pytest.raises(ValueError, match="unknown datasets"):
            _validate_mapping_keys({"ds_a": "m1", "ds_c": "m2"}, ["ds_a", "ds_b"], "model")

    def test_empty_mapping_passes(self):
        _validate_mapping_keys({}, ["ds_a"], "model")


# ---------------------------------------------------------------------------
# run_task — multi-dataset
# ---------------------------------------------------------------------------


class TestRunTaskMultiDataset:
    """Tests for multi-dataset run_task behaviour."""

    def _make_config(self, ds_names: list[str]) -> MagicMock:
        from dataeval_app.config.schemas.dataset import DatasetConfig

        datasets = [DatasetConfig(name=n, format="huggingface", path=f"./{n}", split="train") for n in ds_names]
        config = MagicMock()
        config.datasets = datasets
        config.preprocessors = None
        config.models = None
        config.selections = None
        return config

    def _mock_workflow(self) -> MagicMock:
        mock_result = MagicMock()
        mock_result.success = True
        mock_wf = MagicMock()
        mock_wf.params_schema = None
        mock_wf.execute.return_value = mock_result
        return mock_wf

    @patch("dataeval_app.dataset.load_dataset")
    def test_datasets_string_backward_compat(self, mock_load_ds: MagicMock):
        """datasets as a plain string works (single dataset)."""
        from dataeval_app.config.schemas.task import TaskConfig

        config = self._make_config(["ds"])
        task = TaskConfig(name="t", workflow="data-cleaning", datasets="ds")
        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with (
            patch("dataeval_app.workflow.get_workflow", return_value=mock_wf),
            patch("dataeval_app.workflow.orchestrator._write_output"),
        ):
            result = run_task(task, config)

        assert result.success
        mock_load_ds.assert_called_once()

    @patch("dataeval_app.dataset.load_dataset")
    def test_datasets_list_loads_multiple(self, mock_load_ds: MagicMock):
        """datasets as a list loads each dataset."""
        from dataeval_app.config.schemas.task import TaskConfig

        config = self._make_config(["ds_a", "ds_b"])
        task = TaskConfig(name="t", workflow="data-cleaning", datasets=["ds_a", "ds_b"])
        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with (
            patch("dataeval_app.workflow.get_workflow", return_value=mock_wf),
            patch("dataeval_app.workflow.orchestrator._write_output"),
        ):
            result = run_task(task, config)

        assert result.success
        assert mock_load_ds.call_count == 2

    @patch("dataeval_app.dataset.load_dataset")
    def test_shared_model_string_applies_to_all(self, mock_load_ds: MagicMock):
        """A model specified as string is shared across all datasets."""
        from dataeval_app.config.models import ModelConfig, OnnxExtractorConfig
        from dataeval_app.config.schemas.task import TaskConfig

        config = self._make_config(["ds_a", "ds_b"])
        config.models = [
            ModelConfig(name="resnet", extractor=OnnxExtractorConfig(model_path="/m.onnx", output_name="out")),
        ]
        task = TaskConfig(name="t", workflow="data-cleaning", datasets=["ds_a", "ds_b"], models="resnet")
        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with (
            patch("dataeval_app.workflow.get_workflow", return_value=mock_wf),
            patch("dataeval_app.workflow.orchestrator._write_output"),
        ):
            result = run_task(task, config)

        assert result.success
        context = mock_wf.execute.call_args[0][0]
        # Both datasets should have the same extractor
        for dc in context.dataset_contexts.values():
            assert dc.extractor is not None
            assert dc.extractor.model_path == "/m.onnx"

    @patch("dataeval_app.dataset.load_dataset")
    def test_per_dataset_model_mapping(self, mock_load_ds: MagicMock):
        """Per-dataset model mapping gives each dataset its own extractor."""
        from dataeval_app.config.models import FlattenExtractorConfig, ModelConfig, OnnxExtractorConfig
        from dataeval_app.config.schemas.task import TaskConfig

        config = self._make_config(["ds_a", "ds_b"])
        config.models = [
            ModelConfig(name="onnx_m", extractor=OnnxExtractorConfig(model_path="/m.onnx", output_name="out")),
            ModelConfig(name="flat_m", extractor=FlattenExtractorConfig()),
        ]
        task = TaskConfig(
            name="t",
            workflow="data-cleaning",
            datasets=["ds_a", "ds_b"],
            models={"ds_a": "onnx_m", "ds_b": "flat_m"},
        )
        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with (
            patch("dataeval_app.workflow.get_workflow", return_value=mock_wf),
            patch("dataeval_app.workflow.orchestrator._write_output"),
        ):
            result = run_task(task, config)

        assert result.success
        context = mock_wf.execute.call_args[0][0]
        assert context.dataset_contexts["ds_a"].extractor is not None
        assert context.dataset_contexts["ds_a"].extractor.model_path == "/m.onnx"
        assert context.dataset_contexts["ds_b"].extractor is not None

    @patch("dataeval_app.dataset.load_dataset")
    def test_mapping_missing_key_gives_none(self, mock_load_ds: MagicMock):
        """A mapping that omits a dataset key means that dataset gets None."""
        from dataeval_app.config.models import ModelConfig, OnnxExtractorConfig
        from dataeval_app.config.schemas.task import TaskConfig

        config = self._make_config(["ds_a", "ds_b"])
        config.models = [
            ModelConfig(name="m", extractor=OnnxExtractorConfig(model_path="/m.onnx", output_name="out")),
        ]
        # Only ds_a has a model; ds_b is absent from the mapping
        task = TaskConfig(
            name="t",
            workflow="data-cleaning",
            datasets=["ds_a", "ds_b"],
            models={"ds_a": "m"},
        )
        mock_load_ds.return_value = MagicMock()
        mock_wf = self._mock_workflow()

        with (
            patch("dataeval_app.workflow.get_workflow", return_value=mock_wf),
            patch("dataeval_app.workflow.orchestrator._write_output"),
        ):
            result = run_task(task, config)

        assert result.success
        context = mock_wf.execute.call_args[0][0]
        assert context.dataset_contexts["ds_a"].extractor is not None
        assert context.dataset_contexts["ds_b"].extractor is None

    def test_mapping_unknown_dataset_raises(self):
        """A mapping referencing an unknown dataset raises ValueError."""
        from dataeval_app.config.schemas.task import TaskConfig

        config = self._make_config(["ds_a"])
        task = TaskConfig(
            name="t",
            workflow="data-cleaning",
            datasets=["ds_a"],
            models={"ds_a": "m", "ds_unknown": "m"},
        )

        with pytest.raises(ValueError, match="unknown datasets"):
            run_task(task, config)

    @patch("dataeval_app.dataset.load_dataset")
    @patch("dataeval_app.preprocessing.build_preprocessing")
    def test_per_dataset_preprocessor_mapping(self, mock_build_pre: MagicMock, mock_load_ds: MagicMock):
        """Per-dataset preprocessor mapping works."""
        from dataeval_app.config.schemas.preprocessor import PreprocessorConfig
        from dataeval_app.config.schemas.task import TaskConfig
        from dataeval_app.preprocessing import PreprocessingStep

        config = self._make_config(["ds_a", "ds_b"])
        config.preprocessors = [
            PreprocessorConfig(name="pre_a", steps=[PreprocessingStep(step="ToTensor")]),
            PreprocessorConfig(name="pre_b", steps=[PreprocessingStep(step="Resize", params={"size": [224, 224]})]),
        ]
        task = TaskConfig(
            name="t",
            workflow="data-cleaning",
            datasets=["ds_a", "ds_b"],
            preprocessors={"ds_a": "pre_a", "ds_b": "pre_b"},
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
        assert mock_build_pre.call_count == 2

    @patch("dataeval_app.dataset.load_dataset")
    def test_single_dataset_context_fields(self, mock_load_ds: MagicMock):
        """Single-dataset WorkflowContext populates dataset_contexts correctly."""
        from dataeval_app.config.schemas.task import TaskConfig

        config = self._make_config(["ds"])
        task = TaskConfig(name="t", workflow="data-cleaning", datasets="ds")
        mock_dataset = MagicMock()
        mock_load_ds.return_value = mock_dataset
        mock_wf = self._mock_workflow()

        with (
            patch("dataeval_app.workflow.get_workflow", return_value=mock_wf),
            patch("dataeval_app.workflow.orchestrator._write_output"),
        ):
            run_task(task, config)

        context = mock_wf.execute.call_args[0][0]
        # dataset_contexts should have exactly one entry with the dataset object
        assert len(context.dataset_contexts) == 1
        assert "ds" in context.dataset_contexts
        assert context.dataset_contexts["ds"].dataset is mock_dataset

    def test_write_output_multi_dataset_metadata(self, tmp_path: Path):
        """Multi-dataset writes comma-joined dataset_id in metadata."""
        task = _make_task(output_format="json")
        result = _make_result()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        _write_output(result, task, dataset_names=["ds_a", "ds_b"], output_dir=output_dir)

        metadata_file = output_dir / task.name / "metadata.json"
        assert metadata_file.exists()
        metadata = json.loads(metadata_file.read_text())
        assert metadata["dataset_id"] == "ds_a,ds_b"
