"""Tests for _resolve_config in runner.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dataeval_flow.config._models import PipelineConfig
from dataeval_flow.runner import _resolve_config

pytestmark = pytest.mark.required


@pytest.fixture
def dummy_config() -> PipelineConfig:
    return PipelineConfig()


_LOADER = "dataeval_flow.config._loader"


class TestResolveConfig:
    def test_explicit_absolute_file(self, tmp_path: Path, dummy_config: PipelineConfig):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("tasks: []")

        with patch(f"{_LOADER}.load_config", return_value=dummy_config) as mock_load:
            result = _resolve_config(cfg_file, tmp_path)

        mock_load.assert_called_once_with(cfg_file)
        assert result is dummy_config

    def test_explicit_relative_file(self, tmp_path: Path, dummy_config: PipelineConfig):
        cfg_file = tmp_path / "sub" / "config.yaml"
        cfg_file.parent.mkdir()
        cfg_file.write_text("tasks: []")

        with patch(f"{_LOADER}.load_config", return_value=dummy_config) as mock_load:
            result = _resolve_config(Path("sub/config.yaml"), tmp_path)

        mock_load.assert_called_once_with(tmp_path / "sub" / "config.yaml")
        assert result is dummy_config

    def test_explicit_directory(self, tmp_path: Path, dummy_config: PipelineConfig):
        cfg_dir = tmp_path / "conf"
        cfg_dir.mkdir()

        with patch(f"{_LOADER}.load_config_folder", return_value=dummy_config) as mock_load:
            result = _resolve_config(cfg_dir, tmp_path)

        mock_load.assert_called_once_with(cfg_dir)
        assert result is dummy_config

    def test_none_config_uses_data_dir(self, tmp_path: Path, dummy_config: PipelineConfig):
        with patch(f"{_LOADER}.load_config_folder", return_value=dummy_config) as mock_load:
            result = _resolve_config(None, tmp_path)

        mock_load.assert_called_once_with(tmp_path)
        assert result is dummy_config

    def test_missing_path_raises(self, tmp_path: Path):
        missing = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="Config path not found"):
            _resolve_config(missing, tmp_path)

    def test_string_config_arg(self, tmp_path: Path, dummy_config: PipelineConfig):
        cfg_file = tmp_path / "my_config.yaml"
        cfg_file.write_text("tasks: []")

        with patch(f"{_LOADER}.load_config", return_value=dummy_config) as mock_load:
            result = _resolve_config("my_config.yaml", tmp_path)

        mock_load.assert_called_once_with(tmp_path / "my_config.yaml")
        assert result is dummy_config


class TestWriteEncodingDescriptor:
    """A run writes the descriptor its results were computed under, beside them."""

    @staticmethod
    def _record(digest: str, edges: list) -> dict:
        return {
            "encoding_digest": digest,
            "descriptor_version": 1,
            "factors": {
                "temp_c": {"encoding": {"kind": "bins", "edges": edges, "provenance": "edges", "method": None}}
            },
        }

    def test_writes_one_when_the_tasks_agree(self, tmp_path: Path):
        from dataeval_flow.runner import _write_encoding_descriptor

        record = self._record("abc", ["-inf", 0.0, "inf"])
        _write_encoding_descriptor({"a": record, "b": record}, tmp_path)

        import json

        written = json.loads((tmp_path / "encoding.json").read_text())
        assert written["factors"]["temp_c"]["edges"] == ["-inf", 0.0, "inf"]
        assert written["version"] == 1

    def test_writes_nothing_when_the_tasks_disagree(self, tmp_path: Path, caplog):
        """Writing one of them would hand somebody a policy nobody chose."""
        from dataeval_flow.runner import _write_encoding_descriptor

        _write_encoding_descriptor(
            {"a": self._record("abc", ["-inf", 0.0, "inf"]), "b": self._record("def", ["-inf", 5.0, "inf"])},
            tmp_path,
        )

        assert not (tmp_path / "encoding.json").exists()
        assert "dataeval-flow encoding" in caplog.text

    def test_writes_nothing_when_no_task_built_metadata(self, tmp_path: Path):
        from dataeval_flow.runner import _write_encoding_descriptor

        _write_encoding_descriptor({}, tmp_path)
        assert not (tmp_path / "encoding.json").exists()

    def test_a_record_with_no_encodings_is_skipped_quietly(self, tmp_path: Path):
        """Never fatal: result.json already carries every record it is built from."""
        from dataeval_flow.runner import _write_encoding_descriptor

        _write_encoding_descriptor({"a": {"factors": {}}}, tmp_path)
        assert not (tmp_path / "encoding.json").exists()


# ---------------------------------------------------------------------------
# run() — task selection, result pairing, and the health gate
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, *, disable: str | None = None) -> Path:
    """A two-task config, optionally with one task disabled."""
    lines = [
        "datasets:",
        "  - name: ds",
        "    format: huggingface",
        "    path: ./d",
        "    task: image_classification",
        "sources:",
        "  - name: src",
        "    dataset: ds",
        "workflows:",
        "  - name: wf",
        "    type: data-cleaning",
        "    outlier_method: modzscore",
        "    outlier_flags: [dimension]",
        "tasks:",
    ]
    for name in ("task_a", "task_b"):
        lines += [f"  - name: {name}", "    workflow: wf", "    sources: src"]
        if name == disable:
            lines.append("    enabled: false")
    path = tmp_path / "config.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def _fake_result(*, warnings: int = 0):
    """A stand-in workflow result with a controllable warning count."""
    from unittest.mock import MagicMock

    result = MagicMock()
    result.success = True
    result.report.return_value = "report"
    result.to_dict.return_value = {"metadata": {}}
    result.metadata = MagicMock(metadata_binning=None)
    result.warning_count = warnings
    return result


class TestRunTaskPairing:
    """A disabled task must not misalign results against the tasks that produced them."""

    def test_disabled_task_does_not_break_the_run(self, tmp_path: Path):
        """Regression: run() paired results against every task, not the executed ones."""
        import dataeval_flow.workflow.orchestrator as orch
        from dataeval_flow.runner import run

        config = _write_config(tmp_path, disable="task_b")
        with patch.object(orch, "_run_single_task", return_value=_fake_result()):
            assert run(config, tmp_path / "out", data_dir=tmp_path) == 0

        import json

        merged = json.loads((tmp_path / "out" / "results" / "result.json").read_text())
        assert list(merged) == ["task_a"]

    def test_results_are_keyed_by_the_task_that_produced_them(self, tmp_path: Path):
        """A result names its workflow type, so keying has to come from the selection."""
        import dataeval_flow.workflow.orchestrator as orch
        from dataeval_flow.runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_fake_result()):
            assert run(config, tmp_path / "out", data_dir=tmp_path) == 0

        import json

        merged = json.loads((tmp_path / "out" / "results" / "result.json").read_text())
        assert list(merged) == ["task_a", "task_b"]


class TestRunTaskSelection:
    def test_names_select_a_subset(self, tmp_path: Path):
        import dataeval_flow.workflow.orchestrator as orch
        from dataeval_flow.runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_fake_result()):
            assert run(config, tmp_path / "out", data_dir=tmp_path, tasks=["task_b"]) == 0

        import json

        merged = json.loads((tmp_path / "out" / "results" / "result.json").read_text())
        assert list(merged) == ["task_b"]

    def test_naming_a_disabled_task_runs_it(self, tmp_path: Path):
        import dataeval_flow.workflow.orchestrator as orch
        from dataeval_flow.runner import run

        config = _write_config(tmp_path, disable="task_b")
        with patch.object(orch, "_run_single_task", return_value=_fake_result()):
            assert run(config, tmp_path / "out", data_dir=tmp_path, tasks="task_b") == 0

        import json

        merged = json.loads((tmp_path / "out" / "results" / "result.json").read_text())
        assert list(merged) == ["task_b"]

    def test_unknown_task_name_raises(self, tmp_path: Path):
        from dataeval_flow.runner import run

        config = _write_config(tmp_path)
        with pytest.raises(ValueError, match="Unknown task: 'nope'"):
            run(config, tmp_path / "out", data_dir=tmp_path, tasks="nope")


class TestFailOnWarning:
    def test_warnings_are_not_fatal_by_default(self, tmp_path: Path, caplog):
        """A warning is a prompt to look, so it is reported without failing the run."""
        import dataeval_flow.workflow.orchestrator as orch
        from dataeval_flow.runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_fake_result(warnings=2)):
            assert run(config, tmp_path / "out", data_dir=tmp_path) == 0

        assert "Health warnings raised by: task_a, task_b" in caplog.text

    def test_flag_makes_warnings_fatal(self, tmp_path: Path):
        import dataeval_flow.workflow.orchestrator as orch
        from dataeval_flow.runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_fake_result(warnings=1)):
            assert run(config, tmp_path / "out", data_dir=tmp_path, fail_on_warning=True) == 1

    def test_flag_is_a_no_op_without_warnings(self, tmp_path: Path):
        import dataeval_flow.workflow.orchestrator as orch
        from dataeval_flow.runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_fake_result(warnings=0)):
            assert run(config, tmp_path / "out", data_dir=tmp_path, fail_on_warning=True) == 0

    def test_results_are_still_written_when_warnings_are_fatal(self, tmp_path: Path):
        """The gate decides the exit code, not whether the run's artifacts survive."""
        import dataeval_flow.workflow.orchestrator as orch
        from dataeval_flow.runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_fake_result(warnings=1)):
            run(config, tmp_path / "out", data_dir=tmp_path, fail_on_warning=True)

        assert (tmp_path / "out" / "results" / "result.json").exists()
        assert (tmp_path / "out" / "results" / "result.txt").exists()
