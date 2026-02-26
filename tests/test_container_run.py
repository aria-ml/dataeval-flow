"""Tests for container_run.py error logging."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import container_run


class TestContainerRunLogging:
    def _make_config(self, tasks=None):
        config = MagicMock()
        config.tasks = tasks or []
        config.logging = None
        return config

    @patch("dataeval_app.workflow.run_task")
    @patch("dataeval_app.config.load_config_folder")
    @patch.object(container_run, "get_output_path")
    @patch.dict(container_run.CONTAINER_MOUNTS, {"config": Path("/tmp")})  # noqa: S108
    def test_failed_task_logs_errors(
        self,
        mock_output: MagicMock,
        mock_load: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        mock_output.return_value = tmp_path
        task = MagicMock()
        task.name = "bad_task"

        mock_load.return_value = self._make_config(tasks=[task])

        result = MagicMock()
        result.success = False
        result.errors = ["some error"]
        mock_run.return_value = result

        exit_code = container_run.main()

        assert exit_code == 1
        assert any("some error" in r.message and r.levelno == logging.ERROR for r in caplog.records)

    @patch("dataeval_app.workflow.run_task")
    @patch("dataeval_app.config.load_config_folder")
    @patch.object(container_run, "get_output_path")
    @patch.dict(container_run.CONTAINER_MOUNTS, {"config": Path("/tmp")})  # noqa: S108
    def test_successful_task_logs_ok(
        self,
        mock_output: MagicMock,
        mock_load: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        mock_output.return_value = tmp_path
        task = MagicMock()
        task.name = "good_task"

        mock_load.return_value = self._make_config(tasks=[task])

        result = MagicMock()
        result.success = True
        result.report.return_value = tmp_path / "good_task" / "results.json"
        mock_run.return_value = result

        exit_code = container_run.main()

        assert exit_code == 0
        assert "good_task: OK" in caplog.text

    @patch("dataeval_app.workflow.run_task")
    @patch("dataeval_app.config.load_config_folder")
    @patch.object(container_run, "get_output_path")
    @patch.dict(container_run.CONTAINER_MOUNTS, {"config": Path("/tmp")})  # noqa: S108
    def test_log_file_written_on_failure(
        self,
        mock_output: MagicMock,
        mock_load: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ):
        mock_output.return_value = tmp_path
        task = MagicMock()
        task.name = "fail_task"

        mock_load.return_value = self._make_config(tasks=[task])

        result = MagicMock()
        result.success = False
        result.errors = ["disk full"]
        mock_run.return_value = result

        container_run.main()

        # Flush handlers so content is written
        for h in logging.getLogger().handlers:
            h.flush()

        log_file = tmp_path / "log.txt"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "disk full" in content
