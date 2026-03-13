"""Tests for runner.py and __main__.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestRunTasks:
    @patch("dataeval_app.workflow.run_task")
    @patch("dataeval_app.config.load_config_folder")
    def test_no_tasks_exits_zero(self, mock_load: MagicMock, mock_run: MagicMock):  # noqa: ARG002
        from dataeval_app.runner import run_all_tasks

        config = MagicMock()
        config.tasks = []
        config.logging = None
        mock_load.return_value = config

        assert run_all_tasks(Path("/fake/config"), Path("/fake/output")) == 0

    @patch("dataeval_app.workflow.run_task")
    @patch("dataeval_app.config.load_config_folder")
    def test_successful_tasks(
        self, mock_load: MagicMock, mock_run: MagicMock, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ):
        from dataeval_app.runner import run_all_tasks

        task1 = MagicMock()
        task1.name = "task1"
        task1.workflow = "data-cleaning"
        task2 = MagicMock()
        task2.name = "task2"
        task2.workflow = "data-cleaning"

        config = MagicMock()
        config.tasks = [task1, task2]
        config.logging = None
        mock_load.return_value = config

        result1 = MagicMock()
        result1.success = True
        result1.name = "task1"
        result1.data = MagicMock()
        result1.data.report = MagicMock()
        result1.data.report.summary = "All clean"
        result1.report.return_value = "text report 1"

        result2 = MagicMock()
        result2.success = True
        result2.name = "task2"
        result2.data = MagicMock()
        result2.data.report = MagicMock()
        result2.data.report.summary = "Done"
        result2.report.return_value = "text report 2"

        mock_run.side_effect = [result1, result2]

        assert run_all_tasks(Path("/fake/config"), tmp_path) == 0
        assert "2/2 succeeded" in caplog.text
        assert (tmp_path / "task1" / "report.txt").exists()
        assert (tmp_path / "task2" / "report.txt").exists()

    @patch("dataeval_app.workflow.run_task")
    @patch("dataeval_app.config.load_config_folder")
    def test_failed_task_exits_one(self, mock_load: MagicMock, mock_run: MagicMock, caplog: pytest.LogCaptureFixture):
        from dataeval_app.runner import run_all_tasks

        task1 = MagicMock()
        task1.name = "task1"
        task1.workflow = "data-cleaning"

        config = MagicMock()
        config.tasks = [task1]
        config.logging = None
        mock_load.return_value = config

        result = MagicMock()
        result.success = False
        result.errors = ["Something went wrong"]
        mock_run.return_value = result

        assert run_all_tasks(Path("/fake/config"), Path("/fake/output")) == 1
        assert "FAILED" in caplog.text
        assert "Something went wrong" in caplog.text

    @patch("dataeval_app.workflow.run_task")
    @patch("dataeval_app.config.load_config_folder")
    def test_result_without_report(
        self, mock_load: MagicMock, mock_run: MagicMock, caplog: pytest.LogCaptureFixture, tmp_path: Path
    ):
        """Result.data without .report attribute still logs OK."""
        from dataeval_app.runner import run_all_tasks

        task = MagicMock()
        task.name = "task1"
        task.workflow = "data-cleaning"

        config = MagicMock()
        config.tasks = [task]
        config.logging = None
        mock_load.return_value = config

        result = MagicMock()
        result.success = True
        result.name = "task1"
        result.data = MagicMock(spec=[])  # No .report attribute
        result.report.side_effect = [tmp_path / "task1" / "results.json", "text report"]

        mock_run.return_value = result

        assert run_all_tasks(Path("/fake/config"), tmp_path) == 0
        assert "OK" in caplog.text
        assert (tmp_path / "task1" / "report.txt").exists()

    @patch("dataeval_app.config.load_config_folder")
    def test_uses_default_config_path(self, mock_load: MagicMock):
        from dataeval_app.runner import run_all_tasks

        config = MagicMock()
        config.tasks = []
        config.logging = None
        mock_load.return_value = config

        run_all_tasks(None, Path("/fake/output"))

        mock_load.assert_called_once_with(None)

    @patch("dataeval_app._logging.configure_log_levels")
    @patch("dataeval_app.config.load_config_folder")
    def test_logging_config_applies_levels(self, mock_load: MagicMock, mock_configure: MagicMock):
        """config.logging triggers configure_log_levels after config loads."""
        from dataeval_app.runner import run_all_tasks

        config = MagicMock()
        config.tasks = []
        config.logging.app_level = "WARNING"
        config.logging.lib_level = "ERROR"
        mock_load.return_value = config

        run_all_tasks(Path("/fake/config"), Path("/fake/output"))

        mock_configure.assert_called_once_with("WARNING", "ERROR")


class TestParseArgs:
    def test_no_args_exits_error(self):
        """--output is required; omitting it causes argparse to exit with error."""
        from dataeval_app.__main__ import parse_args

        with patch("sys.argv", ["dataeval_app"]), pytest.raises(SystemExit) as exc_info:
            parse_args()
        assert exc_info.value.code == 2  # argparse exits with code 2 for missing required args

    def test_with_config(self):
        from dataeval_app.__main__ import parse_args

        with patch("sys.argv", ["dataeval_app", "--config", "/my/config", "--output", "/my/output"]):
            args = parse_args()
        assert args.config == Path("/my/config")
        assert args.output == Path("/my/output")

    def test_with_output(self):
        from dataeval_app.__main__ import parse_args

        with patch("sys.argv", ["dataeval_app", "--output", "/my/output"]):
            args = parse_args()
        assert args.output == Path("/my/output")


class TestMain:
    @patch("dataeval_app.runner.run_all_tasks")
    @patch("dataeval_app.__main__.parse_args")
    def test_main_calls_run_tasks(self, mock_parse: MagicMock, mock_run_tasks: MagicMock):
        from dataeval_app.__main__ import main

        args = MagicMock()
        args.config = Path("/cfg")
        args.output = Path("/out")
        mock_parse.return_value = args
        mock_run_tasks.return_value = 0

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_run_tasks.assert_called_once_with(Path("/cfg"), Path("/out"))

    @patch("dataeval_app.runner.run_all_tasks")
    @patch("dataeval_app.__main__.parse_args")
    def test_main_handles_errors(self, mock_parse: MagicMock, mock_run_tasks: MagicMock, capsys: pytest.CaptureFixture):
        from dataeval_app.__main__ import main

        args = MagicMock()
        args.config = None
        args.output = None
        mock_parse.return_value = args
        mock_run_tasks.side_effect = FileNotFoundError("not found")

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "ERROR" in out
