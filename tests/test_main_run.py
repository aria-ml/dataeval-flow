"""Tests for __main__.py — _run_tasks path."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestRunTasks:
    @patch("dataeval_app.workflow.run_task")
    @patch("dataeval_app.config.load_config_folder")
    def test_no_tasks_exits_zero(self, mock_load: MagicMock, mock_run: MagicMock):  # noqa: ARG002
        from dataeval_app.__main__ import _run_tasks

        config = MagicMock()
        config.tasks = []
        mock_load.return_value = config

        with pytest.raises(SystemExit) as exc_info:
            _run_tasks(Path("/fake/config"))

        assert exc_info.value.code == 0

    @patch("dataeval_app.workflow.run_task")
    @patch("dataeval_app.config.load_config_folder")
    def test_successful_tasks(self, mock_load: MagicMock, mock_run: MagicMock, capsys: pytest.CaptureFixture):
        from dataeval_app.__main__ import _run_tasks

        task1 = MagicMock()
        task1.name = "task1"
        task1.workflow = "data-cleaning"
        task2 = MagicMock()
        task2.name = "task2"
        task2.workflow = "data-cleaning"

        config = MagicMock()
        config.tasks = [task1, task2]
        mock_load.return_value = config

        result1 = MagicMock()
        result1.success = True
        result1.name = "task1"
        result1.data = MagicMock()
        result1.data.report = MagicMock()
        result1.data.report.summary = "All clean"

        result2 = MagicMock()
        result2.success = True
        result2.name = "task2"
        result2.data = MagicMock()
        result2.data.report = MagicMock()
        result2.data.report.summary = "Done"

        mock_run.side_effect = [result1, result2]

        with pytest.raises(SystemExit) as exc_info:
            _run_tasks(Path("/fake/config"))

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "2/2 succeeded" in out

    @patch("dataeval_app.workflow.run_task")
    @patch("dataeval_app.config.load_config_folder")
    def test_failed_task_exits_one(self, mock_load: MagicMock, mock_run: MagicMock, capsys: pytest.CaptureFixture):
        from dataeval_app.__main__ import _run_tasks

        task1 = MagicMock()
        task1.name = "task1"
        task1.workflow = "data-cleaning"

        config = MagicMock()
        config.tasks = [task1]
        mock_load.return_value = config

        result = MagicMock()
        result.success = False
        result.errors = ["Something went wrong"]
        mock_run.return_value = result

        with pytest.raises(SystemExit) as exc_info:
            _run_tasks(Path("/fake/config"))

        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "FAILED" in out

    @patch("dataeval_app.workflow.run_task")
    @patch("dataeval_app.config.load_config_folder")
    def test_result_without_report(self, mock_load: MagicMock, mock_run: MagicMock, capsys: pytest.CaptureFixture):
        """Result.data without .report attribute still prints OK."""
        from dataeval_app.__main__ import _run_tasks

        task = MagicMock()
        task.name = "task1"
        task.workflow = "data-cleaning"

        config = MagicMock()
        config.tasks = [task]
        mock_load.return_value = config

        result = MagicMock()
        result.success = True
        result.name = "task1"
        result.data = MagicMock(spec=[])  # No .report attribute

        mock_run.return_value = result

        with pytest.raises(SystemExit) as exc_info:
            _run_tasks(Path("/fake/config"))

        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "OK" in out

    @patch("dataeval_app.config.load_config_folder")
    def test_uses_default_config_path(self, mock_load: MagicMock):
        from dataeval_app.__main__ import _run_tasks

        config = MagicMock()
        config.tasks = []
        mock_load.return_value = config

        with pytest.raises(SystemExit):
            _run_tasks(None)

        mock_load.assert_called_once_with(None)


class TestParseArgs:
    def test_no_args_defaults_none(self):
        from dataeval_app.__main__ import parse_args

        with patch("sys.argv", ["dataeval_app"]):
            args = parse_args()
        assert args.config is None
        assert args.output is None

    def test_with_config(self):
        from dataeval_app.__main__ import parse_args

        with patch("sys.argv", ["dataeval_app", "--config", "/my/config"]):
            args = parse_args()
        assert args.config == Path("/my/config")

    def test_with_output(self):
        from dataeval_app.__main__ import parse_args

        with patch("sys.argv", ["dataeval_app", "--output", "/my/output"]):
            args = parse_args()
        assert args.output == Path("/my/output")


class TestMain:
    @patch("dataeval_app.__main__._run_tasks")
    @patch("dataeval_app.__main__.parse_args")
    def test_main_calls_run_tasks(self, mock_parse: MagicMock, mock_run_tasks: MagicMock):
        from dataeval_app.__main__ import main

        args = MagicMock()
        args.config = Path("/cfg")
        args.output = Path("/out")
        mock_parse.return_value = args

        main()
        mock_run_tasks.assert_called_once_with(Path("/cfg"), Path("/out"))

    @patch("dataeval_app.__main__.parse_args")
    def test_main_handles_errors(self, mock_parse: MagicMock, capsys: pytest.CaptureFixture):
        from dataeval_app.__main__ import main

        args = MagicMock()
        args.config = None
        args.output = None
        mock_parse.return_value = args

        with (
            patch("dataeval_app.__main__._run_tasks", side_effect=FileNotFoundError("not found")),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()

        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "ERROR" in out
