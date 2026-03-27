"""Tests for runner.py and __main__.py."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestRunTasks:
    @patch("dataeval_flow.workflow.run_tasks")
    @patch("dataeval_flow.runner._resolve_config")
    def test_no_tasks_exits_zero(self, mock_load: MagicMock, mock_run: MagicMock):
        from dataeval_flow.runner import run

        config = MagicMock()
        config.tasks = []
        config.logging = None
        mock_load.return_value = config

        assert run(Path("/fake/config"), Path("/fake/output")) == 0

    @patch("dataeval_flow.workflow.run_tasks")
    @patch("dataeval_flow.runner._resolve_config")
    def test_successful_tasks(self, mock_load: MagicMock, mock_run: MagicMock, tmp_path: Path):
        from dataeval_flow.runner import run

        task1 = MagicMock()
        task1.name = "task1"
        task1.workflow = "data-cleaning"
        task1.sources = "src1"
        task2 = MagicMock()
        task2.name = "task2"
        task2.workflow = "data-cleaning"
        task2.sources = "src2"

        config = MagicMock()
        config.tasks = [task1, task2]
        config.logging = None
        mock_load.return_value = config

        result1 = MagicMock()
        result1.success = True
        result1.name = "task1"
        result1.report.return_value = "text report 1"
        result1.to_dict.return_value = {"metadata": {}, "score": 0.9}

        result2 = MagicMock()
        result2.success = True
        result2.name = "task2"
        result2.report.return_value = "text report 2"
        result2.to_dict.return_value = {"metadata": {}, "score": 0.8}

        mock_run.return_value = [result1, result2]

        assert run(Path("/fake/config"), tmp_path) == 0
        # Single merged result files
        assert (tmp_path / "result.json").exists()
        assert (tmp_path / "result.txt").exists()
        merged = json.loads((tmp_path / "result.json").read_text())
        assert "task1" in merged
        assert "task2" in merged

    @patch("dataeval_flow.workflow.run_tasks")
    @patch("dataeval_flow.runner._resolve_config")
    def test_no_output_dir_no_files(self, mock_load: MagicMock, mock_run: MagicMock):
        """When output_dir is None, no file artifacts are created."""
        from dataeval_flow.runner import run

        task = MagicMock()
        task.name = "task1"
        task.workflow = "data-cleaning"
        task.sources = "src1"

        config = MagicMock()
        config.tasks = [task]
        config.logging = None
        mock_load.return_value = config

        result = MagicMock()
        result.success = True
        result.name = "task1"

        result.report.return_value = "text report"
        result.to_dict.return_value = {"metadata": {}}

        mock_run.return_value = [result]

        assert run(Path("/fake/config"), None) == 0
        # No files should be written anywhere

    @patch("dataeval_flow.workflow.run_tasks")
    @patch("dataeval_flow.runner._resolve_config")
    def test_verbosity_1_detailed_report(
        self, mock_load: MagicMock, mock_run: MagicMock, capsys: pytest.CaptureFixture
    ):
        """At verbosity >= 1, full detailed text report is printed."""
        from dataeval_flow.runner import run

        task = MagicMock()
        task.name = "task1"
        task.workflow = "data-cleaning"
        task.sources = "src1"

        config = MagicMock()
        config.tasks = [task]
        config.logging = None
        mock_load.return_value = config

        result = MagicMock()
        result.success = True
        result.name = "task1"

        result.report.return_value = "== Full Report =="

        mock_run.return_value = [result]

        run(Path("/fake/config"), None, verbosity=1)
        captured = capsys.readouterr()
        assert "== Full Report ==" in captured.out
        # Console call uses detailed=True at verbosity >= 1
        result.report.assert_any_call(detailed=True)

    @patch("dataeval_flow.workflow.run_tasks")
    @patch("dataeval_flow.runner._resolve_config")
    def test_verbosity_0_summary_report(self, mock_load: MagicMock, mock_run: MagicMock, capsys: pytest.CaptureFixture):
        """At verbosity 0, summary-only text report is printed."""
        from dataeval_flow.runner import run

        task = MagicMock()
        task.name = "task1"
        task.workflow = "data-cleaning"
        task.sources = "src1"

        config = MagicMock()
        config.tasks = [task]
        config.logging = None
        mock_load.return_value = config

        result = MagicMock()
        result.success = True
        result.name = "task1"

        result.report.return_value = "== Summary =="

        mock_run.return_value = [result]

        run(Path("/fake/config"), None, verbosity=0)
        captured = capsys.readouterr()
        assert "== Summary ==" in captured.out
        # Console call uses detailed=False at verbosity 0
        result.report.assert_any_call(detailed=False)

    @patch("dataeval_flow.workflow.run_tasks")
    @patch("dataeval_flow.runner._resolve_config")
    def test_failed_task_exits_one(self, mock_load: MagicMock, mock_run: MagicMock, caplog: pytest.LogCaptureFixture):
        from dataeval_flow.runner import run

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
        mock_run.return_value = [result]

        assert run(Path("/fake/config"), Path("/fake/output")) == 1
        assert "FAILED" in caplog.text
        assert "Something went wrong" in caplog.text

    @patch("dataeval_flow.workflow.run_tasks")
    @patch("dataeval_flow.runner._resolve_config")
    def test_result_without_report(self, mock_load: MagicMock, mock_run: MagicMock, tmp_path: Path):
        """Result.data without .report attribute still works."""
        from dataeval_flow.runner import run

        task = MagicMock()
        task.name = "task1"
        task.workflow = "data-cleaning"
        task.sources = "src1"

        config = MagicMock()
        config.tasks = [task]
        config.logging = None
        mock_load.return_value = config

        result = MagicMock()
        result.success = True
        result.name = "task1"

        result.data = MagicMock(spec=[])  # No .report attribute
        result.report.side_effect = [
            "summary report",  # console print (detailed=False)
            "full text report",  # written to .txt file (detailed=True)
        ]
        result.to_dict.return_value = {"metadata": {}, "data": "test"}

        mock_run.return_value = [result]

        assert run(Path("/fake/config"), tmp_path) == 0
        assert (tmp_path / "result.json").exists()
        merged = json.loads((tmp_path / "result.json").read_text())
        assert "task1" in merged

    @patch("dataeval_flow.runner._resolve_config")
    def test_uses_default_config_path(self, mock_load: MagicMock):
        from dataeval_flow.runner import run

        config = MagicMock()
        config.tasks = []
        config.logging = None
        mock_load.return_value = config

        run(None, Path("/fake/output"))

        # _resolve_config is called with config_arg=None and a resolved data_dir
        mock_load.assert_called_once()
        args = mock_load.call_args[0]
        assert args[0] is None  # config_arg
        assert args[1] is not None  # data_dir resolved

    @patch("dataeval_flow._logging.configure_log_levels")
    @patch("dataeval_flow.runner._resolve_config")
    def test_logging_config_applies_levels(self, mock_load: MagicMock, mock_configure: MagicMock):
        """config.logging triggers configure_log_levels after config loads."""
        from dataeval_flow.runner import run

        config = MagicMock()
        config.tasks = []
        config.logging.app_level = "WARNING"
        config.logging.lib_level = "ERROR"
        mock_load.return_value = config

        run(Path("/fake/config"), Path("/fake/output"))

        mock_configure.assert_called_once_with("WARNING", "ERROR")


class TestParseArgs:
    def test_no_args_succeeds(self):
        """No arguments defaults to headless execution."""
        from dataeval_flow.__main__ import parse_args

        with patch("sys.argv", ["dataeval_flow"]):
            args = parse_args()
        assert args.command is None
        assert args.output is None
        assert args.verbose == 0

    def test_with_config_and_output(self):
        from dataeval_flow.__main__ import parse_args

        with patch("sys.argv", ["dataeval_flow", "--config", "/my/config", "--output", "/my/output"]):
            args = parse_args()
        assert args.config == Path("/my/config")
        assert args.output == Path("/my/output")

    def test_with_data(self):
        from dataeval_flow.__main__ import parse_args

        with patch("sys.argv", ["dataeval_flow", "--output", "/out", "--data", "/my/data"]):
            args = parse_args()
        assert args.data == Path("/my/data")

    def test_verbose_flags(self):
        from dataeval_flow.__main__ import parse_args

        with patch("sys.argv", ["dataeval_flow", "-v"]):
            args = parse_args()
        assert args.verbose == 1

        with patch("sys.argv", ["dataeval_flow", "-vvv"]):
            args = parse_args()
        assert args.verbose == 3

    def test_with_cache(self):
        from dataeval_flow.__main__ import parse_args

        with patch("sys.argv", ["dataeval_flow", "--cache", "/my/cache"]):
            args = parse_args()
        assert args.cache == Path("/my/cache")


class TestCacheDir:
    @patch("dataeval_flow.workflow.run_tasks")
    @patch("dataeval_flow.runner._resolve_config")
    def test_cache_dir_forwarded_to_run_tasks(self, mock_load: MagicMock, mock_run: MagicMock):
        """Global --cache is forwarded to run_tasks()."""
        from dataeval_flow.runner import run

        task = MagicMock()
        task.name = "task1"

        config = MagicMock()
        config.tasks = [task]
        config.logging = None
        mock_load.return_value = config

        r1 = MagicMock(success=False, errors=["e"])
        mock_run.return_value = [r1]

        run(Path("/fake/config"), cache_dir=Path("/global/cache"))

        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["cache_dir"] == Path("/global/cache")

    @patch("dataeval_flow.workflow.run_tasks")
    @patch("dataeval_flow.runner._resolve_config")
    def test_no_cache_forwards_none(self, mock_load: MagicMock, mock_run: MagicMock):
        """Without --cache, run_tasks() receives cache_dir=None."""
        from dataeval_flow.runner import run

        task = MagicMock()
        task.name = "task1"

        config = MagicMock()
        config.tasks = [task]
        config.logging = None
        mock_load.return_value = config

        r1 = MagicMock(success=False, errors=["e"])
        mock_run.return_value = [r1]

        run(Path("/fake/config"))

        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["cache_dir"] is None


class TestMain:
    @patch("dataeval_flow.runner.run")
    @patch("dataeval_flow.__main__.parse_args")
    def test_main_calls_run_tasks(self, mock_parse: MagicMock, mock_run_tasks: MagicMock):
        from dataeval_flow.__main__ import main

        args = MagicMock()
        args.command = None
        args.config = Path("/cfg")
        args.output = Path("/out")
        args.data = None
        args.verbose = 0
        args.cache = None
        mock_parse.return_value = args
        mock_run_tasks.return_value = 0

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_run_tasks.assert_called_once_with(Path("/cfg"), Path("/out"), data_dir=None, verbosity=0, cache_dir=None)

    @patch("dataeval_flow.runner.run")
    @patch("dataeval_flow.__main__.parse_args")
    def test_main_handles_errors(self, mock_parse: MagicMock, mock_run_tasks: MagicMock, capsys: pytest.CaptureFixture):
        from dataeval_flow.__main__ import main

        args = MagicMock()
        args.command = None
        args.config = None
        args.output = None
        args.data = None
        args.verbose = 0
        args.cache = None
        mock_parse.return_value = args
        mock_run_tasks.side_effect = FileNotFoundError("not found")

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "ERROR" in out
