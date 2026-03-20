"""Tests for container_run.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import container_run


class TestContainerRunMain:
    @patch("dataeval_flow.runner.run")
    def test_failed_task_returns_one(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = 1

        with (
            patch("sys.argv", ["container_run.py", "--data", str(tmp_path)]),
            patch.object(container_run, "_DEFAULT_CACHE", tmp_path / "no_cache"),
        ):
            exit_code = container_run.main()

        assert exit_code == 1
        mock_run.assert_called_once_with(None, Path("/output"), data_dir=tmp_path, verbosity=0, cache_dir=None)

    @patch("dataeval_flow.runner.run")
    def test_successful_task_returns_zero(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = 0

        with (
            patch("sys.argv", ["container_run.py", "--data", str(tmp_path)]),
            patch.object(container_run, "_DEFAULT_CACHE", tmp_path / "no_cache"),
        ):
            exit_code = container_run.main()

        assert exit_code == 0
        mock_run.assert_called_once_with(None, Path("/output"), data_dir=tmp_path, verbosity=0, cache_dir=None)

    def test_missing_data_mount_returns_one(self, capsys: pytest.CaptureFixture):
        with patch("sys.argv", ["container_run.py", "--data", "/nonexistent_path_xyz"]):
            exit_code = container_run.main()

        assert exit_code == 1
        out = capsys.readouterr().out
        assert "ERROR" in out

    def test_not_mounted_marker_returns_one(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        (tmp_path / ".not_mounted").touch()

        with patch("sys.argv", ["container_run.py", "--data", str(tmp_path)]):
            exit_code = container_run.main()

        assert exit_code == 1
        out = capsys.readouterr().out
        assert "ERROR" in out

    @patch("dataeval_flow.runner.run")
    def test_config_arg_passed_through(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = 0

        with (
            patch("sys.argv", ["container_run.py", "--data", str(tmp_path), "--config", "my_config/"]),
            patch.object(container_run, "_DEFAULT_CACHE", tmp_path / "no_cache"),
        ):
            exit_code = container_run.main()

        assert exit_code == 0
        mock_run.assert_called_once_with(
            Path("my_config/"), Path("/output"), data_dir=tmp_path, verbosity=0, cache_dir=None
        )

    def test_import_error_returns_one(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        with (
            patch("sys.argv", ["container_run.py", "--data", str(tmp_path)]),
            patch.dict("sys.modules", {"dataeval_flow.runner": None}),
        ):
            exit_code = container_run.main()

        assert exit_code == 1
        out = capsys.readouterr().out
        assert "ERROR" in out

    @patch("dataeval_flow.runner.run")
    def test_verbosity_forwarded(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = 0

        with (
            patch("sys.argv", ["container_run.py", "--data", str(tmp_path), "-vv"]),
            patch.object(container_run, "_DEFAULT_CACHE", tmp_path / "no_cache"),
        ):
            exit_code = container_run.main()

        assert exit_code == 0
        mock_run.assert_called_once_with(None, Path("/output"), data_dir=tmp_path, verbosity=2, cache_dir=None)

    @patch("dataeval_flow.runner.run")
    def test_cache_auto_detected_when_mounted(self, mock_run: MagicMock, tmp_path: Path):
        """When /cache is a writable directory and no --cache given, it's used automatically."""
        mock_run.return_value = 0
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        with (
            patch("sys.argv", ["container_run.py", "--data", str(tmp_path)]),
            patch.object(container_run, "_DEFAULT_CACHE", cache_dir),
        ):
            exit_code = container_run.main()

        assert exit_code == 0
        mock_run.assert_called_once_with(None, Path("/output"), data_dir=tmp_path, verbosity=0, cache_dir=cache_dir)

    @patch("dataeval_flow.runner.run")
    def test_explicit_cache_overrides_default(self, mock_run: MagicMock, tmp_path: Path):
        mock_run.return_value = 0
        custom_cache = tmp_path / "my_cache"

        with (
            patch("sys.argv", ["container_run.py", "--data", str(tmp_path), "--cache", str(custom_cache)]),
            patch.object(container_run, "_DEFAULT_CACHE", tmp_path / "no_cache"),
        ):
            exit_code = container_run.main()

        assert exit_code == 0
        mock_run.assert_called_once_with(None, Path("/output"), data_dir=tmp_path, verbosity=0, cache_dir=custom_cache)
