"""Tests for container_run.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import container_run


class TestContainerRunMain:
    @patch("dataeval_flow.runner.run")
    @patch.dict(container_run.CONTAINER_MOUNTS, {"config": Path("/tmp"), "output": Path("/tmp/out")})  # noqa: S108
    def test_failed_task_returns_one(
        self,
        mock_run_all: MagicMock,
    ):
        mock_run_all.return_value = 1

        exit_code = container_run.main()

        assert exit_code == 1
        mock_run_all.assert_called_once_with(Path("/tmp"), Path("/tmp/out"))  # noqa: S108

    @patch("dataeval_flow.runner.run")
    @patch.dict(container_run.CONTAINER_MOUNTS, {"config": Path("/tmp"), "output": Path("/tmp/out")})  # noqa: S108
    def test_successful_task_returns_zero(
        self,
        mock_run_all: MagicMock,
    ):
        mock_run_all.return_value = 0

        exit_code = container_run.main()

        assert exit_code == 0
        mock_run_all.assert_called_once_with(Path("/tmp"), Path("/tmp/out"))  # noqa: S108

    def test_missing_config_mount_returns_one(self, capsys: pytest.CaptureFixture):
        with patch.dict(container_run.CONTAINER_MOUNTS, {"config": Path("/nonexistent_path_xyz")}):
            exit_code = container_run.main()

        assert exit_code == 1
        out = capsys.readouterr().out
        assert "ERROR" in out

    @patch.dict(container_run.CONTAINER_MOUNTS, {"config": Path("/tmp")})  # noqa: S108
    def test_import_error_returns_one(self, capsys: pytest.CaptureFixture):
        with patch.dict("sys.modules", {"dataeval_flow.runner": None}):
            exit_code = container_run.main()

        assert exit_code == 1
        out = capsys.readouterr().out
        assert "ERROR" in out
