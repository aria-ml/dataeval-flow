"""Tests for dataeval_app._logging module."""

import logging
from pathlib import Path

import pytest

from dataeval_app._logging import configure_log_levels, setup_logging


class TestSetupLogging:
    def test_setup_creates_file_handler(self, tmp_path: Path):
        setup_logging(tmp_path)

        log_file = tmp_path / "log.txt"
        assert log_file.exists()

        logging.getLogger("dataeval_app").info("hello from test")
        # Flush handlers to ensure content is written
        for h in logging.getLogger().handlers:
            h.flush()

        content = log_file.read_text(encoding="utf-8")
        assert "hello from test" in content

    def test_setup_creates_stream_handler(self, tmp_path: Path):
        setup_logging(tmp_path)

        root = logging.getLogger()
        handler_types = [type(h).__name__ for h in root.handlers]
        assert "StreamHandler" in handler_types

    def test_duplicate_guard(self, tmp_path: Path):
        setup_logging(tmp_path)
        count_after_first = len(logging.getLogger().handlers)

        setup_logging(tmp_path)
        count_after_second = len(logging.getLogger().handlers)

        assert count_after_first == count_after_second

    def test_unwritable_dir_fallback(self, tmp_path: Path):
        from unittest.mock import patch

        with patch("dataeval_app._logging.os.makedirs", side_effect=OSError("Permission denied")):
            setup_logging(tmp_path)

        root = logging.getLogger()
        handler_types = [type(h).__name__ for h in root.handlers]
        assert "StreamHandler" in handler_types
        assert "FileHandler" not in handler_types

    def test_file_captures_debug_stream_filters_debug(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        setup_logging(tmp_path)
        app_logger = logging.getLogger("dataeval_app.test")

        app_logger.debug("debug-only-msg")
        app_logger.info("info-msg")

        for h in logging.getLogger().handlers:
            h.flush()

        log_content = (tmp_path / "log.txt").read_text(encoding="utf-8")
        stdout = capsys.readouterr().out

        # DEBUG appears in file but not stdout
        assert "debug-only-msg" in log_content
        assert "debug-only-msg" not in stdout

        # INFO appears in both
        assert "info-msg" in log_content
        assert "info-msg" in stdout

    def test_third_party_debug_suppressed(self, tmp_path: Path):
        setup_logging(tmp_path)

        torch_logger = logging.getLogger("torch")
        torch_logger.debug("torch-debug-noise")

        for h in logging.getLogger().handlers:
            h.flush()

        log_content = (tmp_path / "log.txt").read_text(encoding="utf-8")
        assert "torch-debug-noise" not in log_content


class TestConfigureLogLevels:
    def test_configure_log_levels_app(self, tmp_path: Path):
        setup_logging(tmp_path)
        configure_log_levels(app_level="WARNING")

        assert logging.getLogger("dataeval_app").level == logging.WARNING
        assert logging.getLogger("container_run").level == logging.WARNING

    def test_configure_log_levels_lib(self, tmp_path: Path):
        setup_logging(tmp_path)
        configure_log_levels(lib_level="DEBUG")

        assert logging.getLogger().level == logging.DEBUG
