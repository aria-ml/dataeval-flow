"""Tests for dataeval_flow._logging module."""

import logging
from pathlib import Path

import pytest

from dataeval_flow._logging import configure_log_levels, setup_logging


def _find_log_file(tmp_path: Path) -> Path:
    """Find the result.log file in the output directory."""
    return tmp_path / "result.log"


class TestSetupLogging:
    def test_setup_creates_file_handler(self, tmp_path: Path):
        setup_logging(tmp_path)

        log_file = _find_log_file(tmp_path)
        assert log_file.exists()

        logging.getLogger("dataeval_flow").info("hello from test")
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

        with patch("dataeval_flow._logging.os.makedirs", side_effect=OSError("Permission denied")):
            setup_logging(tmp_path)

        root = logging.getLogger()
        handler_types = [type(h).__name__ for h in root.handlers]
        assert "StreamHandler" in handler_types
        assert "FileHandler" not in handler_types

    def test_file_captures_debug_stream_filters_debug(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        setup_logging(tmp_path, verbosity=0)
        app_logger = logging.getLogger("dataeval_flow.test")

        app_logger.debug("debug-only-msg")
        app_logger.info("info-msg")

        for h in logging.getLogger().handlers:
            h.flush()

        log_content = _find_log_file(tmp_path).read_text(encoding="utf-8")
        stdout = capsys.readouterr().out

        # DEBUG appears in file but not stdout
        assert "debug-only-msg" in log_content
        assert "debug-only-msg" not in stdout

        # INFO does NOT appear in stdout at verbosity 0 (WARNING level)
        assert "info-msg" in log_content
        assert "info-msg" not in stdout

    def test_third_party_debug_suppressed(self, tmp_path: Path):
        setup_logging(tmp_path)

        torch_logger = logging.getLogger("torch")
        torch_logger.debug("torch-debug-noise")

        for h in logging.getLogger().handlers:
            h.flush()

        log_content = _find_log_file(tmp_path).read_text(encoding="utf-8")
        assert "torch-debug-noise" not in log_content

    def test_no_output_dir_creates_no_file_handler(self):
        setup_logging(None)

        root = logging.getLogger()
        handler_types = [type(h).__name__ for h in root.handlers]
        assert "FileHandler" not in handler_types
        assert "StreamHandler" in handler_types

    def test_verbosity_0_stream_at_warning(self, tmp_path: Path):
        setup_logging(tmp_path, verbosity=0)

        stream_handlers = [h for h in logging.getLogger().handlers if type(h) is logging.StreamHandler]
        assert len(stream_handlers) == 1
        assert stream_handlers[0].level == logging.WARNING

    def test_verbosity_2_stream_at_info(self, tmp_path: Path):
        setup_logging(tmp_path, verbosity=2)

        stream_handlers = [h for h in logging.getLogger().handlers if type(h) is logging.StreamHandler]
        assert len(stream_handlers) == 1
        assert stream_handlers[0].level == logging.INFO

    def test_verbosity_3_stream_at_debug(self, tmp_path: Path):
        setup_logging(tmp_path, verbosity=3)

        stream_handlers = [h for h in logging.getLogger().handlers if type(h) is logging.StreamHandler]
        assert len(stream_handlers) == 1
        assert stream_handlers[0].level == logging.DEBUG


class TestConfigureLogLevels:
    def test_configure_log_levels_app(self, tmp_path: Path):
        setup_logging(tmp_path)
        configure_log_levels(app_level="WARNING")

        assert logging.getLogger("dataeval_flow").level == logging.WARNING

    def test_configure_log_levels_lib(self, tmp_path: Path):
        setup_logging(tmp_path)
        configure_log_levels(lib_level="DEBUG")

        assert logging.getLogger().level == logging.DEBUG
