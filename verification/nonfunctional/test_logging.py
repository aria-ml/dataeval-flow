"""TC-16-1 (NFR-5) — stdlib logging integration."""

from __future__ import annotations

import logging

import pytest


@pytest.mark.test_case("16-1")
class TestLogging:
    def test_logging_module_exports_setup(self) -> None:
        from dataeval_flow import _logging

        assert hasattr(_logging, "setup_logging"), "expected setup_logging entrypoint"

    def test_setup_logging_attaches_root_handler(self) -> None:
        import logging

        from dataeval_flow import _logging
        from dataeval_flow._logging import setup_logging

        root = logging.getLogger()
        original_handlers = list(root.handlers)
        original_level = root.level
        original_initialized = _logging._initialized
        try:
            for h in original_handlers:
                root.removeHandler(h)
            _logging._initialized = False
            setup_logging()
            assert len(root.handlers) >= 1
        finally:
            for h in list(root.handlers):
                root.removeHandler(h)
            for h in original_handlers:
                root.addHandler(h)
            root.setLevel(original_level)
            _logging._initialized = original_initialized

    def test_module_logger_is_logging_logger(self) -> None:
        import dataeval_flow.runner as runner

        assert isinstance(runner._logger, logging.Logger)
