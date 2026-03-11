"""Pytest configuration and shared fixtures.

Note: pytest's pythonpath is configured in pyproject.toml to include 'src/',
so imports of both 'dataeval_app' and 'container_run' work automatically.
"""

import logging
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_hf_dataset() -> MagicMock:
    """Create a mock HuggingFace dataset for testing."""
    mock = MagicMock()
    mock.keys.return_value = ["train", "test"]
    mock.__getitem__ = MagicMock(return_value="split_content")
    return mock


@pytest.fixture(autouse=True)
def _reset_logging():
    yield
    import dataeval_app._logging as log_mod

    log_mod._initialized = False
    root = logging.getLogger()
    for h in root.handlers[:]:
        h.close()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    logging.getLogger("dataeval_app").setLevel(logging.NOTSET)
    logging.getLogger("container_run").setLevel(logging.NOTSET)
