"""Pytest configuration and shared fixtures.

Note: pytest's pythonpath is configured in pyproject.toml to include 'src/',
so imports of both 'dataeval_app' and 'container_run' work automatically.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_hf_dataset() -> MagicMock:
    """Create a mock HuggingFace dataset for testing."""
    mock = MagicMock()
    mock.keys.return_value = ["train", "test"]
    mock.__getitem__ = MagicMock(return_value="split_content")
    return mock
