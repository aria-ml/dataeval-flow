"""Pytest configuration and shared fixtures.

Note: pytest's pythonpath is configured in pyproject.toml to include 'src/',
so imports of both 'dataeval_app' and 'container_run' work automatically.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_dataset_path(tmp_path: Path) -> Path:
    """Create a temporary dataset directory for testing."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    return dataset_dir


@pytest.fixture
def mock_maite_dataset() -> MagicMock:
    """Create a mock MAITE dataset for testing."""
    mock = MagicMock()
    mock.__len__ = MagicMock(return_value=10)
    mock.__getitem__ = MagicMock(return_value=(MagicMock(shape=(3, 224, 224)), MagicMock(), None))
    mock.metadata = {
        "id": "test_dataset",
        "index2label": {0: "class_a", 1: "class_b"},
    }
    return mock


@pytest.fixture
def mock_hf_dataset() -> MagicMock:
    """Create a mock HuggingFace dataset for testing."""
    mock = MagicMock()
    mock.keys.return_value = ["train", "test"]
    mock.__getitem__ = MagicMock(return_value="split_content")
    return mock
