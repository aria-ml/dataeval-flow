"""Pytest configuration and shared fixtures."""

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
    import dataeval_flow._logging as log_mod

    log_mod._initialized = False
    root = logging.getLogger()
    for h in root.handlers[:]:
        h.close()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    logging.getLogger("dataeval_flow").setLevel(logging.NOTSET)


class _BandTarget:
    """A duck-typed detection target. `dataeval.types` exports no constructible one."""

    def __init__(self, labels, boxes, scores):
        self.labels = labels
        self.boxes = boxes
        self.scores = scores


@pytest.fixture
def toy_multiband_dataset():
    """A tiny four-band object-detection dataset, for band-group and background tests."""
    import numpy as np

    rng = np.random.default_rng(0)

    class _Toy:
        def __init__(self, n=8):
            self.n = n
            self._images = [rng.integers(0, 255, (4, 16, 16), dtype=np.uint8) for _ in range(n)]
            self.metadata = {"id": "toy-multiband", "index2label": {0: "a", 1: "b"}}

        def __len__(self):
            return self.n

        def __getitem__(self, index):
            target = _BandTarget(
                np.array([index % 2], dtype=np.intp),
                np.array([[2.0, 2.0, 9.0, 9.0]], dtype=np.float32),
                np.ones(1, dtype=np.float32),
            )
            return self._images[index], target, {"id": index}

    return _Toy()


@pytest.fixture
def toy_images():
    """A tiny three-band classification dataset with one obvious pixel outlier."""
    import numpy as np

    rng = np.random.default_rng(0)

    class _Toy:
        def __init__(self, n=24):
            self.n = n
            self._images = [rng.integers(0, 255, (3, 16, 16), dtype=np.uint8) for _ in range(n)]
            self._images[5][:] = 250
            self.metadata = {"id": "toy", "index2label": {0: "a"}}

        def __len__(self):
            return self.n

        def __getitem__(self, index):
            return self._images[index], np.array([1.0]), {"id": index}

    return _Toy()
