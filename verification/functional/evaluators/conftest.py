"""Start each test with empty in-memory dataset caches, so datasets that share an id never share results."""

from pathlib import Path

import pytest

from verification.functional.chains._toys import SUFFICIENCY_TRAINER


@pytest.fixture(autouse=True)
def _fresh_caches():
    from dataeval_flow._cache import DatasetCache

    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


@pytest.fixture(autouse=True)
def _data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A data root holding the trainer the `sufficiency` toy task names, which it reads from ``DATAEVAL_DATA``."""
    (tmp_path / "trainers").mkdir()
    (tmp_path / "trainers" / "majority.py").write_text(SUFFICIENCY_TRAINER)
    monkeypatch.setenv("DATAEVAL_DATA", str(tmp_path))
