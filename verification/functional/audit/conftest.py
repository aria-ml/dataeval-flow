"""Start each test with empty in-memory dataset caches, so datasets that share an id never share results."""

import pytest


@pytest.fixture(autouse=True)
def _fresh_caches():
    from dataeval_flow._cache import DatasetCache

    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()
