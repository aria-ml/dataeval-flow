"""The `factor-leakage` evaluator: the raw values of named factors that two sources hold (audit spec §10.1)."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from dataeval_flow import run
from dataeval_flow._cache import DatasetCache
from dataeval_flow._policy import strip_row_level
from dataeval_flow.evaluators.quality import FactorLeakageConfig
from tests.evaluator_toys import ToyFactors


class _Scenes(ToyFactors):
    """ToyFactors whose items also carry a `scene`: s0, s1, s2 in turn, shifted by `offset`."""

    def __init__(self, count: int = 30, offset: int = 0) -> None:
        super().__init__(count)
        self._offset = offset

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        image, target, meta = super().__getitem__(index)
        return image, target, {**meta, "scene": f"s{(index + self._offset) % 3}"}


def test_it_counts_each_value_both_sources_hold() -> None:
    result = run(FactorLeakageConfig(factors=["scene"]), {"train": _Scenes(30), "test": _Scenes(30, offset=1)})
    assert result.success, result.errors
    data = result.output.data()
    assert data["sources"] == ["train", "test"]
    assert data["items"] == [30, 30]
    assert data["factors"]["scene"] == {"s0": [10, 10], "s1": [10, 10], "s2": [10, 10]}


def test_it_reads_a_factor_the_policy_excludes() -> None:
    # Group IDs are what policies exclude from binning; the raw column survives in the Metadata frame.
    result = run(
        FactorLeakageConfig(factors=["scene"], metadata="p"),
        {"train": _Scenes(30), "test": _Scenes(30)},
        definitions=[_policy_excluding_scene()],
    )
    assert result.success, result.errors
    assert set(result.output.data()["factors"]["scene"]) == {"s0", "s1", "s2"}


def test_a_level_prefixed_name_matches_its_bare_form() -> None:
    assert strip_row_level("unit_scene") == strip_row_level("scene") == "scene"
    result = run(FactorLeakageConfig(factors=["unit_scene"]), {"a": _Scenes(12), "b": _Scenes(12)})
    assert result.success, result.errors
    assert "unit_scene" in result.output.data()["factors"]


def test_a_source_that_lacks_a_factor_fails_naming_it() -> None:
    result = run(FactorLeakageConfig(factors=["nope"]), {"train": _Scenes(12), "test": _Scenes(12)})
    assert not result.success
    assert "Source 'train' has no factor 'nope'" in result.errors[0]


def _policy_excluding_scene() -> Any:
    from dataeval_flow.config import MetadataPolicyConfig

    return MetadataPolicyConfig(name="p", exclude=["scene"])


def test_it_reads_an_excluded_factor_from_metadata_loaded_back_from_the_cache(tmp_path: Path) -> None:
    import dataeval_flow._cache as cache_module

    config = FactorLeakageConfig(factors=["scene"], metadata="p")
    data = {"train": _Scenes(30), "test": _Scenes(30)}
    policy = [_policy_excluding_scene()]
    first = run(config, data, definitions=policy, cache_dir=tmp_path)
    assert first.success, first.errors
    DatasetCache.clear_instances()
    try:
        with patch.object(cache_module, "_do_compute_metadata", side_effect=AssertionError("recomputed")) as compute:
            second = run(config, data, definitions=policy, cache_dir=tmp_path)
    finally:
        DatasetCache.clear_instances()
    assert second.success, second.errors
    assert compute.call_count == 0
    assert set(second.output.data()["factors"]["scene"]) == {"s0", "s1", "s2"}
