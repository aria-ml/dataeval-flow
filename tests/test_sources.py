"""Tests for source flattening — merge operands, cycles, and depth."""

import pytest

from dataeval_flow.config import SourceConfig
from dataeval_flow.sources import MergeConfigError, flatten_source


def _pool(*sources: SourceConfig) -> list[SourceConfig]:
    return list(sources)


@pytest.mark.required
class TestFlattenSource:
    """flatten_source resolves a source to the leaf sources it reads."""

    def test_plain_source_is_its_own_operand(self):
        pool = _pool(SourceConfig(name="a", dataset="ds_a"))
        assert [s.name for s in flatten_source("a", pool)] == ["a"]

    def test_merge_returns_operands_in_order(self):
        pool = _pool(
            SourceConfig(name="a", dataset="ds_a"),
            SourceConfig(name="b", dataset="ds_b"),
            SourceConfig(name="m", merge=["a", "b"]),
        )
        assert [s.name for s in flatten_source("m", pool)] == ["a", "b"]

    def test_nested_merge_flattens_depth_first(self):
        pool = _pool(
            SourceConfig(name="a", dataset="ds_a"),
            SourceConfig(name="b", dataset="ds_b"),
            SourceConfig(name="c", dataset="ds_c"),
            SourceConfig(name="inner", merge=["a", "b"]),
            SourceConfig(name="outer", merge=["inner", "c"]),
        )
        assert [s.name for s in flatten_source("outer", pool)] == ["a", "b", "c"]

    def test_direct_cycle_is_refused(self):
        pool = _pool(SourceConfig(name="m", merge=["m", "m"]))
        with pytest.raises(MergeConfigError, match="merges itself"):
            flatten_source("m", pool)

    def test_indirect_cycle_names_the_path(self):
        pool = _pool(
            SourceConfig(name="a", dataset="ds_a"),
            SourceConfig(name="m", merge=["a", "n"]),
            SourceConfig(name="n", merge=["a", "m"]),
        )
        with pytest.raises(MergeConfigError, match=r"m -> n -> m"):
            flatten_source("m", pool)

    def test_depth_bound_is_enforced(self):
        pool = [SourceConfig(name="leaf", dataset="ds")]
        pool += [SourceConfig(name="lvl0", merge=["leaf", "leaf"])]
        for level in range(1, 12):
            pool.append(SourceConfig(name=f"lvl{level}", merge=[f"lvl{level - 1}", "leaf"]))
        with pytest.raises(MergeConfigError, match="nests merges more than 8 deep"):
            flatten_source("lvl11", pool)

    def test_unknown_operand_names_the_pool(self):
        pool = _pool(
            SourceConfig(name="a", dataset="ds_a"),
            SourceConfig(name="m", merge=["a", "nope"]),
        )
        with pytest.raises(ValueError, match=r"Unknown source: 'nope'"):
            flatten_source("m", pool)


@pytest.mark.required
class TestSourceConfigValidation:
    """A source names exactly one of `dataset` and `merge`."""

    def test_dataset_alone_is_valid(self):
        assert SourceConfig(name="a", dataset="ds").merge is None

    def test_merge_alone_is_valid(self):
        assert SourceConfig(name="m", merge=["a", "b"]).dataset is None

    def test_both_is_refused(self):
        with pytest.raises(ValueError, match="names both"):
            SourceConfig(name="m", dataset="ds", merge=["a", "b"])

    def test_neither_is_refused(self):
        with pytest.raises(ValueError, match="names neither"):
            SourceConfig(name="m")

    def test_single_operand_merge_is_refused(self):
        with pytest.raises(ValueError, match="at least two"):
            SourceConfig(name="m", merge=["a"])
