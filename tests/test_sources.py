"""Tests for source flattening — merge operands, cycles, and depth."""

import numpy as np
import pytest
from datamaite import (
    CategoryEntry,
    DatasetMetadata,
    ImageObjectDetectionSample,
    ObjectDetectionAnnotation,
    ObjectDetectionDataset,
    Taxonomy,
)

from dataeval_flow.config import PipelineConfig, SourceConfig, ViewConfig, ViewOperation
from dataeval_flow.config.schemas import DatasetProtocolConfig
from dataeval_flow.sources import MergeConfigError, flatten_source, resolve_source


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


_TARGET = ["Person", "Car", "Truck"]


def _od_dataset(dataset_id: str, names: list[str], label: int, n: int = 2) -> ObjectDetectionDataset:
    """A tiny datamaite OD dataset: one detection per image, all of class *label*."""
    taxonomy = Taxonomy(
        entries=tuple(CategoryEntry(source_id=i, name=name) for i, name in enumerate(names)),
        id_density="dense",
    )
    samples = tuple(
        ImageObjectDetectionSample(
            image_id=i,
            image_bytes=_PNG,
            file_name=f"img_{i:03d}.png",
            width=8,
            height=8,
            metadata={"weather": "sun" if i % 2 else "rain"},
            detections=(
                ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=label, category_name=names[label]),
            ),
        )
        for i in range(n)
    )
    return ObjectDetectionDataset(
        samples=samples, dataset_metadata=DatasetMetadata(taxonomy=taxonomy), dataset_id=dataset_id
    )


def _png() -> bytes:
    """An 8x8 black PNG, encoded through the codec datamaite decodes with."""
    import cv2

    ok, buf = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return bytes(buf)


_PNG = _png()


def _merge_config() -> PipelineConfig:
    """Two OD datasets conformed to one target, merged into a third source."""
    return PipelineConfig(
        datasets=[
            DatasetProtocolConfig(name="ds_a", dataset=_od_dataset("a", ["car", "van"], 0)),
            DatasetProtocolConfig(name="ds_b", dataset=_od_dataset("b", ["lorry", "bike"], 0)),
        ],
        views=[
            ViewConfig(
                name="conform_a",
                operations=[ViewOperation(type="Relabel", params={"class_remap": {"car": "Car"}, "target": _TARGET})],
            ),
            ViewConfig(
                name="conform_b",
                operations=[
                    ViewOperation(type="Relabel", params={"class_remap": {"lorry": "Truck"}, "target": _TARGET})
                ],
            ),
        ],
        sources=[
            SourceConfig(name="a", dataset="ds_a", view="conform_a"),
            SourceConfig(name="b", dataset="ds_b", view="conform_b"),
            SourceConfig(name="merged", merge=["a", "b"]),
        ],
    )


@pytest.mark.required
class TestResolveSource:
    """resolve_source loads a source's operands and merges them."""

    def test_plain_source_is_unviewed(self):
        """A non-merged source hands back the dataset as loaded. The workflow applies the view."""
        config = _merge_config()
        resolved = resolve_source("a", config)
        assert not resolved.is_merged
        assert len(resolved.operands) == 1
        assert resolved.dataset is resolved.operands[0].raw
        assert resolved.view_config is not None
        assert resolved.view_config.name == "conform_a"
        # unviewed, so the source vocabulary is still the dataset's own
        index2label = resolved.dataset.metadata["index2label"]  # type: ignore[reportTypedDictNotRequiredAccess]
        assert index2label == {0: "car", 1: "van"}

    def test_merged_source_concatenates_conformed_operands(self):
        config = _merge_config()
        resolved = resolve_source("merged", config)
        assert resolved.is_merged
        assert [op.source.name for op in resolved.operands] == ["a", "b"]
        assert len(resolved.dataset) == 4
        index2label = resolved.dataset.metadata["index2label"]  # type: ignore[reportTypedDictNotRequiredAccess]
        assert index2label == {0: "Person", 1: "Car", 2: "Truck"}

    def test_merged_ids_carry_their_operand_position(self):
        resolved = resolve_source("merged", _merge_config())
        ids = [resolved.dataset[i][2]["id"] for i in range(len(resolved.dataset))]
        assert ids == ["0:0", "0:1", "1:0", "1:1"]

    def test_mismatched_targets_are_refused(self):
        """Operands conformed to different targets index their labels differently."""
        config = _merge_config()
        assert config.views is not None
        config.views[1].operations[0].params = {
            "class_remap": {"lorry": "Truck"},
            "target": ["Truck", "Car"],
        }
        with pytest.raises(ValueError, match="index2label"):
            resolve_source("merged", config)

    def test_merged_source_applies_its_own_view_last(self):
        config = _merge_config()
        assert config.views is not None
        assert config.sources is not None
        head = ViewConfig(name="head", operations=[ViewOperation(type="Limit", params={"size": 3})])
        config.views.append(head)  # type: ignore[reportAttributeAccessIssue]
        config.sources[2] = SourceConfig(name="merged", merge=["a", "b"], view="head")  # type: ignore[reportIndexIssue]
        resolved = resolve_source("merged", config)
        assert len(resolved.dataset) == 4
        assert len(resolved.realized()) == 3

    def test_merged_cache_key_covers_every_operand(self):
        resolved = resolve_source("merged", _merge_config())
        assert resolved.cache_name == "merged"
        for operand in resolved.operands:
            assert operand.cache_key in resolved.cache_key
