"""Tests for adapting a resolved source into a datamaite dataset."""

from pathlib import Path
from types import SimpleNamespace

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
from dataeval_flow.export import build_od_dataset
from dataeval_flow.sources import resolve_source
from tests.test_sources import _PNG, _merge_config

_CAR = ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=0, category_name="Car")


def _disk_config(
    tmp_path: Path,
    *,
    operations: list[ViewOperation] | None = None,
    splits: tuple[str | None, ...] = (None,),
    declare_size: bool = True,
) -> PipelineConfig:
    """A single source whose images are PNGs on disk rather than bytes in memory."""
    samples = []
    for index, split in enumerate(splits):
        png = tmp_path / f"on_disk_{index}.png"
        png.write_bytes(_PNG)
        samples.append(
            ImageObjectDetectionSample(
                image_id=index,
                path_or_uri=str(png),
                file_name=png.name,
                width=8 if declare_size else None,
                height=8 if declare_size else None,
                split=split,
                metadata={"weather": "rain"},
                detections=(_CAR,),
            )
        )
    dataset = ObjectDetectionDataset(
        samples=tuple(samples),
        dataset_metadata=DatasetMetadata(
            taxonomy=Taxonomy(entries=(CategoryEntry(source_id=0, name="Car"),), id_density="dense")
        ),
        dataset_id="on_disk",
    )
    return PipelineConfig(
        datasets=[DatasetProtocolConfig(name="ds_disk", dataset=dataset)],
        views=[ViewConfig(name="transform", operations=operations)] if operations else None,
        sources=[SourceConfig(name="disk", dataset="ds_disk", view="transform" if operations else None)],
    )


def _built_from_disk(tmp_path: Path, **kwargs) -> ObjectDetectionDataset:
    """Export the on-disk source built with *kwargs*."""
    resolved = resolve_source("disk", _disk_config(tmp_path, **kwargs))
    return build_od_dataset(resolved, dataset_metadata=DatasetMetadata())


class _PlainOdDataset:
    """A MAITE OD dataset that is not datamaite-backed, so no source records exist."""

    @property
    def metadata(self) -> dict:
        return {"id": "plain", "index2label": {0: "Car"}}

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        target = SimpleNamespace(
            boxes=np.array([[1.0, 2.0, 4.0, 6.0]], dtype=np.float32),
            labels=np.array([0], dtype=np.int64),
        )
        return np.zeros((3, 8, 8), dtype=np.uint8), target, {"id": index, "weather": "rain"}


def _plain_config() -> PipelineConfig:
    """A source backed by a dataset datamaite never loaded."""
    return PipelineConfig(
        datasets=[DatasetProtocolConfig(name="ds_plain", dataset=_PlainOdDataset())],
        sources=[SourceConfig(name="plain", dataset="ds_plain")],
    )


def _sparse_config() -> PipelineConfig:
    """A source whose category ids are COCO-shaped: sparse, and not starting at zero."""
    names = {1: "person", 17: "cat", 90: "toothbrush"}
    dataset = ObjectDetectionDataset(
        samples=(
            ImageObjectDetectionSample(
                image_id=0,
                image_bytes=_PNG,
                file_name="img_000.png",
                width=8,
                height=8,
                detections=(ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=17, category_name="cat"),),
            ),
        ),
        dataset_metadata=DatasetMetadata(
            taxonomy=Taxonomy(
                entries=tuple(CategoryEntry(source_id=code, name=name) for code, name in names.items()),
                id_density="sparse",
            )
        ),
        dataset_id="sparse",
    )
    return PipelineConfig(
        datasets=[DatasetProtocolConfig(name="ds_sparse", dataset=dataset)],
        sources=[SourceConfig(name="sparse", dataset="ds_sparse")],
    )


@pytest.mark.required
class TestBuildOdDataset:
    """build_od_dataset materializes the corpus datamaite's writers need."""

    def _built(self, source: str = "merged"):
        return build_od_dataset(resolve_source(source, _merge_config()), dataset_metadata=DatasetMetadata())

    def test_every_datum_becomes_a_sample(self):
        assert len(self._built().samples) == 4

    def test_image_ids_are_integers(self):
        """COCO drops a sample whose image_id is not an int, so allocate them."""
        assert [s.image_id for s in self._built().samples] == [0, 1, 2, 3]

    def test_the_merged_id_survives_as_source_id(self):
        assert [s.metadata["source_id"] for s in self._built().samples] == ["0:0", "0:1", "1:0", "1:1"]

    def test_file_names_are_prefixed_per_operand(self):
        """Two operands both hold img_000.png; without a prefix one overwrites the other."""
        names = [s.file_name for s in self._built().samples]
        assert names == ["0/img_000.png", "0/img_001.png", "1/img_000.png", "1/img_001.png"]

    def test_a_single_source_keeps_its_file_names(self):
        assert [s.file_name for s in self._built("a").samples] == ["img_000.png", "img_001.png"]

    def test_boxes_convert_from_xyxy_to_xywh(self):
        """The fixture's source box is xywh (1,2,3,4); MAITE hands it back as xyxy (1,2,4,6)."""
        assert self._built().samples[0].detections[0].bbox == (1.0, 2.0, 3.0, 4.0)

    def test_detections_carry_the_conformed_class(self):
        first = self._built().samples[0].detections[0]
        assert first.category_id == 1
        assert first.category_name == "Car"
        last = self._built().samples[2].detections[0]
        assert last.category_id == 2
        assert last.category_name == "Truck"

    def test_datum_metadata_passes_through_without_the_derived_keys(self):
        meta = self._built().samples[0].metadata
        assert meta["weather"] == "rain"
        assert "id" not in meta
        assert "width" not in meta

    def test_bytes_backed_source_falls_back_to_encoded_bytes(self):
        """A source with no path on disk still exports, by carrying its pixels."""
        sample = self._built().samples[0]
        assert sample.path_or_uri is None
        assert sample.image_bytes is not None

    def test_a_relabel_that_dropped_everything_yields_no_samples(self):
        config = _merge_config()
        assert config.views is not None
        config.views[0].operations[0].params = {"class_remap": {"absent": "Car"}, "target": ["Person", "Car"]}
        config.views[1].operations[0].params = {"class_remap": {"absent": "Car"}, "target": ["Person", "Car"]}
        built = build_od_dataset(resolve_source("merged", config), dataset_metadata=DatasetMetadata())
        assert built.samples == ()


@pytest.mark.required
class TestImageReference:
    """An export references the file on disk only where the file still matches the boxes."""

    def test_a_path_backed_source_is_referenced_not_re_encoded(self, tmp_path):
        """Referencing the file on disk is what makes exporting a large corpus affordable."""
        sample = _built_from_disk(tmp_path).samples[0]
        assert sample.path_or_uri == str(tmp_path / "on_disk_0.png")
        assert sample.image_bytes is None

    def test_a_view_that_leaves_the_imagery_alone_still_references_the_file(self, tmp_path):
        operations = [ViewOperation(type="Limit", params={"size": 1})]
        sample = _built_from_disk(tmp_path, operations=operations).samples[0]
        assert sample.path_or_uri == str(tmp_path / "on_disk_0.png")
        assert sample.image_bytes is None

    def test_a_view_that_transforms_the_imagery_stops_referencing_the_file(self, tmp_path):
        """A crop rewrites the boxes, so the untransformed file would hold the wrong pixels."""
        operations = [ViewOperation(type="Crop", params={"region": (0, 0, 4, 4)})]
        sample = _built_from_disk(tmp_path, operations=operations).samples[0]
        assert sample.path_or_uri is None
        assert sample.image_bytes is not None
        assert (sample.width, sample.height) == (4, 4)
        assert sample.detections[0].bbox == (1.0, 2.0, 3.0, 2.0)

    def test_a_view_that_selects_channels_stops_referencing_the_file(self, tmp_path):
        """datamaite decodes every referenced file as 3-channel RGB, so one band is not it."""
        import cv2

        operations = [ViewOperation(type="SelectChannels", params={"channels": [0]})]
        sample = _built_from_disk(tmp_path, operations=operations).samples[0]
        assert sample.path_or_uri is None
        assert sample.image_bytes is not None
        decoded = cv2.imdecode(np.frombuffer(sample.image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        assert decoded.shape == (8, 8)

    def test_imagery_cv2_cannot_encode_is_refused(self, tmp_path):
        """A two-channel PNG raises inside OpenCV, so name the count rather than pass it on."""
        operations = [ViewOperation(type="SelectChannels", params={"channels": [0, 1]})]
        with pytest.raises(ValueError, match="2 channels"):
            _built_from_disk(tmp_path, operations=operations)

    def test_a_source_that_declares_no_size_still_references_the_file(self, tmp_path):
        """Fall back to the datum metadata, which datamaite fills in from the image itself."""
        sample = _built_from_disk(tmp_path, declare_size=False).samples[0]
        assert sample.path_or_uri == str(tmp_path / "on_disk_0.png")
        assert (sample.width, sample.height) == (8, 8)

    def test_a_source_datamaite_never_loaded_carries_its_pixels(self):
        built = build_od_dataset(resolve_source("plain", _plain_config()), dataset_metadata=DatasetMetadata())
        assert [s.file_name for s in built.samples] == ["00000000.png", "00000001.png"]
        assert all(s.path_or_uri is None and s.image_bytes is not None for s in built.samples)
        assert (built.samples[0].width, built.samples[0].height) == (8, 8)
        assert built.samples[0].detections[0].bbox == (1.0, 2.0, 3.0, 4.0)

    def test_splits_survive_the_export(self, tmp_path):
        """A writer resolves `sample.split or default_split`, so dropping it leaks val into train."""
        built = _built_from_disk(tmp_path, splits=("train", "val"))
        assert [s.split for s in built.samples] == ["train", "val"]

    def test_imagery_that_is_not_8_bit_is_refused(self, tmp_path):
        """cv2 casts silently, so a normalized view would otherwise export as black."""
        operations = [ViewOperation(type="Resize", params={"size": 16})]
        with pytest.raises(ValueError, match="8-bit"):
            _built_from_disk(tmp_path, operations=operations)


@pytest.mark.required
class TestTaxonomy:
    """The taxonomy an export attaches describes the corpus's own vocabulary."""

    def test_taxonomy_comes_from_the_conformed_vocabulary(self):
        built = build_od_dataset(resolve_source("merged", _merge_config()), dataset_metadata=DatasetMetadata())
        taxonomy = built.dataset_metadata.taxonomy
        assert taxonomy is not None
        assert list(taxonomy.ordered_names) == ["Person", "Car", "Truck"]
        assert taxonomy.id_density == "dense"

    def test_sparse_source_ids_are_reported_as_sparse(self):
        """A source read straight from COCO keeps that format's gapped ids."""
        built = build_od_dataset(resolve_source("sparse", _sparse_config()), dataset_metadata=DatasetMetadata())
        taxonomy = built.dataset_metadata.taxonomy
        assert taxonomy is not None
        assert taxonomy.id_density == "sparse"
        assert [e.source_id for e in taxonomy.entries] == [1, 17, 90]
        assert taxonomy.ordered_names == ("person", "cat", "toothbrush")
        assert built.samples[0].detections[0].category_name == "cat"

    def test_a_taxonomy_already_set_is_kept(self):
        declared = Taxonomy(entries=(CategoryEntry(source_id=7, name="Declared"),), source_dataset="upstream")
        built = build_od_dataset(
            resolve_source("merged", _merge_config()), dataset_metadata=DatasetMetadata(taxonomy=declared)
        )
        assert built.dataset_metadata.taxonomy is declared
