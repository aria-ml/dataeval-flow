"""TC-3-1 — dataset configs and load_dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataeval_flow import (
    CocoDatasetConfig,
    DatasetProtocolConfig,
    HuggingFaceDatasetConfig,
    ImageFolderDatasetConfig,
    YoloDatasetConfig,
    load_dataset,
)
from verification.fixtures import write_image_folder


@pytest.mark.test_case("3-1")
class TestDatasetConfigs:
    def test_image_folder_config_roundtrip(self) -> None:
        cfg = ImageFolderDatasetConfig(name="imgs", path="images")
        assert cfg.model_dump()["path"] == "images"

    def test_huggingface_config_roundtrip(self) -> None:
        cfg = HuggingFaceDatasetConfig(name="hf", path="placeholder")
        assert cfg.format == "huggingface"
        assert cfg.name == "hf"

    def test_coco_config_roundtrip(self) -> None:
        cfg = CocoDatasetConfig(name="coco", path="imgs", annotations_file="ann.json")
        assert cfg.annotations_file is not None
        assert cfg.annotations_file.endswith("ann.json")

    def test_yolo_config_roundtrip(self) -> None:
        cfg = YoloDatasetConfig(name="yolo", path="yolo")
        assert cfg.path == "yolo"

    def test_protocol_config_roundtrip(self) -> None:
        cfg = DatasetProtocolConfig(name="proto", dataset=object())
        assert cfg.name == "proto"


@pytest.mark.test_case("3-1")
class TestLoadDataset:
    def test_load_dataset_image_folder(self, tmp_path: Path) -> None:
        from dataeval.protocols import AnnotatedDataset

        root = write_image_folder(tmp_path / "data")
        ds = load_dataset(root, dataset_format="image_folder", infer_labels=True)
        assert isinstance(ds, AnnotatedDataset)
        assert len(ds) > 0
