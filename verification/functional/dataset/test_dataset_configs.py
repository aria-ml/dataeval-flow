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
from verification.fixtures import make_synthetic_dataset, write_coco_dataset, write_image_folder, write_yolo_dataset

pytestmark = pytest.mark.required


class TestDatasetConfigs:
    def test_image_folder_config_roundtrip(self) -> None:
        cfg = ImageFolderDatasetConfig(name="imgs", path="images")
        assert cfg.model_dump()["path"] == "images"

    def test_huggingface_config_roundtrip(self) -> None:
        cfg = HuggingFaceDatasetConfig(name="hf", path="placeholder", task="image_classification")
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


class TestLoadDataset:
    def test_load_dataset_image_folder(self, tmp_path: Path) -> None:
        from dataeval.protocols import AnnotatedDataset

        root = write_image_folder(tmp_path / "data")
        ds = load_dataset(root, dataset_format="image_folder", infer_labels=True)
        assert isinstance(ds, AnnotatedDataset)
        assert len(ds) > 0

    def test_load_dataset_coco(self, tmp_path: Path) -> None:
        root = write_coco_dataset(tmp_path / "coco", n=3)
        ds = load_dataset(root, dataset_format="coco", annotations_file="annotations.json", images_dir="images")
        assert len(ds) == 3
        image, target, _ = ds[0]
        assert image.shape == (3, 16, 16)
        assert len(target.boxes) == 1
        assert list(target.labels) == [1]

    def test_load_dataset_yolo(self, tmp_path: Path) -> None:
        root = write_yolo_dataset(tmp_path / "yolo", n=3)
        ds = load_dataset(root, dataset_format="yolo")
        assert len(ds) == 3
        image, target, _ = ds[0]
        assert image.shape == (3, 16, 16)
        assert len(target.boxes) == 1
        assert list(target.labels) == [0]

    def test_protocol_config_factory_dataset_is_returned(self) -> None:
        """A user factory builds the dataset; the config hands that very object to the pipeline."""
        from dataeval_flow.dataset import resolve_dataset

        calls: list[int] = []

        def factory(n: int) -> object:
            calls.append(n)
            return make_synthetic_dataset(n=n)

        built = factory(12)
        resolved = resolve_dataset(DatasetProtocolConfig(name="user_ds", dataset=built))

        assert calls == [12]
        assert resolved.dataset is built
        assert len(resolved.dataset) == 12  # type: ignore[arg-type]
