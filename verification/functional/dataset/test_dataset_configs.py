"""TC-3-1 — dataset formats and loading."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from dataeval.protocols import AnnotatedDataset
from PIL import Image
from pydantic import ValidationError

from dataeval_flow import load_config, load_dataset, load_source, run_tasks
from dataeval_flow.config import (
    CocoDatasetConfig,
    DatasetProtocolConfig,
    DemoDatasetConfig,
    HuggingFaceDatasetConfig,
    ImageFolderDatasetConfig,
    PipelineConfig,
    SourceConfig,
    YoloDatasetConfig,
)
from verification.fixtures import write_coco_dataset, write_image_folder, write_yolo_dataset
from verification.functional.dataset.formats import FORMATS, entry
from verification.functional.orchestration.support import InMemoryImages, pipeline_dict

pytestmark = pytest.mark.required


def _huggingface_detection(root: Path, n: int = 3) -> Path:
    """A local Hugging Face vision repository for detection: images and a `metadata.jsonl` of `objects`."""
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(root / f"{i}.png")
        rows.append({"file_name": f"{i}.png", "objects": {"bbox": [[1, 1, 6, 6]], "categories": [1]}})
    (root / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    return root


class TestDatasetConfigs:
    @pytest.mark.parametrize("fmt", sorted(FORMATS))
    def test_the_format_key_selects_the_config_class(self, fmt: str) -> None:
        config = PipelineConfig.model_validate({"datasets": [entry(fmt)]})
        assert config.datasets is not None
        assert type(config.datasets[0]) is FORMATS[fmt][0]

    @pytest.mark.parametrize("fmt", sorted(FORMATS))
    def test_a_config_round_trips_through_serialization(self, fmt: str) -> None:
        cls, _ = FORMATS[fmt]
        config = cls(**entry(fmt))
        assert cls(**config.model_dump()) == config
        assert config.model_dump()["format"] == fmt
        # And through a YAML file, as a user would keep it.
        assert PipelineConfig.model_validate({"datasets": [yaml.safe_load(yaml.safe_dump(config.model_dump()))]})

    def test_the_format_defaults_to_the_classs_own(self) -> None:
        assert ImageFolderDatasetConfig(name="a", path="p").format == "image_folder"
        assert CocoDatasetConfig(name="a", path="p").format == "coco"
        assert YoloDatasetConfig(name="a", path="p").format == "yolo"
        assert HuggingFaceDatasetConfig(name="a", path="p", task="image_classification").format == "huggingface"

    def test_an_unknown_format_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"datasets": [{"name": "a", "format": "tfrecord", "path": "p"}]})

    def test_a_setting_of_another_format_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
            ImageFolderDatasetConfig(name="a", path="p", annotations_file="ann.json")  # type: ignore[call-arg]

    def test_huggingface_needs_a_task(self) -> None:
        with pytest.raises(ValidationError, match="task"):
            HuggingFaceDatasetConfig(name="a", path="p")  # type: ignore[call-arg]
        with pytest.raises(ValidationError, match="task"):
            HuggingFaceDatasetConfig(name="a", path="p", task="segmentation")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("cls", "field", "value"),
        [
            (CocoDatasetConfig, "annotations_file", "../../../ann.json"),
            (CocoDatasetConfig, "images_dir", "/abs/imgs"),
            (YoloDatasetConfig, "yaml_file", "../../../data.yaml"),
            (YoloDatasetConfig, "ann_dir", "/abs/labels"),
        ],
    )
    def test_a_file_named_inside_a_dataset_must_stay_under_the_data_root(
        self, cls: type, field: str, value: str
    ) -> None:
        assert getattr(cls(name="a", path="sets/d", **{field: "../shared/x"}), field) == "../shared/x"
        with pytest.raises(ValidationError, match="must stay under the data root"):
            cls(name="a", path="sets/d", **{field: value})

    def test_the_dataset_path_is_required(self) -> None:
        with pytest.raises(ValidationError, match="path"):
            ImageFolderDatasetConfig(name="a")  # type: ignore[call-arg]


class TestLoadDataset:
    def test_load_dataset_image_folder(self, tmp_path: Path) -> None:
        root = write_image_folder(tmp_path / "data", n_per_class=3, n_classes=2)
        dataset = load_dataset(root, dataset_format="image_folder", infer_labels=True)
        assert isinstance(dataset, AnnotatedDataset)
        assert len(dataset) == 6
        image, target, datum = dataset[0]
        assert image.shape == (3, 8, 8)
        assert list(target) == [1.0, 0.0]
        assert datum["filename"] == "img_0.png"
        assert dataset.metadata["index2label"] == {0: "class_0", 1: "class_1"}

    def test_image_folder_without_infer_labels_has_no_labels(self, tmp_path: Path) -> None:
        root = write_image_folder(tmp_path / "data", n_per_class=2, n_classes=2)
        (root / "top.png").write_bytes((root / "class_0" / "img_0.png").read_bytes())
        flat = load_dataset(root, dataset_format="image_folder")
        recursive = load_dataset(root, dataset_format="image_folder", recursive=True)
        assert len(flat) == 1  # only the image at the top of the folder
        assert len(recursive) == 5
        assert len(flat[0][1]) == 0
        assert flat.metadata["index2label"] == {}

    def test_image_folder_ignores_files_that_are_not_images(self, tmp_path: Path) -> None:
        root = write_image_folder(tmp_path / "data", n_per_class=2, n_classes=2)
        (root / "class_0" / "notes.txt").write_text("not an image")
        assert len(load_dataset(root, dataset_format="image_folder", infer_labels=True)) == 4

    def test_load_dataset_coco(self, tmp_path: Path) -> None:
        root = write_coco_dataset(tmp_path / "coco", n=3)
        dataset = load_dataset(root, dataset_format="coco", annotations_file="annotations.json", images_dir="images")
        assert isinstance(dataset, AnnotatedDataset)
        assert len(dataset) == 3
        image, target, _ = dataset[0]
        assert image.shape == (3, 16, 16)
        assert len(target.boxes) == 1
        assert list(target.labels) == [1]
        assert dataset.metadata["index2label"] == {1: "thing"}

    def test_load_dataset_yolo(self, tmp_path: Path) -> None:
        root = write_yolo_dataset(tmp_path / "yolo", n=3)
        dataset = load_dataset(root, dataset_format="yolo")
        assert isinstance(dataset, AnnotatedDataset)
        assert len(dataset) == 3
        image, target, _ = dataset[0]
        assert image.shape == (3, 16, 16)
        assert len(target.boxes) == 1
        assert list(target.labels) == [0]
        assert dataset.metadata["index2label"] == {0: "thing"}

    def test_yolo_boxes_are_returned_as_pixel_corners(self, tmp_path: Path) -> None:
        """`0 0.5 0.5 0.4 0.4` on a 16 pixel image is the box from (4.8, 4.8) to (11.2, 11.2)."""
        dataset = load_dataset(write_yolo_dataset(tmp_path / "yolo", n=1), dataset_format="yolo")
        assert np.allclose(dataset[0][1].boxes[0], [4.8, 4.8, 11.2, 11.2])

    def test_yolo_split_selects_images(self, tmp_path: Path) -> None:
        root = write_yolo_dataset(tmp_path / "yolo", n=3)
        assert len(load_dataset(root, dataset_format="yolo", split="train")) == 3
        with pytest.raises(ValueError, match="No images matched split 'val'"):
            load_dataset(root, dataset_format="yolo", split="val")

    def test_load_dataset_huggingface_classification(self, tmp_path: Path) -> None:
        root = write_image_folder(tmp_path / "hf", n_per_class=3, n_classes=2)
        dataset = load_dataset(root, dataset_format="huggingface", task="image_classification")
        assert isinstance(dataset, AnnotatedDataset)
        assert len(dataset) == 6
        assert dataset.metadata["index2label"] == {0: "class_0", 1: "class_1"}
        assert dataset[0][0].shape == (3, 8, 8)

    def test_load_dataset_huggingface_detection_reads_a_split_folder(self, tmp_path: Path) -> None:
        _huggingface_detection(tmp_path / "hf" / "train")
        dataset = load_dataset(tmp_path / "hf", "train", dataset_format="huggingface", task="object_detection")
        assert isinstance(dataset, AnnotatedDataset)
        assert len(dataset) == 3
        _, target, _ = dataset[0]
        assert np.allclose(target.boxes, [[1, 1, 7, 7]])  # [x, y, width, height] becomes corners
        assert list(target.labels) == [1]

    def test_huggingface_defaults_to_the_classification_loader(self, tmp_path: Path) -> None:
        root = write_image_folder(tmp_path / "hf", n_per_class=2, n_classes=2)
        assert len(load_dataset(root)) == 4

    def test_a_dataset_that_loads_no_items_is_refused_with_a_hint(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(ValueError, match=r"Loaded 0 items .*format='coco'.*Check the path"):
            load_dataset(tmp_path / "empty", dataset_format="coco")

    def test_an_empty_or_missing_image_folder_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="No supported image files"):
            load_dataset(tmp_path / "empty", dataset_format="image_folder")
        with pytest.raises(FileNotFoundError, match="Image folder not found"):
            load_dataset(tmp_path / "missing", dataset_format="image_folder")

    def test_an_unsupported_format_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported dataset format: 'tfrecord'"):
            load_dataset(tmp_path, dataset_format="tfrecord")  # type: ignore[arg-type]

    def test_a_saved_arrow_dataset_gets_a_specific_message(self, tmp_path: Path) -> None:
        (tmp_path / "dump").mkdir()
        (tmp_path / "dump" / "dataset_dict.json").write_text("{}")
        with pytest.raises(ValueError, match=r"save_to_disk\(\).*Arrow dump"):
            load_dataset(tmp_path / "dump", dataset_format="huggingface")


class TestDatasetFromConfig:
    """Each config class loads its format through the pipeline, as a task's source reads it."""

    @pytest.mark.parametrize(
        ("writer", "entry", "expected"),
        [
            (
                lambda p: write_image_folder(p, n_per_class=2, n_classes=3),
                {"format": "image_folder", "infer_labels": True},
                6,
            ),
            (write_coco_dataset, {"format": "coco", "annotations_file": "annotations.json", "images_dir": "images"}, 3),
            (write_yolo_dataset, {"format": "yolo", "split": "train"}, 3),
            (_huggingface_detection, {"format": "huggingface", "task": "object_detection"}, 3),
        ],
        ids=["image_folder", "coco", "yolo", "huggingface"],
    )
    def test_a_configured_dataset_loads_from_the_data_root(
        self, tmp_path: Path, writer: Any, entry: dict[str, Any], expected: int
    ) -> None:
        writer(tmp_path / "data" / "ds")
        config = PipelineConfig.model_validate(
            {"datasets": [{"name": "ds", "path": "data/ds", **entry}], "sources": [{"name": "s", "dataset": "ds"}]}
        )
        dataset = load_source(config, "s", data_dir=tmp_path)
        assert isinstance(dataset, AnnotatedDataset)
        assert len(dataset) == expected

    def test_a_dataset_the_config_names_but_does_not_define_is_refused(self) -> None:
        config = PipelineConfig.model_validate({"sources": [{"name": "s", "dataset": "missing"}]})
        with pytest.raises(ValueError, match="No dataset configs defined"):
            load_source(config, "s")
        config = PipelineConfig.model_validate(pipeline_dict(sources=[{"name": "s", "dataset": "missing"}]))
        with pytest.raises(ValueError, match=r"Unknown dataset: 'missing'. Available: \['ds'\]"):
            load_source(config, "s")


class TestProtocolConfig:
    def test_protocol_config_factory_dataset_is_returned(self) -> None:
        """A user factory builds the dataset; the config hands that very object to the pipeline."""
        calls: list[int] = []

        def factory(n: int) -> object:
            calls.append(n)
            return InMemoryImages(n)

        built = factory(12)
        config = PipelineConfig(
            datasets=[DatasetProtocolConfig(name="user_ds", dataset=built)],
            sources=[SourceConfig(name="s", dataset="user_ds")],
        )
        loaded = load_source(config, "s")

        assert calls == [12]
        assert loaded is built
        assert isinstance(loaded, AnnotatedDataset)
        assert len(loaded) == 12  # type: ignore[arg-type]

    def test_a_protocol_dataset_runs_in_a_task(self) -> None:
        config = PipelineConfig(
            datasets=[DatasetProtocolConfig(name="user_ds", dataset=InMemoryImages(12))],
            sources=[SourceConfig(name="s", dataset="user_ds")],
            evaluators=[{"name": "labels", "type": "label-health"}],  # type: ignore[list-item]
            tasks=[{"name": "t", "evaluator": "labels", "sources": "s"}],  # type: ignore[list-item]
        )
        result = run_tasks(config)["t"]
        assert result.success, result.errors
        assert result.metadata.dataset_id == "user_ds"

    def test_the_format_is_maite_or_torchvision(self) -> None:
        assert DatasetProtocolConfig(name="a", dataset=object()).format == "maite"
        with pytest.raises(ValidationError):
            DatasetProtocolConfig(name="a", dataset=object(), format="coco")  # type: ignore[arg-type]

    def test_a_protocol_dataset_cannot_be_written_in_a_config_file(self, tmp_path: Path) -> None:
        """It holds an object, which a file cannot: the entry has no `dataset` to give."""
        entry = {"name": "a", "format": "maite"}
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"datasets": [entry]}))
        with pytest.raises(ValidationError):
            load_config(path)


class TestTorchvisionAdapter:
    def test_a_torchvision_classification_dataset_becomes_a_maite_dataset(self) -> None:
        class Cifar:
            classes = ["cat", "dog", "bird"]

            def __len__(self) -> int:
                return 4

            def __getitem__(self, index: int) -> tuple[Image.Image, int]:
                return Image.new("RGB", (6, 6), color=(index * 20, 0, 0)), index % 3

        config = PipelineConfig(
            datasets=[DatasetProtocolConfig(name="d", format="torchvision", dataset=Cifar())],
            sources=[SourceConfig(name="s", dataset="d")],
        )
        dataset = load_source(config, "s")

        assert isinstance(dataset, AnnotatedDataset)
        assert len(dataset) == 4
        assert dataset.metadata["index2label"] == {0: "cat", 1: "dog", 2: "bird"}
        image, target, datum = dataset[1]
        assert image.shape == (3, 6, 6)
        assert image.dtype == np.float32
        assert list(target) == [0.0, 1.0, 0.0]  # one-hot
        assert datum == {"id": 1}

    def test_a_torchvision_detection_dataset_gets_corner_boxes(self) -> None:
        import torch
        from torchvision import tv_tensors

        class Detection:
            classes = ["a", "b"]

            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int) -> tuple[Image.Image, dict[str, Any]]:
                boxes = tv_tensors.BoundingBoxes(torch.tensor([[1, 2, 3, 4]]), format="XYWH", canvas_size=(10, 10))
                return Image.new("RGB", (10, 10)), {"boxes": boxes, "labels": torch.tensor([1])}

        config = PipelineConfig(
            datasets=[DatasetProtocolConfig(name="d", format="torchvision", dataset=Detection())],
            sources=[SourceConfig(name="s", dataset="d")],
        )
        _, target, _ = load_source(config, "s")[0]
        assert np.allclose(target.boxes, [[1, 2, 4, 6]])  # x, y, x + width, y + height
        assert list(target.labels) == [1]
        assert target.scores.shape == (1, 2)


class TestDemoDatasets:
    def test_an_unknown_demo_dataset_is_refused_naming_the_known_ones(self) -> None:
        with pytest.raises(ValidationError, match=r"Unknown demo dataset 'os.system'.*DroneVehicle.*M3FD.*SeaDrone"):
            DemoDatasetConfig(name="d", dataset="os.system", path="data")

    def test_the_demo_loader_builds_a_known_dataset_from_the_data_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []

        class M3FD:
            metadata = {"id": "m3fd", "index2label": {0: "a"}}

            def __init__(self, root: str, **kwargs: Any) -> None:
                calls.append((root, kwargs))

            def __len__(self) -> int:
                return 1

            def __getitem__(self, index: int) -> Any:
                return np.zeros((3, 4, 4), dtype=np.uint8), np.zeros(1), {"id": index}

        module = types.ModuleType("maite_datasets.object_detection")
        module.M3FD = M3FD  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "maite_datasets.object_detection", module)

        config = PipelineConfig.model_validate(
            {
                "datasets": [
                    {"name": "d", "format": "demo", "dataset": "M3FD", "path": "tutorial", "image_set": "train"}
                ],
                "sources": [{"name": "s", "dataset": "d"}],
            }
        )
        load_source(config, "s", data_dir=tmp_path)
        # Nothing is downloaded unless the config asks, and the split is passed through.
        assert calls == [(str(tmp_path / "tutorial"), {"download": False, "image_set": "train"})]

    def test_the_demo_loader_says_what_to_install_when_maite_datasets_is_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "maite_datasets.object_detection", None)  # type: ignore[arg-type]
        config = PipelineConfig.model_validate(
            {
                "datasets": [{"name": "d", "format": "demo", "dataset": "SeaDrone", "path": "tutorial"}],
                "sources": [{"name": "s", "dataset": "d"}],
            }
        )
        with pytest.raises(ValueError, match=r"needs `maite-datasets`.*`coco`, `yolo`"):
            load_source(config, "s", data_dir=tmp_path)
