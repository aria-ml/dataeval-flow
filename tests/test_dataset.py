"""Tests for the dataeval_flow package."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from torchvision.tv_tensors import BoundingBoxes


@pytest.mark.required
class TestImports:
    """Test that core imports work correctly."""

    def test_dataeval_import(self) -> None:
        """Verify DataEval can be imported successfully."""
        import dataeval

        assert dataeval is not None

    def test_package_import(self) -> None:
        """Verify the dataeval_flow package can be imported."""
        import dataeval_flow

        assert hasattr(dataeval_flow, "__version__")

    def test_public_api_exports(self) -> None:
        """Verify __all__ exports are available from package root."""
        from dataeval_flow import load_config, load_dataset, run_tasks

        assert callable(load_config)
        assert callable(load_dataset)
        assert callable(run_tasks)

    def test_workflow_api_exports(self) -> None:
        """Verify workflow exports are available from package root."""
        from dataeval_flow import get_workflow, list_workflows, run_tasks

        assert callable(get_workflow)
        assert callable(list_workflows)
        assert callable(run_tasks)


@pytest.mark.required
class TestDatasetModule:
    """Test the dataset library module."""

    def test_load_dataset_huggingface_datasetdict(self, mock_hf_dataset: MagicMock) -> None:
        """Test loading a DatasetDict with explicit split selection."""
        with (
            patch("datasets.load_from_disk") as mock_load,
            patch("maite_datasets.adapters.from_huggingface") as mock_adapter,
        ):
            mock_load.return_value = mock_hf_dataset
            mock_adapter.return_value = MagicMock()

            from dataeval_flow.dataset import load_dataset_huggingface

            load_dataset_huggingface(Path("dummy/path"), split="train")

            # Should use the explicitly requested split
            mock_hf_dataset.__getitem__.assert_called_with("train")
            mock_adapter.assert_called()

    def test_load_dataset_delegates(self) -> None:
        """Test load_dataset defaults to huggingface."""
        with patch("dataeval_flow.dataset.load_dataset_huggingface") as mock_hf:
            mock_hf.return_value = MagicMock()

            from dataeval_flow.dataset import load_dataset

            load_dataset(Path("/some/path"))
            mock_hf.assert_called_once_with(Path("/some/path"), split=None)

    def test_load_dataset_image_folder_dispatch(self) -> None:
        """Test load_dataset dispatches to image_folder loader."""
        with patch("dataeval_flow.dataset.load_dataset_image_folder") as mock_if:
            mock_if.return_value = MagicMock()

            from dataeval_flow.dataset import load_dataset

            load_dataset(Path("/some/path"), dataset_format="image_folder")
            mock_if.assert_called_once_with(Path("/some/path"), recursive=False, infer_labels=False)

    def test_load_dataset_image_folder_with_infer_labels(self) -> None:
        """Test load_dataset passes infer_labels through."""
        with patch("dataeval_flow.dataset.load_dataset_image_folder") as mock_if:
            mock_if.return_value = MagicMock()

            from dataeval_flow.dataset import load_dataset

            load_dataset(Path("/some/path"), dataset_format="image_folder", infer_labels=True)
            mock_if.assert_called_once_with(Path("/some/path"), recursive=False, infer_labels=True)

    def test_load_dataset_unsupported_format_raises(self) -> None:
        """Test load_dataset raises ValueError for unsupported formats."""
        from dataeval_flow.dataset import load_dataset

        with pytest.raises(ValueError, match="Unsupported dataset format"):
            load_dataset(Path("/some/path"), dataset_format="unknown")  # type: ignore[arg-type]

    def test_load_dataset_coco_dispatch(self) -> None:
        """Test load_dataset dispatches to COCO loader."""
        with patch("dataeval_flow.dataset.load_dataset_coco") as mock_coco:
            mock_coco.return_value = MagicMock()

            from dataeval_flow.dataset import load_dataset

            load_dataset(
                Path("/some/path"),
                dataset_format="coco",
                annotations_file="ann.json",
                images_dir="imgs",
                classes_file="cls.txt",
            )
            mock_coco.assert_called_once_with(
                Path("/some/path"), annotations_file="ann.json", images_dir="imgs", classes_file="cls.txt"
            )

    def test_load_dataset_yolo_dispatch(self) -> None:
        """Test load_dataset dispatches to YOLO loader."""
        with patch("dataeval_flow.dataset.load_dataset_yolo") as mock_yolo:
            mock_yolo.return_value = MagicMock()

            from dataeval_flow.dataset import load_dataset

            load_dataset(
                Path("/some/path"), dataset_format="yolo", images_dir="imgs", labels_dir="lbls", classes_file="cls.txt"
            )
            mock_yolo.assert_called_once_with(
                Path("/some/path"), images_dir="imgs", labels_dir="lbls", classes_file="cls.txt"
            )


# ---------------------------------------------------------------------------
# ImageFolderDataset — helpers
# ---------------------------------------------------------------------------


def _create_image(path: Path, width: int = 8, height: int = 8, mode: str = "RGB") -> None:
    """Create a minimal test image at the given path."""
    from PIL import Image

    img = Image.new(mode, (width, height), color="red")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


# ---------------------------------------------------------------------------
# ImageFolderDataset — unlabeled
# ---------------------------------------------------------------------------


@pytest.mark.required
class TestImageFolderDatasetUnlabeled:
    """Tests for ImageFolderDataset in unlabeled mode (infer_labels=False)."""

    def test_happy_path(self, tmp_path: Path) -> None:
        """Discover images in a flat directory."""
        from dataeval_flow.dataset import ImageFolderDataset

        for name in ["a.png", "b.jpg", "c.jpeg"]:
            _create_image(tmp_path / name)

        ds = ImageFolderDataset(tmp_path)
        assert len(ds) == 3
        img, target, meta = ds[0]
        assert img.shape == (3, 8, 8)
        assert img.dtype == np.float32
        assert target.shape == (0,)
        assert meta["filename"] == "a.png"
        assert ds.metadata["index2label"] == {}

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        """Non-existent path raises FileNotFoundError."""
        from dataeval_flow.dataset import ImageFolderDataset

        with pytest.raises(FileNotFoundError, match="Image folder not found"):
            ImageFolderDataset(tmp_path / "nonexistent")

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        """Empty directory raises FileNotFoundError."""
        from dataeval_flow.dataset import ImageFolderDataset

        with pytest.raises(FileNotFoundError, match="No supported image files"):
            ImageFolderDataset(tmp_path)

    def test_mixed_files_filters(self, tmp_path: Path) -> None:
        """Only image files are discovered; others are ignored."""
        from dataeval_flow.dataset import ImageFolderDataset

        _create_image(tmp_path / "img.png")
        (tmp_path / "data.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("a,b")

        ds = ImageFolderDataset(tmp_path)
        assert len(ds) == 1

    def test_grayscale_image(self, tmp_path: Path) -> None:
        """Grayscale image is converted to 3-channel RGB."""
        from dataeval_flow.dataset import ImageFolderDataset

        _create_image(tmp_path / "gray.png", mode="L")
        ds = ImageFolderDataset(tmp_path)
        img, _, _ = ds[0]
        assert img.shape == (3, 8, 8)

    def test_rgba_image(self, tmp_path: Path) -> None:
        """RGBA image drops alpha channel."""
        from dataeval_flow.dataset import ImageFolderDataset

        _create_image(tmp_path / "rgba.png", mode="RGBA")
        ds = ImageFolderDataset(tmp_path)
        img, _, _ = ds[0]
        assert img.shape == (3, 8, 8)

    def test_negative_indexing(self, tmp_path: Path) -> None:
        """Negative indices work correctly."""
        from dataeval_flow.dataset import ImageFolderDataset

        for name in ["a.png", "b.png", "c.png"]:
            _create_image(tmp_path / name)

        ds = ImageFolderDataset(tmp_path)
        _, _, meta_last = ds[-1]
        assert meta_last["filename"] == "c.png"
        _, _, meta_first = ds[-3]
        assert meta_first["filename"] == "a.png"
        with pytest.raises(IndexError):
            ds[-4]

    def test_recursive(self, tmp_path: Path) -> None:
        """recursive=True finds images in subdirectories."""
        from dataeval_flow.dataset import ImageFolderDataset

        _create_image(tmp_path / "top.png")
        _create_image(tmp_path / "sub" / "deep.png")

        ds_flat = ImageFolderDataset(tmp_path, recursive=False)
        assert len(ds_flat) == 1

        ds_recursive = ImageFolderDataset(tmp_path, recursive=True)
        assert len(ds_recursive) == 2


# ---------------------------------------------------------------------------
# ImageFolderDataset — labeled
# ---------------------------------------------------------------------------


@pytest.mark.required
class TestImageFolderDatasetLabeled:
    """Tests for ImageFolderDataset in labeled mode (infer_labels=True)."""

    def test_happy_path(self, tmp_path: Path) -> None:
        """2 subdirs → 2 classes, one-hot targets, correct index2label."""
        from dataeval_flow.dataset import ImageFolderDataset

        _create_image(tmp_path / "cats" / "c1.png")
        _create_image(tmp_path / "cats" / "c2.png")
        _create_image(tmp_path / "dogs" / "d1.png")
        _create_image(tmp_path / "dogs" / "d2.png")

        ds = ImageFolderDataset(tmp_path, infer_labels=True)
        assert len(ds) == 4
        assert ds.metadata["index2label"] == {0: "cats", 1: "dogs"}

        # First two images are cats (class 0)
        _, target0, _ = ds[0]
        assert target0.dtype == np.float32
        assert target0.tolist() == [1.0, 0.0]

        # Last two images are dogs (class 1)
        _, target3, _ = ds[3]
        assert target3.tolist() == [0.0, 1.0]

    def test_empty_subdir_skipped(self, tmp_path: Path) -> None:
        """Empty subdirs are skipped; class indices remain dense."""
        from dataeval_flow.dataset import ImageFolderDataset

        _create_image(tmp_path / "alpha" / "a1.png")
        (tmp_path / "beta").mkdir()  # empty
        _create_image(tmp_path / "gamma" / "g1.png")

        ds = ImageFolderDataset(tmp_path, infer_labels=True)
        assert len(ds) == 2
        assert ds.metadata["index2label"] == {0: "alpha", 1: "gamma"}

    def test_top_level_images_ignored(self, tmp_path: Path) -> None:
        """Images directly in root are ignored in labeled mode."""
        from dataeval_flow.dataset import ImageFolderDataset

        _create_image(tmp_path / "stray.png")  # top-level — ignored
        _create_image(tmp_path / "cats" / "c1.png")

        ds = ImageFolderDataset(tmp_path, infer_labels=True)
        assert len(ds) == 1
        _, _, meta = ds[0]
        assert meta["filename"] == "c1.png"

    def test_single_class(self, tmp_path: Path) -> None:
        """Single class → num_classes=1, one-hot [1.0]."""
        from dataeval_flow.dataset import ImageFolderDataset

        _create_image(tmp_path / "only_class" / "img.png")
        ds = ImageFolderDataset(tmp_path, infer_labels=True)
        assert len(ds) == 1
        _, target, _ = ds[0]
        assert target.tolist() == [1.0]

    def test_no_subdirs_raises(self, tmp_path: Path) -> None:
        """infer_labels=True on flat dir → FileNotFoundError."""
        from dataeval_flow.dataset import ImageFolderDataset

        _create_image(tmp_path / "img.png")
        with pytest.raises(FileNotFoundError, match="No class subdirectories"):
            ImageFolderDataset(tmp_path, infer_labels=True)

    def test_all_subdirs_empty_raises(self, tmp_path: Path) -> None:
        """infer_labels=True, subdirs exist but no images → FileNotFoundError."""
        from dataeval_flow.dataset import ImageFolderDataset

        (tmp_path / "empty_a").mkdir()
        (tmp_path / "empty_b").mkdir()
        with pytest.raises(FileNotFoundError, match="No supported image files"):
            ImageFolderDataset(tmp_path, infer_labels=True)


# ---------------------------------------------------------------------------
# Config schema — image_folder
# ---------------------------------------------------------------------------


@pytest.mark.required
class TestDatasetConfigImageFolder:
    """Tests for ImageFolderDatasetConfig."""

    def test_image_folder_format_accepted(self) -> None:
        from dataeval_flow.config import ImageFolderDatasetConfig

        cfg = ImageFolderDatasetConfig(name="test", path="/data")
        assert cfg.format == "image_folder"

    def test_recursive_field(self) -> None:
        from dataeval_flow.config import ImageFolderDatasetConfig

        cfg = ImageFolderDatasetConfig(name="test", path="/data", recursive=True)
        assert cfg.recursive is True

    def test_infer_labels_field(self) -> None:
        from dataeval_flow.config import ImageFolderDatasetConfig

        cfg = ImageFolderDatasetConfig(name="test", path="/data", infer_labels=True)
        assert cfg.infer_labels is True

    def test_defaults_backward_compat(self) -> None:
        """ImageFolderDatasetConfig defaults are safe."""
        from dataeval_flow.config import ImageFolderDatasetConfig

        cfg = ImageFolderDatasetConfig(name="test", path="/data")
        assert cfg.recursive is False
        assert cfg.infer_labels is False


@pytest.mark.required
class TestDatasetConfigNewFields:
    """Tests for COCO/YOLO config fields."""

    def test_coco_fields_parse(self) -> None:
        from dataeval_flow.config import CocoDatasetConfig

        cfg = CocoDatasetConfig(
            name="coco-ds", path="/data/coco", annotations_file="ann.json", images_dir="imgs", classes_file="cls.txt"
        )
        assert cfg.annotations_file == "ann.json"
        assert cfg.images_dir == "imgs"
        assert cfg.classes_file == "cls.txt"

    def test_yolo_fields_parse(self) -> None:
        from dataeval_flow.config import YoloDatasetConfig

        cfg = YoloDatasetConfig(
            name="yolo-ds", path="/data/yolo", images_dir="imgs", labels_dir="lbls", classes_file="cls.txt"
        )
        assert cfg.images_dir == "imgs"
        assert cfg.labels_dir == "lbls"
        assert cfg.classes_file == "cls.txt"

    def test_new_fields_default_to_none(self) -> None:
        from dataeval_flow.config import CocoDatasetConfig

        cfg = CocoDatasetConfig(name="test", path="/data")
        assert cfg.annotations_file is None
        assert cfg.images_dir is None
        assert cfg.classes_file is None


# ---------------------------------------------------------------------------
# COCO / YOLO fixture-based round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.required
class TestCocoDatasetFixture:
    """Test COCO dataset loading with real fixtures."""

    def test_load_dataset_coco_with_fixtures(self, tmp_path: Path) -> None:
        """Round-trip: create tiny COCO dataset, load via load_dataset_coco."""
        import json

        from dataeval_flow.dataset import load_dataset_coco

        # Create a minimal 1x1 PNG image
        _create_image(tmp_path / "images" / "img_0.png", width=1, height=1)

        # Minimal COCO annotations JSON
        annotations = {
            "images": [{"id": 0, "file_name": "img_0.png", "width": 1, "height": 1}],
            "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0}],
            "categories": [{"id": 0, "name": "cat"}],
        }
        ann_path = tmp_path / "annotations.json"
        ann_path.write_text(json.dumps(annotations))

        # classes.txt
        (tmp_path / "classes.txt").write_text("cat\n")

        ds = load_dataset_coco(tmp_path)
        assert len(ds) == 1
        img, target, meta = ds[0]
        assert len(img.shape) == 3  # CHW array
        assert img.shape[0] == 3  # RGB channels
        assert hasattr(target, "boxes")  # OD target has bounding boxes
        assert isinstance(meta, dict)


@pytest.mark.required
class TestYoloDatasetFixture:
    """Test YOLO dataset loading with real fixtures."""

    def test_load_dataset_yolo_with_fixtures(self, tmp_path: Path) -> None:
        """Round-trip: create tiny YOLO dataset, load via load_dataset_yolo."""
        from dataeval_flow.dataset import load_dataset_yolo

        # Create a minimal 1x1 PNG image
        _create_image(tmp_path / "images" / "img_0.png", width=1, height=1)

        # YOLO label file (class_id cx cy w h, normalized)
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "img_0.txt").write_text("0 0.5 0.5 1.0 1.0\n")

        # classes.txt
        (tmp_path / "classes.txt").write_text("cat\n")

        ds = load_dataset_yolo(tmp_path)
        assert len(ds) == 1
        img, target, meta = ds[0]
        assert len(img.shape) == 3  # CHW array
        assert img.shape[0] == 3  # RGB channels
        assert hasattr(target, "boxes")  # OD target has bounding boxes
        assert isinstance(meta, dict)


# ---------------------------------------------------------------------------
# TorchvisionDataset — classification
# ---------------------------------------------------------------------------


def _make_cls_dataset(num_samples: int = 10, classes: list[str] | None = None):
    """Create a fake torchvision classification dataset."""

    class _FakeCls:
        def __init__(self):
            self.classes = classes

        def __len__(self):
            return num_samples

        def __getitem__(self, i):
            from PIL import Image

            rng = np.random.RandomState(i)
            img = Image.fromarray(rng.randint(0, 255, (16, 16, 3), dtype=np.uint8))
            return img, i % max(len(classes), 1) if classes else i

    return _FakeCls()


def _make_bboxes(
    data: Any,
    format: Any,  # noqa: A002
    canvas_size: tuple[int, int],
) -> BoundingBoxes:
    """Typed wrapper around BoundingBoxes() to satisfy pyright."""
    return BoundingBoxes(data, format=format, canvas_size=canvas_size)  # pyright: ignore[reportCallIssue]


def _make_od_dataset(num_samples: int = 5, classes: list[str] | None = None, box_format: str = "XYXY"):
    """Create a fake torchvision v2-wrapped object detection dataset."""
    import torch
    from torchvision.tv_tensors import BoundingBoxFormat

    fmt = BoundingBoxFormat[box_format]

    class _FakeOD:
        def __init__(self):
            self.classes = classes

        def __len__(self):
            return num_samples

        def __getitem__(self, i):
            from PIL import Image

            rng = np.random.RandomState(i)
            img = Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            n_boxes = rng.randint(1, 4)
            raw_boxes = rng.randint(1, 50, (n_boxes, 4)).astype(np.float32)
            boxes = _make_bboxes(torch.tensor(raw_boxes), format=fmt, canvas_size=(64, 64))
            labels = torch.tensor(rng.randint(0, max(len(classes), 2) if classes else 2, n_boxes))
            return img, {"boxes": boxes, "labels": labels}

    return _FakeOD()


@pytest.mark.required
class TestTorchvisionDatasetClassification:
    """Tests for TorchvisionDataset with classification datasets."""

    def test_len(self) -> None:
        """Adapter preserves dataset length."""
        from dataeval_flow.dataset import TorchvisionDataset

        ds = TorchvisionDataset(_make_cls_dataset(num_samples=7, classes=["a", "b"]))
        assert len(ds) == 7

    def test_getitem_returns_three_tuple(self) -> None:
        """__getitem__ returns (image, target, metadata) tuple."""
        from dataeval_flow.dataset import TorchvisionDataset

        ds = TorchvisionDataset(_make_cls_dataset(classes=["a", "b"]))
        result = ds[0]
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_image_chw_float32(self) -> None:
        """Image is converted to CHW float32 numpy array."""
        from dataeval_flow.dataset import TorchvisionDataset

        ds = TorchvisionDataset(_make_cls_dataset(classes=["a"]))
        img, _, _ = ds[0]
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.float32
        assert img.shape == (3, 16, 16)

    def test_one_hot_target_with_classes(self) -> None:
        """Integer label → one-hot vector when classes is available."""
        from dataeval_flow.dataset import TorchvisionDataset

        ds = TorchvisionDataset(_make_cls_dataset(num_samples=4, classes=["cat", "dog", "bird"]))
        # sample 0 → label 0 → [1, 0, 0]
        _, tgt0, _ = ds[0]
        assert tgt0.shape == (3,)
        assert tgt0.dtype == np.float32
        assert tgt0[0] == 1.0
        assert tgt0[1] == 0.0
        assert tgt0[2] == 0.0

        # sample 1 → label 1 → [0, 1, 0]
        _, tgt1, _ = ds[1]
        assert tgt1[1] == 1.0

    def test_scalar_target_without_classes(self) -> None:
        """Without classes, target is passed through as float32 array."""
        from dataeval_flow.dataset import TorchvisionDataset

        ds = TorchvisionDataset(_make_cls_dataset(classes=None))
        _, tgt, _ = ds[0]
        assert tgt.dtype == np.float32

    def test_metadata_has_id(self) -> None:
        """Datum metadata contains the sample index."""
        from dataeval_flow.dataset import TorchvisionDataset

        ds = TorchvisionDataset(_make_cls_dataset(classes=["a"]))
        _, _, meta = ds[3]
        assert meta["id"] == 3

    def test_dataset_metadata_index2label(self) -> None:
        """Dataset-level metadata exposes index2label from classes."""
        from dataeval_flow.dataset import TorchvisionDataset

        ds = TorchvisionDataset(_make_cls_dataset(classes=["airplane", "ship"]))
        m = ds.metadata
        assert m["index2label"] == {0: "airplane", 1: "ship"}  # type: ignore
        assert isinstance(m["id"], str)

    def test_dataset_metadata_empty_without_classes(self) -> None:
        """index2label is empty when dataset has no classes attribute."""
        from dataeval_flow.dataset import TorchvisionDataset

        ds = TorchvisionDataset(_make_cls_dataset(classes=None))
        assert ds.metadata["index2label"] == {}  # type: ignore

    def test_torch_tensor_image(self) -> None:
        """Adapter handles torch Tensor images (CHW)."""
        import torch

        from dataeval_flow.dataset import TorchvisionDataset

        class _TensorDs:
            classes = ["a", "b"]

            def __len__(self):
                return 1

            def __getitem__(self, i):
                return torch.randn(3, 32, 32), 0

        ds = TorchvisionDataset(_TensorDs())
        img, _, _ = ds[0]
        assert img.shape == (3, 32, 32)
        assert img.dtype == np.float32

    def test_hwc_tensor_image(self) -> None:
        """Adapter transposes HWC tensors to CHW."""
        import torch

        from dataeval_flow.dataset import TorchvisionDataset

        class _HWCDs:
            classes = ["a"]

            def __len__(self):
                return 1

            def __getitem__(self, i):
                return torch.randn(32, 32, 3), 0  # HWC

        ds = TorchvisionDataset(_HWCDs())
        img, _, _ = ds[0]
        assert img.shape == (3, 32, 32)

    def test_grayscale_pil_image(self) -> None:
        """Grayscale PIL image is converted to 3-channel RGB CHW."""
        from PIL import Image

        from dataeval_flow.dataset import TorchvisionDataset

        class _GrayDs:
            classes = ["a"]

            def __len__(self):
                return 1

            def __getitem__(self, i):
                return Image.new("L", (8, 8)), 0

        ds = TorchvisionDataset(_GrayDs())
        img, _, _ = ds[0]
        assert img.shape == (3, 8, 8)


# ---------------------------------------------------------------------------
# TorchvisionDataset — object detection
# ---------------------------------------------------------------------------


@pytest.mark.required
class TestTorchvisionDatasetObjectDetection:
    """Tests for TorchvisionDataset with object detection datasets."""

    def test_od_target_type(self) -> None:
        """OD targets conform to ObjectDetectionTarget protocol."""
        from dataeval.protocols import ObjectDetectionTarget

        from dataeval_flow.dataset import TorchvisionDataset

        ds = TorchvisionDataset(_make_od_dataset(classes=["a", "b"]))
        _, tgt, _ = ds[0]
        assert isinstance(tgt, ObjectDetectionTarget)

    def test_od_target_has_boxes_labels_scores(self) -> None:
        """OD target exposes boxes, labels, and scores as numpy arrays."""
        from dataeval_flow.dataset import TorchvisionDataset

        ds = TorchvisionDataset(_make_od_dataset(classes=["a", "b"]))
        _, tgt, _ = ds[0]
        assert isinstance(tgt.boxes, np.ndarray)
        assert isinstance(tgt.labels, np.ndarray)
        assert isinstance(tgt.scores, np.ndarray)

    def test_od_boxes_shape(self) -> None:
        """Boxes are (N, 4) float32."""
        from dataeval_flow.dataset import TorchvisionDataset

        ds = TorchvisionDataset(_make_od_dataset(classes=["a", "b"]))
        _, tgt, _ = ds[0]
        assert tgt.boxes.ndim == 2
        assert tgt.boxes.shape[1] == 4
        assert tgt.boxes.dtype == np.float32

    def test_od_scores_one_hot(self) -> None:
        """Scores are one-hot with correct shape (N, num_classes)."""
        from dataeval_flow.dataset import TorchvisionDataset

        ds = TorchvisionDataset(_make_od_dataset(classes=["cat", "dog", "bird"]))
        _, tgt, _ = ds[0]
        n_boxes = len(tgt.labels)
        assert tgt.scores.shape == (n_boxes, 3)
        # Each row sums to 1.0 (one-hot)
        np.testing.assert_array_equal(tgt.scores.sum(axis=1), np.ones(n_boxes))

    def test_xywh_converted_to_xyxy(self) -> None:
        """XYWH bounding boxes are converted to XYXY."""
        import torch
        from torchvision.tv_tensors import BoundingBoxFormat

        from dataeval_flow.dataset import TorchvisionDataset

        class _XYWHDs:
            classes = ["a"]

            def __len__(self):
                return 1

            def __getitem__(self, i):
                from PIL import Image

                img = Image.new("RGB", (100, 100))
                # XYWH (10, 20, 30, 40) → XYXY (10, 20, 40, 60)
                boxes = _make_bboxes(
                    torch.tensor([[10.0, 20.0, 30.0, 40.0]]), format=BoundingBoxFormat.XYWH, canvas_size=(100, 100)
                )
                return img, {"boxes": boxes, "labels": torch.tensor([0])}

        ds = TorchvisionDataset(_XYWHDs())
        _, tgt, _ = ds[0]
        np.testing.assert_allclose(tgt.boxes[0], [10.0, 20.0, 40.0, 60.0])

    def test_cxcywh_converted_to_xyxy(self) -> None:
        """CXCYWH bounding boxes are converted to XYXY."""
        import torch
        from torchvision.tv_tensors import BoundingBoxFormat

        from dataeval_flow.dataset import TorchvisionDataset

        class _CXCYWHDs:
            classes = ["a"]

            def __len__(self):
                return 1

            def __getitem__(self, i):
                from PIL import Image

                img = Image.new("RGB", (100, 100))
                # CXCYWH (50, 50, 20, 30) → XYXY (40, 35, 60, 65)
                boxes = _make_bboxes(
                    torch.tensor([[50.0, 50.0, 20.0, 30.0]]), format=BoundingBoxFormat.CXCYWH, canvas_size=(100, 100)
                )
                return img, {"boxes": boxes, "labels": torch.tensor([0])}

        ds = TorchvisionDataset(_CXCYWHDs())
        _, tgt, _ = ds[0]
        np.testing.assert_allclose(tgt.boxes[0], [40.0, 35.0, 60.0, 65.0])

    def test_xyxy_passed_through(self) -> None:
        """XYXY bounding boxes are not modified."""
        import torch
        from torchvision.tv_tensors import BoundingBoxFormat

        from dataeval_flow.dataset import TorchvisionDataset

        class _XYXYDs:
            classes = ["a"]

            def __len__(self):
                return 1

            def __getitem__(self, i):
                from PIL import Image

                img = Image.new("RGB", (100, 100))
                boxes = _make_bboxes(
                    torch.tensor([[10.0, 20.0, 50.0, 60.0]]), format=BoundingBoxFormat.XYXY, canvas_size=(100, 100)
                )
                return img, {"boxes": boxes, "labels": torch.tensor([0])}

        ds = TorchvisionDataset(_XYXYDs())
        _, tgt, _ = ds[0]
        np.testing.assert_allclose(tgt.boxes[0], [10.0, 20.0, 50.0, 60.0])

    def test_plain_tensor_boxes(self) -> None:
        """Plain torch tensors (no BoundingBoxes) are assumed XYXY."""
        import torch

        from dataeval_flow.dataset import TorchvisionDataset

        class _PlainDs:
            classes = ["a"]

            def __len__(self):
                return 1

            def __getitem__(self, i):
                from PIL import Image

                img = Image.new("RGB", (100, 100))
                boxes = torch.tensor([[10.0, 20.0, 50.0, 60.0]])
                return img, {"boxes": boxes, "labels": torch.tensor([0])}

        ds = TorchvisionDataset(_PlainDs())
        _, tgt, _ = ds[0]
        np.testing.assert_allclose(tgt.boxes[0], [10.0, 20.0, 50.0, 60.0])

    def test_od_num_classes_from_labels_when_no_classes_attr(self) -> None:
        """Without .classes, num_classes is inferred from max label."""
        import torch
        from torchvision.tv_tensors import BoundingBoxFormat

        from dataeval_flow.dataset import TorchvisionDataset

        class _NoClassesDs:
            def __len__(self):
                return 1

            def __getitem__(self, i):
                from PIL import Image

                img = Image.new("RGB", (64, 64))
                boxes = _make_bboxes(
                    torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 20.0, 20.0]]),
                    format=BoundingBoxFormat.XYXY,
                    canvas_size=(64, 64),
                )
                return img, {"boxes": boxes, "labels": torch.tensor([0, 2])}

        ds = TorchvisionDataset(_NoClassesDs())
        _, tgt, _ = ds[0]
        # max label is 2 → num_classes = 3
        assert tgt.scores.shape == (2, 3)


# ---------------------------------------------------------------------------
# _ObjectDetectionTarget
# ---------------------------------------------------------------------------


@pytest.mark.required
class TestObjectDetectionTarget:
    """Tests for the _ObjectDetectionTarget helper class."""

    def test_protocol_conformance(self) -> None:
        """_ObjectDetectionTarget satisfies the ObjectDetectionTarget protocol."""
        from dataeval.protocols import ObjectDetectionTarget

        from dataeval_flow.dataset import _ObjectDetectionTarget

        t = _ObjectDetectionTarget(
            boxes=np.zeros((2, 4), dtype=np.float32),
            labels=np.array([0, 1], dtype=np.intp),
            scores=np.eye(2, dtype=np.float32),
        )
        assert isinstance(t, ObjectDetectionTarget)

    def test_properties(self) -> None:
        """Properties return the arrays passed at construction."""
        from dataeval_flow.dataset import _ObjectDetectionTarget

        boxes = np.array([[1, 2, 3, 4]], dtype=np.float32)
        labels = np.array([0], dtype=np.intp)
        scores = np.array([[1.0, 0.0]], dtype=np.float32)
        t = _ObjectDetectionTarget(boxes=boxes, labels=labels, scores=scores)
        np.testing.assert_array_equal(t.boxes, boxes)
        np.testing.assert_array_equal(t.labels, labels)
        np.testing.assert_array_equal(t.scores, scores)


# ---------------------------------------------------------------------------
# DatasetProtocolConfig with format="torchvision"
# ---------------------------------------------------------------------------


@pytest.mark.required
class TestDatasetProtocolConfigTorchvision:
    """Tests for DatasetProtocolConfig with torchvision format."""

    def test_format_torchvision_accepted(self) -> None:
        """format='torchvision' is a valid literal."""
        from dataeval_flow.config import DatasetProtocolConfig

        cfg = DatasetProtocolConfig(name="test", format="torchvision", dataset=_make_cls_dataset(classes=["a"]))
        assert cfg.format == "torchvision"

    def test_format_maite_still_works(self) -> None:
        """format='maite' (default) is still valid."""
        from dataeval_flow.config import DatasetProtocolConfig

        cfg = DatasetProtocolConfig(name="test", dataset=MagicMock())
        assert cfg.format == "maite"

    def test_resolve_torchvision_wraps_dataset(self) -> None:
        """resolve_dataset wraps torchvision datasets in TorchvisionDataset."""
        from dataeval_flow.config import DatasetProtocolConfig
        from dataeval_flow.dataset import TorchvisionDataset, resolve_dataset

        cfg = DatasetProtocolConfig(name="tv-test", format="torchvision", dataset=_make_cls_dataset(classes=["a", "b"]))
        resolved = resolve_dataset(cfg)
        assert isinstance(resolved.dataset, TorchvisionDataset)
        assert resolved.name == "tv-test"
        assert resolved.label_source == "torchvision"
        assert resolved.cache_key == "tv-test:torchvision:1"

    def test_resolve_maite_passes_through(self) -> None:
        """resolve_dataset does not wrap maite datasets."""
        from dataeval_flow.config import DatasetProtocolConfig
        from dataeval_flow.dataset import TorchvisionDataset, resolve_dataset

        raw = MagicMock()
        cfg = DatasetProtocolConfig(name="m-test", format="maite", dataset=raw)
        resolved = resolve_dataset(cfg)
        assert resolved.dataset is raw
        assert not isinstance(resolved.dataset, TorchvisionDataset)

    def test_resolve_cache_key_includes_version(self) -> None:
        """cache_key changes when version changes."""
        from dataeval_flow.config import DatasetProtocolConfig
        from dataeval_flow.dataset import resolve_dataset

        cfg1 = DatasetProtocolConfig(
            name="ds", format="torchvision", dataset=_make_cls_dataset(classes=["a"]), version="1"
        )
        cfg2 = DatasetProtocolConfig(
            name="ds", format="torchvision", dataset=_make_cls_dataset(classes=["a"]), version="2"
        )
        assert resolve_dataset(cfg1).cache_key != resolve_dataset(cfg2).cache_key


# ---------------------------------------------------------------------------
# load_dataset_torchvision
# ---------------------------------------------------------------------------


@pytest.mark.required
class TestLoadDatasetTorchvision:
    """Tests for the load_dataset_torchvision convenience function."""

    def test_returns_torchvision_dataset(self) -> None:
        from dataeval_flow.dataset import TorchvisionDataset, load_dataset_torchvision

        result = load_dataset_torchvision(_make_cls_dataset(classes=["a"]))
        assert isinstance(result, TorchvisionDataset)

    def test_preserves_dataset_data(self) -> None:
        from dataeval_flow.dataset import load_dataset_torchvision

        raw = _make_cls_dataset(num_samples=3, classes=["x", "y"])
        ds = load_dataset_torchvision(raw)
        assert len(ds) == 3
        assert ds.metadata["index2label"] == {0: "x", 1: "y"}  # type: ignore


@pytest.mark.required
class TestMainModule:
    """Test the __main__.py CLI module."""

    def test_parse_args_no_output_exits_error(self) -> None:
        """--output is required; omitting it causes argparse to exit with error."""
        from dataeval_flow.__main__ import parse_args

        with patch("sys.argv", ["dataeval_flow"]), pytest.raises(SystemExit) as exc_info:
            parse_args()
        assert exc_info.value.code == 2

    def test_parse_args_with_config(self) -> None:
        """Test parsing with --config flag."""
        from dataeval_flow.__main__ import parse_args

        with patch("sys.argv", ["dataeval_flow", "--config", "/my/config", "--output", "/my/output"]):
            args = parse_args()
            assert args.config == Path("/my/config")

    def test_main_success(self) -> None:
        """Test main() with successful execution."""
        from dataeval_flow.__main__ import main

        with (
            patch("sys.argv", ["dataeval_flow", "--output", "/fake/output"]),
            patch("dataeval_flow.runner.run", return_value=0),
            pytest.raises(SystemExit) as exc_info,
        ):
            main()
        assert exc_info.value.code == 0

    def test_main_file_not_found(self) -> None:
        """Test main() with FileNotFoundError."""
        from dataeval_flow.__main__ import main

        with (
            patch("sys.argv", ["dataeval_flow", "--output", "/fake/output"]),
            patch("dataeval_flow.runner.run", side_effect=FileNotFoundError("Not found")),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


@pytest.mark.optional
class TestResultMetadata:
    """Test the ResultMetadata schema [IR-3-H-12]."""

    def test_defaults(self) -> None:
        """ResultMetadata has sensible defaults for JATIC fields."""
        from dataeval_flow.config import ResultMetadata

        meta = ResultMetadata()
        assert meta.version == "1.0"
        assert meta.tool == "dataeval-flow"
        assert meta.timestamp is not None

    def test_serializes_to_json(self) -> None:
        """model_dump(mode='json') produces JSON-safe types."""
        from dataeval_flow.config import ResultMetadata

        meta = ResultMetadata(dataset_id="cifar10", tool_version="0.1.0")
        data = meta.model_dump(mode="json")
        assert data["dataset_id"] == "cifar10"
        assert data["tool"] == "dataeval-flow"
        assert isinstance(data["timestamp"], str)


@pytest.mark.optional
class TestContainerRun:
    """Test the container landing script."""

    def test_main_config_not_found(self) -> None:
        """Test main() when config path doesn't exist."""
        from container_run import main

        result = main()
        assert result == 1
