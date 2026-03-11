"""Tests for the dataeval_app package."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.mark.required
class TestImports:
    """Test that core imports work correctly."""

    def test_dataeval_import(self) -> None:
        """Verify DataEval can be imported successfully."""
        import dataeval

        assert dataeval is not None

    def test_package_import(self) -> None:
        """Verify the dataeval_app package can be imported."""
        import dataeval_app

        assert hasattr(dataeval_app, "__version__")

    def test_public_api_exports(self) -> None:
        """Verify __all__ exports are available from package root."""
        from dataeval_app import load_dataset, load_dataset_huggingface

        assert callable(load_dataset)
        assert callable(load_dataset_huggingface)

    def test_workflow_api_exports(self) -> None:
        """Verify workflow exports are available from package root."""
        from dataeval_app import get_workflow, list_workflows, run_task

        assert callable(get_workflow)
        assert callable(list_workflows)
        assert callable(run_task)


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

            from dataeval_app import load_dataset_huggingface

            load_dataset_huggingface(Path("dummy/path"), split="train")

            # Should use the explicitly requested split
            mock_hf_dataset.__getitem__.assert_called_with("train")
            mock_adapter.assert_called()

    def test_load_dataset_delegates(self) -> None:
        """Test load_dataset defaults to huggingface."""
        with patch("dataeval_app.dataset.load_dataset_huggingface") as mock_hf:
            mock_hf.return_value = MagicMock()

            from dataeval_app.dataset import load_dataset

            load_dataset(Path("/some/path"))
            mock_hf.assert_called_once_with(Path("/some/path"), split=None)

    def test_load_dataset_image_folder_dispatch(self) -> None:
        """Test load_dataset dispatches to image_folder loader."""
        with patch("dataeval_app.dataset.load_dataset_image_folder") as mock_if:
            mock_if.return_value = MagicMock()

            from dataeval_app.dataset import load_dataset

            load_dataset(Path("/some/path"), dataset_format="image_folder")
            mock_if.assert_called_once_with(Path("/some/path"), recursive=False, infer_labels=False)

    def test_load_dataset_image_folder_with_infer_labels(self) -> None:
        """Test load_dataset passes infer_labels through."""
        with patch("dataeval_app.dataset.load_dataset_image_folder") as mock_if:
            mock_if.return_value = MagicMock()

            from dataeval_app.dataset import load_dataset

            load_dataset(Path("/some/path"), dataset_format="image_folder", infer_labels=True)
            mock_if.assert_called_once_with(Path("/some/path"), recursive=False, infer_labels=True)

    def test_load_dataset_unsupported_format_raises(self) -> None:
        """Test load_dataset raises ValueError for unsupported formats."""
        from dataeval_app.dataset import load_dataset

        with pytest.raises(ValueError, match="Unsupported dataset format"):
            load_dataset(Path("/some/path"), dataset_format="unknown")  # type: ignore[arg-type]

    def test_load_dataset_coco_dispatch(self) -> None:
        """Test load_dataset dispatches to COCO loader."""
        with patch("dataeval_app.dataset.load_dataset_coco") as mock_coco:
            mock_coco.return_value = MagicMock()

            from dataeval_app.dataset import load_dataset

            load_dataset(
                Path("/some/path"),
                dataset_format="coco",
                annotations_file="ann.json",
                images_dir="imgs",
                classes_file="cls.txt",
            )
            mock_coco.assert_called_once_with(
                Path("/some/path"),
                annotations_file="ann.json",
                images_dir="imgs",
                classes_file="cls.txt",
            )

    def test_load_dataset_yolo_dispatch(self) -> None:
        """Test load_dataset dispatches to YOLO loader."""
        with patch("dataeval_app.dataset.load_dataset_yolo") as mock_yolo:
            mock_yolo.return_value = MagicMock()

            from dataeval_app.dataset import load_dataset

            load_dataset(
                Path("/some/path"),
                dataset_format="yolo",
                images_dir="imgs",
                labels_dir="lbls",
                classes_file="cls.txt",
            )
            mock_yolo.assert_called_once_with(
                Path("/some/path"),
                images_dir="imgs",
                labels_dir="lbls",
                classes_file="cls.txt",
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
        from dataeval_app.dataset import ImageFolderDataset

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
        from dataeval_app.dataset import ImageFolderDataset

        with pytest.raises(FileNotFoundError, match="Image folder not found"):
            ImageFolderDataset(tmp_path / "nonexistent")

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        """Empty directory raises FileNotFoundError."""
        from dataeval_app.dataset import ImageFolderDataset

        with pytest.raises(FileNotFoundError, match="No supported image files"):
            ImageFolderDataset(tmp_path)

    def test_mixed_files_filters(self, tmp_path: Path) -> None:
        """Only image files are discovered; others are ignored."""
        from dataeval_app.dataset import ImageFolderDataset

        _create_image(tmp_path / "img.png")
        (tmp_path / "data.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("a,b")

        ds = ImageFolderDataset(tmp_path)
        assert len(ds) == 1

    def test_grayscale_image(self, tmp_path: Path) -> None:
        """Grayscale image is converted to 3-channel RGB."""
        from dataeval_app.dataset import ImageFolderDataset

        _create_image(tmp_path / "gray.png", mode="L")
        ds = ImageFolderDataset(tmp_path)
        img, _, _ = ds[0]
        assert img.shape == (3, 8, 8)

    def test_rgba_image(self, tmp_path: Path) -> None:
        """RGBA image drops alpha channel."""
        from dataeval_app.dataset import ImageFolderDataset

        _create_image(tmp_path / "rgba.png", mode="RGBA")
        ds = ImageFolderDataset(tmp_path)
        img, _, _ = ds[0]
        assert img.shape == (3, 8, 8)

    def test_negative_indexing(self, tmp_path: Path) -> None:
        """Negative indices work correctly."""
        from dataeval_app.dataset import ImageFolderDataset

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
        from dataeval_app.dataset import ImageFolderDataset

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
        from dataeval_app.dataset import ImageFolderDataset

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
        from dataeval_app.dataset import ImageFolderDataset

        _create_image(tmp_path / "alpha" / "a1.png")
        (tmp_path / "beta").mkdir()  # empty
        _create_image(tmp_path / "gamma" / "g1.png")

        ds = ImageFolderDataset(tmp_path, infer_labels=True)
        assert len(ds) == 2
        assert ds.metadata["index2label"] == {0: "alpha", 1: "gamma"}

    def test_top_level_images_ignored(self, tmp_path: Path) -> None:
        """Images directly in root are ignored in labeled mode."""
        from dataeval_app.dataset import ImageFolderDataset

        _create_image(tmp_path / "stray.png")  # top-level — ignored
        _create_image(tmp_path / "cats" / "c1.png")

        ds = ImageFolderDataset(tmp_path, infer_labels=True)
        assert len(ds) == 1
        _, _, meta = ds[0]
        assert meta["filename"] == "c1.png"

    def test_single_class(self, tmp_path: Path) -> None:
        """Single class → num_classes=1, one-hot [1.0]."""
        from dataeval_app.dataset import ImageFolderDataset

        _create_image(tmp_path / "only_class" / "img.png")
        ds = ImageFolderDataset(tmp_path, infer_labels=True)
        assert len(ds) == 1
        _, target, _ = ds[0]
        assert target.tolist() == [1.0]

    def test_no_subdirs_raises(self, tmp_path: Path) -> None:
        """infer_labels=True on flat dir → FileNotFoundError."""
        from dataeval_app.dataset import ImageFolderDataset

        _create_image(tmp_path / "img.png")
        with pytest.raises(FileNotFoundError, match="No class subdirectories"):
            ImageFolderDataset(tmp_path, infer_labels=True)

    def test_all_subdirs_empty_raises(self, tmp_path: Path) -> None:
        """infer_labels=True, subdirs exist but no images → FileNotFoundError."""
        from dataeval_app.dataset import ImageFolderDataset

        (tmp_path / "empty_a").mkdir()
        (tmp_path / "empty_b").mkdir()
        with pytest.raises(FileNotFoundError, match="No supported image files"):
            ImageFolderDataset(tmp_path, infer_labels=True)


# ---------------------------------------------------------------------------
# Config schema — image_folder
# ---------------------------------------------------------------------------


@pytest.mark.required
class TestDatasetConfigImageFolder:
    """Tests for DatasetConfig with image_folder format."""

    def test_image_folder_format_accepted(self) -> None:
        from dataeval_app.config.schemas.dataset import DatasetConfig

        cfg = DatasetConfig(name="test", format="image_folder", path="/data")
        assert cfg.format == "image_folder"

    def test_recursive_field(self) -> None:
        from dataeval_app.config.schemas.dataset import DatasetConfig

        cfg = DatasetConfig(name="test", format="image_folder", path="/data", recursive=True)
        assert cfg.recursive is True

    def test_infer_labels_field(self) -> None:
        from dataeval_app.config.schemas.dataset import DatasetConfig

        cfg = DatasetConfig(name="test", format="image_folder", path="/data", infer_labels=True)
        assert cfg.infer_labels is True

    def test_defaults_backward_compat(self) -> None:
        """Existing configs without new fields still parse."""
        from dataeval_app.config.schemas.dataset import DatasetConfig

        cfg = DatasetConfig(name="test", format="huggingface", path="/data")
        assert cfg.recursive is False
        assert cfg.infer_labels is False


@pytest.mark.required
class TestDatasetConfigNewFields:
    """Tests for COCO/YOLO config fields."""

    def test_coco_fields_parse(self) -> None:
        from dataeval_app.config.schemas.dataset import DatasetConfig

        cfg = DatasetConfig(
            name="coco-ds",
            format="coco",
            path="/data/coco",
            annotations_file="ann.json",
            images_dir="imgs",
            classes_file="cls.txt",
        )
        assert cfg.annotations_file == "ann.json"
        assert cfg.images_dir == "imgs"
        assert cfg.classes_file == "cls.txt"

    def test_yolo_fields_parse(self) -> None:
        from dataeval_app.config.schemas.dataset import DatasetConfig

        cfg = DatasetConfig(
            name="yolo-ds",
            format="yolo",
            path="/data/yolo",
            images_dir="imgs",
            labels_dir="lbls",
            classes_file="cls.txt",
        )
        assert cfg.images_dir == "imgs"
        assert cfg.labels_dir == "lbls"
        assert cfg.classes_file == "cls.txt"

    def test_new_fields_default_to_none(self) -> None:
        from dataeval_app.config.schemas.dataset import DatasetConfig

        cfg = DatasetConfig(name="test", format="huggingface", path="/data")
        assert cfg.annotations_file is None
        assert cfg.images_dir is None
        assert cfg.labels_dir is None
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

        from dataeval_app.dataset import load_dataset_coco

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
        from dataeval_app.dataset import load_dataset_yolo

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


@pytest.mark.required
class TestMainModule:
    """Test the __main__.py CLI module."""

    def test_parse_args_no_output_exits_error(self) -> None:
        """--output is required; omitting it causes argparse to exit with error."""
        from dataeval_app.__main__ import parse_args

        with patch("sys.argv", ["dataeval_app"]), pytest.raises(SystemExit) as exc_info:
            parse_args()
        assert exc_info.value.code == 2

    def test_parse_args_with_config(self) -> None:
        """Test parsing with --config flag."""
        from dataeval_app.__main__ import parse_args

        with patch("sys.argv", ["dataeval_app", "--config", "/my/config", "--output", "/my/output"]):
            args = parse_args()
            assert args.config == Path("/my/config")

    def test_main_success(self) -> None:
        """Test main() with successful execution."""
        from dataeval_app.__main__ import main

        with (
            patch("sys.argv", ["dataeval_app", "--output", "/fake/output"]),
            patch("dataeval_app.__main__._run_tasks"),
        ):
            main()

    def test_main_file_not_found(self) -> None:
        """Test main() with FileNotFoundError."""
        from dataeval_app.__main__ import main

        with (
            patch("sys.argv", ["dataeval_app", "--output", "/fake/output"]),
            patch("dataeval_app.__main__._run_tasks", side_effect=FileNotFoundError("Not found")),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


@pytest.mark.optional
class TestResultMetadata:
    """Test the ResultMetadata schema [IR-3-H-12]."""

    def test_defaults(self) -> None:
        """ResultMetadata has sensible defaults for JATIC fields."""
        from dataeval_app.config.schemas.metadata import ResultMetadata

        meta = ResultMetadata()
        assert meta.version == "1.0"
        assert meta.tool == "dataeval-app"
        assert meta.timestamp is not None

    def test_serializes_to_json(self) -> None:
        """model_dump(mode='json') produces JSON-safe types."""
        from dataeval_app.config.schemas.metadata import ResultMetadata

        meta = ResultMetadata(dataset_id="cifar10", tool_version="0.1.0")
        data = meta.model_dump(mode="json")
        assert data["dataset_id"] == "cifar10"
        assert data["tool"] == "dataeval-app"
        assert isinstance(data["timestamp"], str)


@pytest.mark.optional
class TestContainerRun:
    """Test the container landing script."""

    def test_get_dataset_path_default(self) -> None:
        """Test default container mount path."""
        from container_run import get_dataset_path

        with patch.dict("os.environ", {}, clear=True):
            path = get_dataset_path()
            assert path == Path("/data/dataset")

    def test_get_dataset_path_env_override(self) -> None:
        """Test environment variable override."""
        from container_run import get_dataset_path

        with patch.dict("os.environ", {"DATASET_PATH": "/custom/path"}):
            path = get_dataset_path()
            assert path == Path("/custom/path")

    def test_get_output_path_default(self) -> None:
        """Test default output mount path."""
        from container_run import get_output_path

        with patch.dict("os.environ", {}, clear=True):
            path = get_output_path()
            assert path == Path("/output")

    def test_get_output_path_env_override(self) -> None:
        """Test output path environment override."""
        from container_run import get_output_path

        with patch.dict("os.environ", {"OUTPUT_PATH": "/custom/output"}):
            path = get_output_path()
            assert path == Path("/custom/output")

    def test_main_config_not_found(self) -> None:
        """Test main() when config path doesn't exist."""
        from container_run import main

        result = main()
        assert result == 1
