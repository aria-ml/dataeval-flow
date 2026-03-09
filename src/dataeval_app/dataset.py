#!/usr/bin/env python3
"""Dataset loading utilities - standalone library functions.

All functions accept parameters. No hardcoded container paths.

This module provides functions for loading datasets in HuggingFace, image folder,
COCO, and YOLO formats, converting them to MAITE-compatible objects for evaluation.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from maite_datasets.adapters import HFImageClassificationDataset, HFObjectDetectionDataset
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dataeval_app.config.schemas.dataset import DatasetFormat

logger: logging.Logger = logging.getLogger(__name__)

MaiteDataset: TypeAlias = HFImageClassificationDataset | HFObjectDetectionDataset

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})


class ImageFolderDataset:
    """MAITE-compatible dataset backed by a directory of image files.

    Loads images lazily via PIL and converts to CHW numpy arrays. Returns
    3-tuples ``(image, target, datum_metadata)``.

    When ``infer_labels=False`` (default), target is an empty array (no
    labels). When ``infer_labels=True``, immediate child directories of
    ``root`` are treated as class names and target is a one-hot float32
    vector.
    """

    def __init__(self, root: Path, *, recursive: bool = False, infer_labels: bool = False) -> None:
        """Initialize dataset from a directory of image files."""
        self._root = root
        if not root.is_dir():
            raise FileNotFoundError(f"Image folder not found: {root}")

        if infer_labels:
            self._paths, self._labels, self._index2label = self._discover_labeled(root)
        else:
            self._paths = self._discover_unlabeled(root, recursive=recursive)
            self._labels: list[int] | None = None
            self._index2label: dict[int, str] = {}

        if not self._paths:
            raise FileNotFoundError(
                f"No supported image files found in {root}. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        logger.info("ImageFolderDataset: found %d images in %s", len(self._paths), root)

    # -- Discovery --------------------------------------------------------

    @staticmethod
    def _discover_unlabeled(root: Path, *, recursive: bool) -> list[Path]:
        """Flat or recursive image discovery (no labels)."""
        glob_pattern = "**/*" if recursive else "*"
        return sorted(p for p in root.glob(glob_pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)

    @staticmethod
    def _discover_labeled(
        root: Path,
    ) -> tuple[list[Path], list[int], dict[int, str]]:
        """Subdirectory-as-label discovery (torchvision ImageFolder convention).

        Immediate child directories of *root* become class names, sorted
        alphabetically and mapped to dense indices 0, 1, 2, ….  Images in
        each class directory are collected via ``rglob``.  Empty class
        directories are silently skipped.
        """
        class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
        if not class_dirs:
            raise FileNotFoundError(f"No class subdirectories found in {root}")

        # Log top-level images that will be ignored
        top_level_images = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
        if top_level_images:
            logger.debug(
                "ImageFolderDataset: ignoring %d top-level images in labeled mode",
                len(top_level_images),
            )

        paths: list[Path] = []
        labels: list[int] = []
        index2label: dict[int, str] = {}
        class_idx = 0
        for class_dir in class_dirs:
            class_images = sorted(
                p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            if not class_images:
                continue  # skip empty class dirs
            index2label[class_idx] = class_dir.name
            paths.extend(class_images)
            labels.extend([class_idx] * len(class_images))
            class_idx += 1

        return paths, labels, index2label

    # -- AnnotatedDataset protocol ----------------------------------------

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self._paths)

    def __getitem__(self, index: int) -> tuple[NDArray[Any], NDArray[Any], dict[str, Any]]:
        """Return (image, target, metadata) for the given index."""
        if index < 0:
            index += len(self._paths)
        if index < 0 or index >= len(self._paths):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")
        img_array = self._load_image(index)

        if self._labels is not None:
            # Labeled mode — one-hot target
            num_classes = len(self._index2label)
            target: NDArray[Any] = np.zeros(num_classes, dtype=np.float32)
            target[self._labels[index]] = 1.0
        else:
            # Unlabeled mode — empty target
            target = np.empty(0, dtype=np.intp)

        datum_metadata: dict[str, Any] = {
            "id": index,
            "filename": self._paths[index].name,
        }
        return img_array, target, datum_metadata

    @property
    def metadata(self) -> dict[str, Any]:
        """Dataset-level metadata (DatasetMetadata TypedDict shape)."""
        return {
            "id": 0,
            "index2label": dict(self._index2label),
        }

    # -- Internal ---------------------------------------------------------

    @lru_cache(maxsize=64)  # noqa: B019
    def _load_image(self, index: int) -> NDArray[Any]:
        """Load and convert a single image to CHW float32 numpy array."""
        from PIL import Image

        path = self._paths[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return np.transpose(
                np.array(img, dtype=np.float32),  # HWC → CHW
                (2, 0, 1),
            )


def load_dataset_huggingface(path: Path, split: str | None = None) -> MaiteDataset:
    """Load a HuggingFace dataset and convert to MAITE format.

    Parameters
    ----------
    path : Path
        Path to the dataset directory containing HuggingFace dataset files.

    Returns
    -------
    MaiteDataset
        MAITE-compatible dataset object that can be used with DataEval.

    Raises
    ------
    ImportError
        If required dependencies (datasets, maite_datasets) are missing.
    RuntimeError
        If dataset loading or conversion fails.
    KeyError
        If split is not specified for a multi-split dataset or the specified split is not found in the dataset.

    Examples
    --------
    >>> from pathlib import Path
    >>> from dataeval_app import load_dataset_huggingface
    >>> ds = load_dataset_huggingface(Path("/data/cifar10"))
    """
    from datasets import load_from_disk
    from maite_datasets.adapters import from_huggingface

    dataset = load_from_disk(str(path))

    logger.info("Loaded type: %s", type(dataset).__name__)
    # union-attr: load_from_disk returns Dataset | DatasetDict; .keys() only on DatasetDict.
    if hasattr(dataset, "keys") and callable(dataset.keys):  # type: ignore[union-attr]
        available_splits = list(dataset.keys())  # type: ignore[union-attr]
        if split is None or split not in available_splits:
            raise KeyError(f"Requested split '{split}' not found in dataset. Available splits: {available_splits}")
        logger.info("DatasetDict detected with %d splits: %s", len(available_splits), available_splits)
        logger.info("Selecting split '%s' as specified in config.", split)
        dataset = dataset[split]
    else:
        logger.info("Dataset detected (single split)")

    # load_from_disk returns Dataset | DatasetDict; after the dict guard above
    # it's a single Dataset, but pyright can't narrow through hasattr.
    return from_huggingface(dataset)  # type: ignore[arg-type]


def load_dataset_image_folder(path: Path, *, recursive: bool = False, infer_labels: bool = False) -> ImageFolderDataset:
    """Load an image folder dataset."""
    return ImageFolderDataset(path, recursive=recursive, infer_labels=infer_labels)


def load_dataset_coco(
    path: Path,
    *,
    annotations_file: str | None = None,
    images_dir: str | None = None,
    classes_file: str | None = None,
) -> Any:
    """Load a COCO-format object detection dataset.

    Parameters
    ----------
    path : Path
        Root directory of the COCO dataset.
    annotations_file : str | None
        Name of the annotations JSON file (default: reader's default).
    images_dir : str | None
        Name of the images subdirectory (default: reader's default).
    classes_file : str | None
        Name of the classes text file (default: reader's default).

    Returns
    -------
    Any
        MAITE-compatible object detection dataset (``COCODataset``).
    """
    from maite_datasets.object_detection import COCODatasetReader

    kwargs: dict[str, str] = {}
    if annotations_file is not None:
        kwargs["annotation_file"] = annotations_file  # reader uses singular
    if images_dir is not None:
        kwargs["images_dir"] = images_dir
    if classes_file is not None:
        kwargs["classes_file"] = classes_file
    reader = COCODatasetReader(path, **kwargs)
    dataset = reader.create_dataset()
    logger.info("COCODataset: loaded %d images from %s", len(dataset), path)
    return dataset


def load_dataset_yolo(
    path: Path,
    *,
    images_dir: str | None = None,
    labels_dir: str | None = None,
    classes_file: str | None = None,
) -> Any:
    """Load a YOLO-format object detection dataset.

    Parameters
    ----------
    path : Path
        Root directory of the YOLO dataset.
    images_dir : str | None
        Name of the images subdirectory (default: reader's default).
    labels_dir : str | None
        Name of the labels subdirectory (default: reader's default).
    classes_file : str | None
        Name of the classes text file (default: reader's default).

    Returns
    -------
    Any
        MAITE-compatible object detection dataset (``YOLODataset``).
    """
    from maite_datasets.object_detection import YOLODatasetReader

    kwargs: dict[str, str] = {}
    if images_dir is not None:
        kwargs["images_dir"] = images_dir
    if labels_dir is not None:
        kwargs["labels_dir"] = labels_dir
    if classes_file is not None:
        kwargs["classes_file"] = classes_file
    reader = YOLODatasetReader(path, **kwargs)  # type: ignore[arg-type]  # kwargs are all str; pyright flags image_extensions
    dataset = reader.create_dataset()
    logger.info("YOLODataset: loaded %d images from %s", len(dataset), path)
    return dataset


def load_dataset(
    path: Path,
    split: str | None = None,
    dataset_format: DatasetFormat = "huggingface",
    *,
    recursive: bool = False,
    infer_labels: bool = False,
    annotations_file: str | None = None,
    images_dir: str | None = None,
    labels_dir: str | None = None,
    classes_file: str | None = None,
) -> Any:
    """Load a dataset and convert to MAITE format.

    This is the main entry point for loading datasets. Dispatches to the
    appropriate loader based on ``dataset_format``.

    Parameters
    ----------
    path : Path
        Path to the dataset directory.
    split : str | None
        Optional split name to load (e.g. "train", "test").
    dataset_format : DatasetFormat
        Dataset format identifier (default ``"huggingface"``).
    recursive : bool
        Scan subdirectories for images (image_folder only).
    infer_labels : bool
        Treat subdirectories as class labels (image_folder only).
    annotations_file : str | None
        Annotations file name (COCO only).
    images_dir : str | None
        Images subdirectory name (COCO/YOLO only).
    labels_dir : str | None
        Labels subdirectory name (YOLO only).
    classes_file : str | None
        Classes file name (COCO/YOLO only).

    Returns
    -------
    MaiteDataset | ImageFolderDataset | Any
        MAITE-compatible dataset object.

    Raises
    ------
    KeyError
        If split is not specified for a multi-split dataset or the specified split is not found in the dataset.
    ValueError
        If the dataset format is not supported.

    Examples
    --------
    >>> from pathlib import Path
    >>> from dataeval_app import load_dataset
    >>> ds = load_dataset(Path("/data/cifar10"))
    """
    if dataset_format == "huggingface":
        return load_dataset_huggingface(path, split=split)
    if dataset_format == "image_folder":
        return load_dataset_image_folder(path, recursive=recursive, infer_labels=infer_labels)
    if dataset_format == "coco":
        return load_dataset_coco(
            path, annotations_file=annotations_file, images_dir=images_dir, classes_file=classes_file
        )
    if dataset_format == "yolo":
        return load_dataset_yolo(path, images_dir=images_dir, labels_dir=labels_dir, classes_file=classes_file)
    msg = f"Unsupported dataset format: {dataset_format!r}"
    raise ValueError(msg)
