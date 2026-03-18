#!/usr/bin/env python3
"""Dataset loading utilities - standalone library functions.

All functions accept parameters. No hardcoded container paths.

This module provides functions for loading datasets in HuggingFace, image folder,
COCO, YOLO, and torchvision formats, converting them to MAITE-compatible objects for evaluation.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from dataeval.protocols import AnnotatedDataset, DatasetMetadata
from maite_datasets.adapters import HFImageClassificationDataset, HFObjectDetectionDataset
from numpy.typing import NDArray
from pydantic import BaseModel

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


class _ObjectDetectionTarget:
    """Lightweight object-detection target conforming to :class:`~dataeval.protocols.ObjectDetectionTarget`."""

    __slots__ = ("_boxes", "_labels", "_scores")

    def __init__(self, boxes: NDArray[Any], labels: NDArray[Any], scores: NDArray[Any]) -> None:
        self._boxes = boxes
        self._labels = labels
        self._scores = scores

    @property
    def boxes(self) -> NDArray[Any]:
        """:class:`NDArray` of shape ``(N, 4)`` — XYXY bounding boxes."""
        return self._boxes

    @property
    def labels(self) -> NDArray[Any]:
        """:class:`NDArray` of shape ``(N,)`` — integer class labels."""
        return self._labels

    @property
    def scores(self) -> NDArray[Any]:
        """:class:`NDArray` of shape ``(N, M)`` — one-hot prediction scores."""
        return self._scores


class TorchvisionDataset:
    """MAITE-compatible adapter for torchvision ``VisionDataset`` instances.

    Wraps a torchvision dataset that returns ``(image, target)`` tuples and
    converts them to the ``(NDArray, target, dict)`` format expected by the
    :class:`~dataeval.protocols.AnnotatedDataset` protocol.

    **Image classification** datasets return ``(image, int_label)``; the
    integer label is converted to a one-hot float32 vector when *classes*
    is discoverable on the wrapped dataset.

    **Object detection** datasets (e.g. those wrapped with
    ``torchvision.datasets.wrap_dataset_for_transforms_v2``) return
    ``(image, dict)`` where the dict contains ``"boxes"``
    (``BoundingBoxes``) and ``"labels"`` keys.  Bounding boxes are
    normalised to XYXY format regardless of the source
    ``BoundingBoxFormat``.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        A torchvision-style dataset whose ``__getitem__`` returns
        ``(image, target)`` tuples.
    """

    def __init__(self, dataset: Any) -> None:
        """Initialize adapter from a torchvision dataset."""
        self._dataset = dataset

        # Discover class names from the dataset if available
        classes: list[str] | None = getattr(dataset, "classes", None)
        self._index2label: dict[int, str] = {i: c for i, c in enumerate(classes)} if classes else {}
        self._num_classes: int | None = len(classes) if classes else None

        name = getattr(dataset, "__class__", type(dataset)).__name__
        logger.info("TorchvisionDataset: wrapping %s (%d samples)", name, len(self))

    # -- AnnotatedDataset protocol ----------------------------------------

    def __len__(self) -> int:
        """Return the number of samples in the wrapped dataset."""
        return len(self._dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[NDArray[Any], Any, dict[str, Any]]:
        """Return ``(image, target, metadata)`` for the given index."""
        image, target = self._dataset[index]

        img_array = self._convert_image(image)

        if isinstance(target, dict) and "boxes" in target:
            converted_target = self._convert_od_target(target)
        else:
            converted_target = self._convert_cls_target(target)

        datum_metadata: dict[str, Any] = {"id": index}
        return img_array, converted_target, datum_metadata

    @property
    def metadata(self) -> DatasetMetadata:
        """Dataset-level metadata (DatasetMetadata TypedDict shape)."""
        name = getattr(self._dataset, "__class__", type(self._dataset)).__name__
        return DatasetMetadata(
            id=name,
            index2label=dict(self._index2label),
        )

    # -- Internal ---------------------------------------------------------

    @staticmethod
    def _convert_image(image: Any) -> NDArray[Any]:
        """Convert a PIL Image or torch Tensor to CHW float32 numpy array."""
        from PIL import Image

        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))

        # torch.Tensor — detach, move to cpu, convert
        arr = np.asarray(image, dtype=np.float32)  # handles Tensor and ndarray
        if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            # HWC → CHW
            arr = np.transpose(arr, (2, 0, 1))
        return arr

    def _convert_cls_target(self, target: Any) -> NDArray[Any]:
        """Convert an integer label to a one-hot vector, or pass through arrays."""
        if isinstance(target, int) and self._num_classes is not None:
            one_hot = np.zeros(self._num_classes, dtype=np.float32)
            one_hot[target] = 1.0
            return one_hot
        return np.asarray(target, dtype=np.float32)

    def _convert_od_target(self, target: dict[str, Any]) -> _ObjectDetectionTarget:
        """Convert a torchvision v2 object-detection target dict.

        Expects *target* to contain at least ``"boxes"`` and ``"labels"``
        keys.  ``"boxes"`` may be a ``torchvision.tv_tensors.BoundingBoxes``
        (with an associated format) or a plain tensor already in XYXY order.
        All non-XYXY formats are converted to XYXY.
        """
        boxes_raw = target["boxes"]
        labels_raw = target["labels"]

        # Convert boxes to XYXY numpy — respect BoundingBoxes.format
        boxes_np = self._boxes_to_xyxy_numpy(boxes_raw)
        labels_np = np.asarray(labels_raw, dtype=np.intp)

        # Build one-hot score matrix
        num_classes = self._num_classes or (int(labels_np.max()) + 1 if len(labels_np) > 0 else 0)
        scores_np = np.zeros((len(labels_np), num_classes), dtype=np.float32)
        for i, lbl in enumerate(labels_np):
            scores_np[i, lbl] = 1.0

        return _ObjectDetectionTarget(boxes=boxes_np, labels=labels_np, scores=scores_np)

    @staticmethod
    def _boxes_to_xyxy_numpy(boxes: Any) -> NDArray[np.float32]:
        """Convert bounding boxes to XYXY float32 numpy array.

        Handles ``torchvision.tv_tensors.BoundingBoxes`` (any format),
        plain torch Tensors (assumed XYXY), and numpy arrays.
        """
        try:
            from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

            if isinstance(boxes, BoundingBoxes) and boxes.format != BoundingBoxFormat.XYXY:
                from torchvision.ops import box_convert

                boxes = box_convert(boxes, in_fmt=boxes.format.name.lower(), out_fmt="xyxy")
        except ImportError:  # pragma: no cover — torchvision not installed
            pass

        return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def load_dataset_torchvision(dataset: Any) -> TorchvisionDataset:
    """Wrap a torchvision dataset as a MAITE-compatible dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        A torchvision-style dataset.

    Returns
    -------
    TorchvisionDataset
        MAITE-compatible wrapper around the torchvision dataset.
    """
    return TorchvisionDataset(dataset)


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
    >>> from dataeval_flow import load_dataset_huggingface
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
    dataset_format: Literal["huggingface", "coco", "yolo", "image_folder"] = "huggingface",
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
    dataset_format : Literal["huggingface", "coco", "yolo", "image_folder"]
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
    >>> from dataeval_flow import load_dataset
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


# ---------------------------------------------------------------------------
# Resolved dataset — unified output of config → dataset resolution
# ---------------------------------------------------------------------------

_LABEL_SOURCE: dict[str, str] = {
    "coco": "annotations",
    "yolo": "annotations",
    "huggingface": "huggingface",
    "maite": "protocol",
    "torchvision": "torchvision",
}


@dataclass
class ResolvedDataset:
    """Result of resolving any dataset config into a ready-to-use dataset.

    Produced by :func:`resolve_dataset` so that downstream code (orchestrator,
    cache) never needs to branch on config type.
    """

    name: str
    dataset: AnnotatedDataset[Any]
    label_source: str | None
    cache_key: str


def resolve_dataset(config: BaseModel) -> ResolvedDataset:
    """Resolve a dataset config into a :class:`ResolvedDataset`.

    Handles both file-backed (:class:`DatasetConfig` union members) and
    in-memory (:class:`DatasetProtocolConfig`) configs, centralizing all
    format-specific branching in one place.
    """
    from dataeval_flow.cache import dataset_fingerprint
    from dataeval_flow.config.schemas._dataset import (
        DatasetProtocolConfig,
        HuggingFaceDatasetConfig,
        ImageFolderDatasetConfig,
        _DatasetConfigBase,
    )

    if isinstance(config, DatasetProtocolConfig):
        dataset = load_dataset_torchvision(config.dataset) if config.format == "torchvision" else config.dataset
        label_source: str | None = _LABEL_SOURCE.get(config.format)
        cache_key = f"{config.name}:{config.format}:{config.version}"
    elif isinstance(config, _DatasetConfigBase):
        # All file-backed dataset configs share path/format; dispatch through load_dataset
        kwargs: dict[str, Any] = {}
        if isinstance(config, HuggingFaceDatasetConfig):
            kwargs["split"] = config.split
        elif isinstance(config, ImageFolderDatasetConfig):
            kwargs["recursive"] = config.recursive
            kwargs["infer_labels"] = config.infer_labels
        else:
            # Coco / Yolo — forward format-specific fields
            for field_name in ("annotations_file", "images_dir", "labels_dir", "classes_file"):
                if hasattr(config, field_name):
                    kwargs[field_name] = getattr(config, field_name)

        dataset = load_dataset(Path(config.path), dataset_format=config.format, **kwargs)

        if isinstance(config, ImageFolderDatasetConfig) and config.infer_labels:
            label_source = "filepath"
        else:
            label_source = _LABEL_SOURCE.get(config.format)
        cache_key = config.model_dump_json(exclude_defaults=False)
    else:
        raise ValueError(f"Unsupported dataset config type: {type(config).__name__}")

    # Append a content-based fingerprint so the cache invalidates when
    # the underlying data changes even if the config metadata is unchanged.
    fingerprint = dataset_fingerprint(dataset)
    cache_key = f"{cache_key}|fp:{fingerprint}"

    return ResolvedDataset(
        name=config.name,
        dataset=dataset,
        label_source=label_source,
        cache_key=cache_key,
    )
