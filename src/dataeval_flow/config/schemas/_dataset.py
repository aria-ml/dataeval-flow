"""Dataset configuration schemas — one class per format."""

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from dataeval_flow.config._paths import validate_config_path


class _DatasetConfigBase(BaseModel):
    """Common fields shared by all dataset formats."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    format: Any
    name: str
    path: str

    @field_validator("path")
    @classmethod
    def _path_must_be_relative(cls, v: str) -> str:
        return validate_config_path(v)


class HuggingFaceDatasetConfig(_DatasetConfigBase):
    """Dataset config for HuggingFace format.

    YAML example::

        datasets:
          - name: cifar10_train
            format: huggingface
            path: ./cifar10
            split: train
    """

    format: Literal["huggingface"] = "huggingface"
    split: str | None = None


class ImageFolderDatasetConfig(_DatasetConfigBase):
    """Dataset config for image_folder format.

    YAML example::

        datasets:
          - name: photos
            format: image_folder
            path: photos
            recursive: true
            infer_labels: true
    """

    format: Literal["image_folder"] = "image_folder"
    recursive: bool = False
    infer_labels: bool = False


class CocoDatasetConfig(_DatasetConfigBase):
    """Dataset config for COCO format.

    YAML example::

        datasets:
          - name: coco_train
            format: coco
            path: coco
            annotations_file: instances_train.json
            images_dir: train2017
    """

    format: Literal["coco"] = "coco"
    annotations_file: str | None = None
    images_dir: str | None = None
    classes_file: str | None = None


class YoloDatasetConfig(_DatasetConfigBase):
    """Dataset config for YOLO format.

    YAML example::

        datasets:
          - name: yolo_train
            format: yolo
            path: yolo
            images_dir: images
            labels_dir: labels
            classes_file: classes.txt
    """

    format: Literal["yolo"] = "yolo"
    images_dir: str | None = None
    labels_dir: str | None = None
    classes_file: str | None = None


class DatasetProtocolConfig(BaseModel):
    """Dataset Configuration schema for an in-memory dataset.

    Not serializable — for programmatic use only. Cannot be loaded from
    YAML/JSON config files or edited in the builder UI.
    """

    serializable: ClassVar[bool] = False
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    name: str
    format: Literal["maite", "torchvision"] = "maite"
    dataset: Any
    version: str = "1"
