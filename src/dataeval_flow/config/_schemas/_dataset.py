"""Dataset configuration schemas — one class per format."""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dataeval_flow.config._paths import validate_config_path


class _DatasetConfigBase(BaseModel):
    """Common fields shared by all dataset formats."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    format: Any
    name: str = Field(description="Identifier for the dataset, referenced by sources.")
    path: str = Field(
        description=(
            "Where the dataset lives, relative to the data root; a path not found there is looked up in the root's "
            "`data` folder."
        )
    )
    value_range: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Interval this dataset's imagery occupies, as (low, high). Integer encodings "
            "state their own range; float data does not, and the statistics that need one "
            "answer NaN without it — the whole visual family, pixel histogram and entropy, "
            "and dimension depth. A description of the data rather than a decision about "
            "it, which is why it lives here and not under `metadata:`, and why every "
            "workflow reading this dataset gets the same one. Leave unset for integer "
            "imagery."
        ),
    )
    channel_groups: Mapping[str, int | Sequence[int]] | None = Field(
        default=None,
        description=(
            "Named groups of bands measured separately, as `name: index` or "
            "`name: [indices]`. That channel 3 is infrared is a fact about the sensor, so "
            "declare it here and every workflow reading this dataset sees the same one. "
            "A group becomes a set of `<name>_<statistic>` columns alongside the "
            "unprefixed ones. Reference the groups from a `stats:` policy's `measure` to "
            "decide which statistics each one gets. Leave unset for single-band or "
            "undifferentiated imagery."
        ),
    )

    @field_validator("path")
    @classmethod
    def _path_must_be_relative(cls, v: str) -> str:
        return validate_config_path(v)

    @field_validator("channel_groups")
    @classmethod
    def _groups_are_usable(
        cls, value: "Mapping[str, int | Sequence[int]] | None"
    ) -> "Mapping[str, int | Sequence[int]] | None":
        """Refuse a group name or band list that cannot produce a column.

        Check here rather than at the stats call: a collision surfaces as a silently
        overwritten column after the dataset walk, and an empty group as a column of NaN.
        """
        if value is None:
            return value

        from dataeval.flags import ImageStats

        from dataeval_flow._metadata import stat_names_for
        from dataeval_flow._policy import _ROW_LEVELS

        reserved = stat_names_for(ImageStats.ALL) | {"background"} | set(_ROW_LEVELS)
        for name, bands in value.items():
            if name in reserved:
                raise ValueError(
                    f"Channel group {name!r} collides with a reserved name. A group name "
                    "must not be a statistic name, `background`, or a row level, because "
                    "its columns are named `<group>_<statistic>` and would be "
                    "indistinguishable. Rename the group.",
                )
            indices = [bands] if isinstance(bands, int) else list(bands)
            if not indices:
                raise ValueError(f"Channel group {name!r} names no bands. Give it an index or a list of indices.")
            if any(index < 0 for index in indices):
                raise ValueError(f"Channel group {name!r} names a negative band index. Band indices start at 0.")
        return value


# Tutorial datasets `format: demo` can name, mapped to their `maite_datasets` module.
# An explicit table rather than a dynamic import: `dataset:` comes from a config file,
# and resolving it by importing whatever it names would let a config run arbitrary code.
_DEMO_DATASETS: "Mapping[str, str]" = {
    "M3FD": "maite_datasets.object_detection",
    "DroneVehicle": "maite_datasets.object_detection",
    "SeaDrone": "maite_datasets.object_detection",
}


class DemoDatasetConfig(_DatasetConfigBase):
    """Dataset config for a dataset the tutorials use.

    Names one of the datasets shipped for tutorials so a tutorial pipeline runs from
    config alone. This is not an ingestion path for your own data. Read that with
    `coco`, `yolo`, `huggingface`, or `image_folder`.

    ``path`` is the root the dataset was downloaded under, not the dataset directory
    itself: each loader appends its own subdirectory.

    YAML example::

        datasets:
          - name: m3fd_train
            format: demo
            dataset: M3FD
            path: data
            image_set: train
            channel_groups:
              rgb: [0, 1, 2]
              ir: 3
    """

    format: Literal["demo"] = Field(default="demo", description="Selects this dataset format: `demo`.")
    dataset: str = Field(
        description=(
            "Which tutorial dataset to load. Accepts a name from the built-in table; "
            "anything else is refused, because a config file must not be able to import "
            "code by naming it."
        ),
    )
    image_set: str | None = Field(
        default=None,
        description=(
            "Split to load, in the loader's own vocabulary: `train`, `val`, `test`, or "
            "`base` where the loader offers them. Leave unset for the loader's default."
        ),
    )
    download: bool = Field(
        default=False,
        description=(
            "Fetch the data if it is not already under `path`. Off by default: a config "
            "run should not reach the network unless you asked it to. Some tutorial "
            "datasets are credential-gated and cannot be fetched this way at all."
        ),
    )

    @field_validator("dataset")
    @classmethod
    def _dataset_is_known(cls, value: str) -> str:
        """Refuse a name that is not in the table."""
        if value not in _DEMO_DATASETS:
            known = ", ".join(sorted(_DEMO_DATASETS))
            raise ValueError(
                f"Unknown demo dataset {value!r}. Choose one of: {known}. To read your own "
                "data, use `coco`, `yolo`, `huggingface`, or `image_folder` instead.",
            )
        return value


class HuggingFaceDatasetConfig(_DatasetConfigBase):
    """Dataset config for HuggingFace format.

    YAML example::

        datasets:
          - name: cifar10_train
            format: huggingface
            path: ./cifar10
            split: train
            task: image_classification
    """

    format: Literal["huggingface"] = Field(
        default="huggingface", description="Selects this dataset format: `huggingface`."
    )
    split: str | None = Field(
        default=None, description="Split to load, read as a subdirectory of `path`. Leave unset to load `path` itself."
    )
    task: Literal["image_classification", "object_detection"] = Field(
        description="Which loader reads the dataset: image classification or object detection."
    )


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

    format: Literal["image_folder"] = Field(
        default="image_folder", description="Selects this dataset format: `image_folder`."
    )
    recursive: bool = Field(default=False, description="Scan subdirectories for images as well.")
    infer_labels: bool = Field(
        default=False, description="Label each image by the subdirectory it sits in, one class per subdirectory."
    )


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

    format: Literal["coco"] = Field(default="coco", description="Selects this dataset format: `coco`.")
    annotations_file: str | None = Field(
        default=None,
        description=(
            "The annotations JSON, relative to `path`, which also selects the split. Leave unset to read the one "
            "found under `path`, preferring `annotations/instances_*.json`."
        ),
    )
    images_dir: str | None = Field(
        default=None,
        description=(
            "The directory the annotations' image file names are relative to, relative to `path`. Leave unset for "
            "the annotations file's folder, or its parent when that folder is `annotations/`."
        ),
    )


class YoloDatasetConfig(_DatasetConfigBase):
    """Dataset config for YOLO format.

    ``path`` is the dataset root — the directory holding ``data.yaml`` and the
    image/label trees — for either Ultralytics arrangement (``images/train/`` +
    ``labels/train/`` or ``train/images/`` + ``train/labels/``).  Select a split
    with ``split``. Pointing ``path`` at a split subdirectory puts
    ``data.yaml`` out of scope and falls back to numeric class names.

    YAML example::

        datasets:
          - name: yolo_train
            format: yolo
            path: yolo
            split: train
    """

    format: Literal["yolo"] = Field(default="yolo", description="Selects this dataset format: `yolo`.")
    split: str | None = Field(
        default=None,
        description=(
            "Load only this split (`train`, `val` or `test`; aliases such as `validation` normalize). Leave unset "
            "to load every split under `path`."
        ),
    )
    yaml_file: str | None = Field(
        default=None,
        description=(
            "The `data.yaml` file, relative to `path`, for one not at the root under the conventional name. It is "
            "authoritative: a missing file, or one whose image sources yield nothing, gives an empty dataset."
        ),
    )
    ann_dir: str | None = Field(
        default=None,
        description="The label directory, relative to `path`, for labels kept outside the conventional `labels/`.",
    )


class DatasetProtocolConfig(BaseModel):
    """Dataset Configuration schema for an in-memory dataset.

    Not serializable — for programmatic use only. Cannot be loaded from
    YAML/JSON config files or edited in the builder UI.
    """

    serializable: ClassVar[bool] = False
    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Identifier for the dataset, referenced by sources.")
    format: Literal["maite", "torchvision"] = Field(
        default="maite",
        description="`maite` for a dataset already in MAITE form; `torchvision` for a torchvision dataset to wrap.",
    )
    dataset: Any = Field(description="The dataset object itself.")
    version: str = Field(
        default="1",
        description=(
            "A label for this dataset's contents, part of its cache key beside a fingerprint of a sample of its "
            "items. Change it when the data changes, so the cache is sure to miss."
        ),
    )
