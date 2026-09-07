"""Dataset configuration schemas — one class per format."""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dataeval_flow.config._paths import validate_config_path


class _DatasetConfigBase(BaseModel):
    """Common fields shared by all dataset formats."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    format: Any
    name: str
    path: str
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

        from dataeval_flow.metadata import stat_names_for
        from dataeval_flow.policy import _ROW_LEVELS

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


#: Tutorial datasets `format: demo` can name, mapped to their `maite_datasets` module.
#: An explicit table rather than a dynamic import: `dataset:` comes from a config file,
#: and resolving it by importing whatever it names would let a config run arbitrary code.
_DEMO_DATASETS: "Mapping[str, str]" = {
    "M3FD": "maite_datasets.object_detection",
    "DroneVehicle": "maite_datasets.object_detection",
    "SeaDrone": "maite_datasets.object_detection",
}


class DemoDatasetConfig(_DatasetConfigBase):
    """Dataset config for a dataset the tutorials use.

    Names one of the datasets shipped for tutorials so a tutorial pipeline runs from
    config alone. This is not an ingestion path for your own data — read that with
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

    format: Literal["demo"] = "demo"
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
            "Split to load, in the loader's own vocabulary — `train`, `val`, `test`, or "
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

    format: Literal["huggingface"] = "huggingface"
    split: str | None = None
    task: Literal["image_classification", "object_detection"]


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


class YoloDatasetConfig(_DatasetConfigBase):
    """Dataset config for YOLO format.

    ``path`` is the dataset root — the directory holding ``data.yaml`` and the
    image/label trees — for either Ultralytics arrangement (``images/train/`` +
    ``labels/train/`` or ``train/images/`` + ``train/labels/``).  Select a split
    with ``split`` rather than by pointing ``path`` at a split subdirectory,
    which puts ``data.yaml`` out of scope and falls back to numeric class names.

    YAML example::

        datasets:
          - name: yolo_train
            format: yolo
            path: yolo
            split: train
    """

    format: Literal["yolo"] = "yolo"
    split: str | None = None
    yaml_file: str | None = None
    ann_dir: str | None = None


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
