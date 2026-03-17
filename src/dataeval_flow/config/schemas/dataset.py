"""Dataset configuration schema."""

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, model_validator

# Which optional fields are relevant for each format.
_FORMAT_FIELDS: dict[str, frozenset[str]] = {
    "huggingface": frozenset({"split"}),
    "image_folder": frozenset({"recursive", "infer_labels"}),
    "coco": frozenset({"annotations_file", "images_dir", "classes_file"}),
    "yolo": frozenset({"images_dir", "labels_dir", "classes_file"}),
}

# Fields that have a non-None/non-False "unset" value (i.e. their default is falsy).
_OPTIONAL_FIELDS: frozenset[str] = frozenset(
    {
        "split",
        "recursive",
        "infer_labels",
        "annotations_file",
        "images_dir",
        "labels_dir",
        "classes_file",
    }
)


class DatasetConfig(BaseModel):
    """Dataset configuration schema."""

    name: str
    format: Literal["huggingface", "coco", "yolo", "image_folder"]
    path: str
    split: str | None = None
    recursive: bool = False
    infer_labels: bool = False
    annotations_file: str | None = None
    images_dir: str | None = None
    labels_dir: str | None = None
    classes_file: str | None = None

    @model_validator(mode="after")
    def _reject_irrelevant_fields(self) -> "DatasetConfig":
        """Raise if format-specific fields are set for a format that ignores them."""
        allowed = _FORMAT_FIELDS.get(self.format, frozenset())
        for field_name in _OPTIONAL_FIELDS - allowed:
            value = getattr(self, field_name)
            # Only flag fields the user explicitly set (non-default).
            if value is not None and value is not False and value != 0:
                msg = f"'{field_name}' is not supported for format='{self.format}'"
                raise ValueError(msg)
        return self


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
