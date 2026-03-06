"""Dataset configuration schema."""

from typing import Literal

from pydantic import BaseModel

DatasetFormat = Literal["huggingface", "coco", "voc", "yolo", "image_folder"]


class DatasetConfig(BaseModel):
    """Dataset configuration schema."""

    name: str
    format: DatasetFormat
    path: str
    split: str | None = None
    recursive: bool = False
    infer_labels: bool = False
