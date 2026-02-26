"""Dataset configuration schema."""

from typing import Literal

from pydantic import BaseModel

DatasetFormat = Literal["huggingface", "coco", "voc", "yolo"]


class DatasetConfig(BaseModel):
    """Dataset configuration schema."""

    name: str
    format: DatasetFormat
    path: str
    split: str | None = None
