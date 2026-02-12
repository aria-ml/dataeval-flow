"""Dataset configuration schema."""

from typing import Literal

from pydantic import BaseModel, Field

AutoBinMethod = Literal["uniform_width", "uniform_count", "clusters"]


class SplitConfig(BaseModel):
    """Dataset split configuration."""

    num_folds: int | None = None
    stratify: bool = False
    split_on: list[str] | None = None
    test_frac: float | None = None
    val_frac: float | None = None


class DatasetConfig(BaseModel):
    """Dataset configuration schema."""

    name: str
    format: Literal["huggingface", "coco", "voc", "yolo"]
    path: str
    splits: list[str] | SplitConfig
    metadata_auto_bin_method: AutoBinMethod | None = None
    metadata_exclude: list[str] = Field(default_factory=list)
    metadata_continuous_factor_bins: dict[str, list[float]] | None = None
