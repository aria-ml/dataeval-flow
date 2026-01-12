"""DataEval Application - Data evaluation and inspection tools.

This package can be used standalone (without container) for loading
and inspecting datasets in MAITE-compatible format.

Examples
--------
>>> from pathlib import Path
>>> from dataeval_app import load_dataset, inspect_dataset
>>> dataset = load_dataset(Path("/my/local/dataset"), split="train")
>>> inspect_dataset(Path("/my/local/dataset"))
0
"""

from dataeval_app.dataset import (
    inspect_dataset,
    load_dataset,
    load_dataset_huggingface,
)

__all__ = ["load_dataset", "load_dataset_huggingface", "inspect_dataset"]
__version__ = "0.1.0"
