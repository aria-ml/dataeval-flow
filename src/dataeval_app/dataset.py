#!/usr/bin/env python3
"""Dataset loading utilities - standalone library functions.

All functions accept parameters. No hardcoded container paths.

This module provides functions for loading datasets in HuggingFace format
and converting them to MAITE-compatible objects for evaluation.
"""

from pathlib import Path
from typing import Any


def load_dataset_huggingface(path: Path, split: str | None = None) -> Any:
    """Load a HuggingFace dataset and convert to MAITE format.

    Parameters
    ----------
    path : Path
        Path to the dataset directory containing HuggingFace dataset files.
    split : str | None, optional
        Split name for DatasetDict (e.g., "train", "test"). Required if
        the dataset contains multiple splits.

    Returns
    -------
    Any
        MAITE-compatible dataset object that can be used with DataEval.

    Raises
    ------
    ImportError
        If required dependencies (datasets, maite_datasets) are missing.
    ValueError
        If split is required but not provided, or if specified split is not found.
    RuntimeError
        If dataset loading or conversion fails.

    Examples
    --------
    >>> from pathlib import Path
    >>> from dataeval_app import load_dataset_huggingface
    >>> ds = load_dataset_huggingface(Path("/data/cifar10"), split="test")
    """
    from datasets import load_from_disk
    from maite_datasets.adapters import from_huggingface

    dataset = load_from_disk(str(path))

    print(f"Loaded type: {type(dataset).__name__}")
    if hasattr(dataset, "keys") and callable(dataset.keys):  # type: ignore[union-attr]
        available_splits = list(dataset.keys())  # type: ignore[union-attr]
        if available_splits:
            print(f"DatasetDict detected. Available splits: {available_splits}")
            if not split:
                raise ValueError(f"Multiple splits found. Specify --split argument.\n  Available: {available_splits}")
            if split not in available_splits:
                raise ValueError(f"Split '{split}' not found.\n  Available: {available_splits}")
            print(f"Using split: '{split}'")
            dataset = dataset[split]
    else:
        print("Dataset detected (single split)")

    return from_huggingface(dataset)  # type: ignore[arg-type]


def load_dataset(path: Path, split: str | None = None) -> Any:
    """Load a dataset and convert to MAITE format.

    This is the main entry point for loading datasets. Currently delegates
    to HuggingFace loader but may support additional formats in the future.

    Parameters
    ----------
    path : Path
        Path to the dataset directory.
    split : str | None, optional
        Split name for multi-split datasets.

    Returns
    -------
    Any
        MAITE-compatible dataset object.

    Examples
    --------
    >>> from pathlib import Path
    >>> from dataeval_app import load_dataset
    >>> ds = load_dataset(Path("/data/cifar10"), split="test")
    """
    return load_dataset_huggingface(path, split)


def inspect_dataset(path: Path, split: str | None = None) -> int:
    """Inspect a dataset and print summary.

    Loads the dataset, displays metadata, and shows the first 10 samples
    with their shapes and class information.

    Parameters
    ----------
    path : Path
        Path to the dataset directory.
    split : str | None, optional
        Split name for multi-split datasets (e.g., "train", "test").

    Returns
    -------
    int
        Exit code: 0 for success, 1 for error.

    Raises
    ------
    FileNotFoundError
        If the dataset path does not exist.

    Examples
    --------
    >>> from pathlib import Path
    >>> from dataeval_app import inspect_dataset
    >>> result = inspect_dataset(Path("/data/cifar10"), split="test")
    >>> result
    0
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    maite_ds = load_dataset(path, split)
    print(f"Dataset loaded: {len(maite_ds)} images")

    # Show MAITE metadata if available
    index2label = None
    if hasattr(maite_ds, "metadata"):
        meta = maite_ds.metadata
        print("\nMAITE Metadata:")
        if "id" in meta:
            print(f"  Dataset ID: {meta['id']}")
        if "index2label" in meta:
            index2label = meta["index2label"]
            class_names = list(index2label.values())
            print(f"  Classes ({len(class_names)}): {class_names}")

    print("\nFirst 10 samples:")
    for i in range(min(10, len(maite_ds))):
        try:
            item = maite_ds[i]
            if isinstance(item, tuple) and len(item) >= 2:
                img, target = item[0], item[1]
                datum_meta = item[2] if len(item) >= 3 else None
                if index2label is not None:
                    class_idx = int(target.argmax())
                    class_info = index2label.get(class_idx, f"class_{class_idx}")
                else:
                    class_info = str(target)
                meta_info = f", meta={datum_meta}" if datum_meta else ""
                print(f"  [{i}] shape={img.shape}, class={class_info}{meta_info}")
            else:
                print(f"  [{i}] item={type(item)}")
        except Exception as e:  # noqa: BLE001, PERF203
            # Intentional: Robust error handling for dataset iteration
            print(f"  [{i}] ERROR: {e}")

    print("\nInspection completed successfully.")
    return 0
