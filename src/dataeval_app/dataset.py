#!/usr/bin/env python3
"""Dataset loading utilities - standalone library functions.

All functions accept parameters. No hardcoded container paths.

This module provides functions for loading datasets in HuggingFace format
and converting them to MAITE-compatible objects for evaluation.
"""

import logging
from pathlib import Path
from typing import TypeAlias

from maite_datasets.adapters import HFImageClassificationDataset, HFObjectDetectionDataset

logger: logging.Logger = logging.getLogger(__name__)

MaiteDataset: TypeAlias = HFImageClassificationDataset | HFObjectDetectionDataset


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
    >>> from dataeval_app import load_dataset_huggingface
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


def load_dataset(path: Path, split: str | None = None) -> "MaiteDataset":
    """Load a dataset and convert to MAITE format.

    This is the main entry point for loading datasets. Currently delegates
    to HuggingFace loader but may support additional formats in the future.

    Parameters
    ----------
    path : Path
        Path to the dataset directory.
    split : str | None
        Optional split name to load (e.g. "train", "test").

    Returns
    -------
    MaiteDataset
        MAITE-compatible dataset object.

    Raises
    ------
    KeyError
        If split is not specified for a multi-split dataset or the specified split is not found in the dataset.

    Examples
    --------
    >>> from pathlib import Path
    >>> from dataeval_app import load_dataset
    >>> ds = load_dataset(Path("/data/cifar10"))
    """
    return load_dataset_huggingface(path, split=split)
