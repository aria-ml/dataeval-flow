#!/usr/bin/env python3
"""Workflow: Load dataset and inspect images."""

import os
import sys
from pathlib import Path
from typing import Any, Optional


def load_dataset_huggingface(path: Path, split: Optional[str] = None) -> Any:
    """Load a HuggingFace dataset and convert to MAITE format.

    Args:
        path: Path to the dataset directory
        split: Optional split name for DatasetDict (e.g., "train", "test")

    Returns:
        MAITE-compatible dataset object

    Raises:
        ImportError: If required dependencies are missing
        ValueError: If split is required but not provided, or split not found
        RuntimeError: If dataset loading or conversion fails
    """
    from datasets import load_from_disk
    from maite_datasets.adapters import from_huggingface

    dataset = load_from_disk(str(path))

    # Handle both Dataset and DatasetDict formats
    print(f"Loaded type: {type(dataset).__name__}")
    if hasattr(dataset, "keys") and callable(dataset.keys):
        # DatasetDict is a dict with split names as keys (e.g., "train", "test")
        available_splits = list(dataset.keys())
        if available_splits:
            print(f"DatasetDict detected. Available splits: {available_splits}")
            if not split:
                raise ValueError(
                    f"Multiple splits found. Set DATASET_SPLIT env var.\n"
                    f"  Example: -e DATASET_SPLIT={available_splits[0]}\n"
                    f"  Available: {available_splits}"
                )
            if split not in available_splits:
                raise ValueError(
                    f"Split '{split}' not found.\n"
                    f"  Available: {available_splits}"
                )
            print(f"Using split: '{split}'")
            dataset = dataset[split]
    else:
        print("Dataset detected (single split)")

    return from_huggingface(dataset)


def load_dataset(path: Path, split: Optional[str] = None) -> Any:
    """Load a dataset and convert to MAITE format.

    Currently supports HuggingFace format. Additional formats can be added
    by implementing new loader functions and adding detection logic here.

    Args:
        path: Path to the dataset directory
        split: Optional split name for multi-split datasets

    Returns:
        MAITE-compatible dataset object

    Raises:
        ImportError: If required dependencies are missing
        ValueError: If split is required but not provided, or split not found
        RuntimeError: If dataset loading or conversion fails
    """
    # Currently only HuggingFace format is supported
    # Future: Add format detection and call appropriate loader
    return load_dataset_huggingface(path, split)


def main() -> int:
    """Load dataset and list images. Returns 0 on success, 1 on error."""
    dataset_path = Path("/data/dataset")

    # Validate dataset path exists
    if not dataset_path.exists():
        print(f"ERROR: Dataset path not found: {dataset_path}")
        print("Troubleshooting:")
        print("  1. Ensure DATASET_DIR is set in docker-compose")
        print("  2. Check that the path contains a valid dataset")
        return 1

    split = os.environ.get("DATASET_SPLIT", "") or None

    try:
        maite_ds = load_dataset(dataset_path, split)
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Dependencies should be pre-installed in container.")
        return 1
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        return 1

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
                # Show class name if index2label available, otherwise show target array
                if index2label is not None:
                    class_idx = int(target.argmax())
                    class_info = index2label.get(class_idx, f"class_{class_idx}")
                else:
                    class_info = str(target)
                # Show datum metadata if available
                meta_info = f", meta={datum_meta}" if datum_meta else ""
                print(f"  [{i}] shape={img.shape}, class={class_info}{meta_info}")
            else:
                print(f"  [{i}] item={type(item)}")
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")

    print("\nInspection completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
