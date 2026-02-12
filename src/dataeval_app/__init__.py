"""DataEval Application - Data evaluation tools.

This package provides dataset loading, preprocessing, and workflow
execution for evaluating datasets using DataEval.

Examples
--------
>>> from pathlib import Path
>>> from dataeval_app import load_dataset
>>> dataset = load_dataset(Path("/my/local/dataset"))
"""

from dataeval_app.dataset import (
    load_dataset,
    load_dataset_huggingface,
)
from dataeval_app.preprocessing import (
    PreprocessingStep,
    build_preprocessing,
)
from dataeval_app.workflow import get_workflow, list_workflows, run_task

__all__ = [
    # Dataset
    "load_dataset",
    "load_dataset_huggingface",
    # Preprocessing
    "PreprocessingStep",
    "build_preprocessing",
    # Workflow
    "get_workflow",
    "list_workflows",
    "run_task",
]
__version__ = "0.1.0"
