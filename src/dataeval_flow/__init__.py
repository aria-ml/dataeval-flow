"""DataEval Workflows - Data evaluation and monitoring pipelines.

Quick start::

    from pathlib import Path
    from dataeval_flow import load_config, run_tasks

    config = load_config(Path("/path/to/data/config.yaml"))
    results = run_tasks(config, data_dir=Path("/path/to/data"))
    print(results[0].report())  # text report
    results[0].export("output/")  # write result JSON

Or build a pipeline programmatically::

    from dataeval_flow import (
        PipelineConfig, HuggingFaceDatasetConfig, FlattenExtractorConfig,
        SourceConfig, DataCleaningWorkflowConfig, TaskConfig, run_tasks,
    )

Discovery helpers::

    >>> from dataeval_flow import list_workflows
    >>> list_workflows()
    [{'name': 'data-cleaning', ...}, {'name': 'drift-monitoring', ...}]
"""

from dataeval_flow.config import (
    BoVWExtractorConfig,
    CocoDatasetConfig,
    DataCleaningTaskConfig,
    DataCleaningWorkflowConfig,
    DatasetProtocolConfig,
    DriftMonitoringTaskConfig,
    DriftMonitoringWorkflowConfig,
    FlattenExtractorConfig,
    HuggingFaceDatasetConfig,
    ImageFolderDatasetConfig,
    OnnxExtractorConfig,
    PipelineConfig,
    PreprocessorConfig,
    SelectionConfig,
    SelectionStep,
    SourceConfig,
    TaskConfig,
    TorchExtractorConfig,
    UncertaintyExtractorConfig,
    YoloDatasetConfig,
    export_params_schema,
    load_config,
    load_config_folder,
)
from dataeval_flow.dataset import load_dataset
from dataeval_flow.workflow import WorkflowResult, get_workflow, list_workflows, run_tasks

__all__ = [
    # --- Core workflow ---
    "load_config",
    "load_config_folder",
    "run_tasks",
    "PipelineConfig",
    "WorkflowResult",
    # --- Discovery ---
    "list_workflows",
    "get_workflow",
    # --- Dataset configs ---
    "HuggingFaceDatasetConfig",
    "ImageFolderDatasetConfig",
    "CocoDatasetConfig",
    "YoloDatasetConfig",
    "DatasetProtocolConfig",
    # --- Extractor configs ---
    "OnnxExtractorConfig",
    "BoVWExtractorConfig",
    "FlattenExtractorConfig",
    "TorchExtractorConfig",
    "UncertaintyExtractorConfig",
    # --- Workflow configs ---
    "DataCleaningWorkflowConfig",
    "DriftMonitoringWorkflowConfig",
    # --- Task configs ---
    "TaskConfig",
    "DataCleaningTaskConfig",
    "DriftMonitoringTaskConfig",
    # --- Composition ---
    "SourceConfig",
    "PreprocessorConfig",
    "SelectionConfig",
    "SelectionStep",
    # --- Utilities ---
    "load_dataset",
    "export_params_schema",
    "__version__",
]

try:
    from dataeval_flow._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "unknown"

# Strongly type for pyright
__version__ = str(__version__)

import logging

# Library-style default: attach a NullHandler so importing dataeval_flow without
# running the CLI (which calls setup_logging) never emits unconfigured output or
# triggers the logging "last resort" handler. The CLI configures real handlers on
# the root logger via dataeval_flow._logging.setup_logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())
