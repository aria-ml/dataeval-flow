"""DataEval Workflows - Data evaluation and monitoring pipelines.

Quick start::

    from dataeval_flow import load_config, run_tasks

    config = load_config(Path("params.yaml"))
    results = run_tasks(config)
    print(results[0].report())

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
]
__version__ = "0.1.0"
