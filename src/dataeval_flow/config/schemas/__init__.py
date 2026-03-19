"""Schema catalog — concrete types and discriminated-union aliases.

Private ``_*.py`` modules define the concrete schema classes.  This
``__init__`` re-exports them and defines the discriminated-union type
aliases (``DatasetConfig``, ``ExtractorConfig``, ``WorkflowConfig``)
consumed by :class:`~dataeval_flow.config._models.PipelineConfig`.
"""

from typing import Annotated

from pydantic import Field

from dataeval_flow.config.schemas._dataset import (
    CocoDatasetConfig,
    DatasetProtocolConfig,
    HuggingFaceDatasetConfig,
    ImageFolderDatasetConfig,
    YoloDatasetConfig,
)
from dataeval_flow.config.schemas._extractor import (
    BoVWExtractorConfig,
    FlattenExtractorConfig,
    OnnxExtractorConfig,
    TorchExtractorConfig,
    UncertaintyExtractorConfig,
)
from dataeval_flow.config.schemas._metadata import ResultMetadata
from dataeval_flow.config.schemas._preprocessor import PreprocessorConfig
from dataeval_flow.config.schemas._selection import SelectionConfig, SelectionStep
from dataeval_flow.config.schemas._task import (
    AutoBinMethod,
    DataCleaningTaskConfig,
    DriftMonitoringTaskConfig,
    OODDetectionTaskConfig,
    TaskConfig,
)
from dataeval_flow.config.schemas._workflow import (
    DataCleaningWorkflowConfig,
    DriftMonitoringWorkflowConfig,
    OODDetectionWorkflowConfig,
)

# -- discriminated-union aliases (internal) ---------------------------------

DatasetConfig = Annotated[
    HuggingFaceDatasetConfig | ImageFolderDatasetConfig | CocoDatasetConfig | YoloDatasetConfig,
    Field(discriminator="format"),
]

ExtractorConfig = Annotated[
    OnnxExtractorConfig
    | BoVWExtractorConfig
    | FlattenExtractorConfig
    | TorchExtractorConfig
    | UncertaintyExtractorConfig,
    Field(discriminator="model"),
]

WorkflowConfig = Annotated[
    DataCleaningWorkflowConfig | DriftMonitoringWorkflowConfig | OODDetectionWorkflowConfig,
    Field(discriminator="type"),
]

__all__ = [
    # Dataset
    "CocoDatasetConfig",
    "DatasetConfig",
    "DatasetProtocolConfig",
    "HuggingFaceDatasetConfig",
    "ImageFolderDatasetConfig",
    "YoloDatasetConfig",
    # Extractor
    "BoVWExtractorConfig",
    "ExtractorConfig",
    "FlattenExtractorConfig",
    "OnnxExtractorConfig",
    "TorchExtractorConfig",
    "UncertaintyExtractorConfig",
    # Workflow
    "DataCleaningWorkflowConfig",
    "DriftMonitoringWorkflowConfig",
    "OODDetectionWorkflowConfig",
    "WorkflowConfig",
    # Task
    "AutoBinMethod",
    "DataCleaningTaskConfig",
    "DriftMonitoringTaskConfig",
    "OODDetectionTaskConfig",
    "TaskConfig",
    # Other
    "PreprocessorConfig",
    "ResultMetadata",
    "SelectionConfig",
    "SelectionStep",
]
