"""Schema catalog — concrete types and discriminated-union aliases.

Private ``_*.py`` modules define the concrete schema classes.  This
``__init__`` re-exports them and defines the discriminated-union type
alias ``DatasetConfig`` consumed by
:class:`~dataeval_flow.PipelineConfig`. Workflow, evaluator and
extractor configs are not listed here: ``PipelineConfig`` validates each
entry with the config class its registered type names.
"""

from typing import Annotated

from pydantic import Field

from dataeval_flow.config._schemas._dataset import (
    CocoDatasetConfig,
    DatasetProtocolConfig,
    DemoDatasetConfig,
    HuggingFaceDatasetConfig,
    ImageFolderDatasetConfig,
    YoloDatasetConfig,
)
from dataeval_flow.config._schemas._export import ExportConfig
from dataeval_flow.config._schemas._metadata import (
    AggregatorConfig,
    MetadataPolicyConfig,
    ParseDateTimeCorrectionConfig,
    ParseValueCorrectionConfig,
    ReductionOptionsConfig,
    RemapCorrectionConfig,
    RemapRuleConfig,
    RescaleCorrectionConfig,
)
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin, StatsConfigMixin
from dataeval_flow.config._schemas._ontology import OntologyConceptConfig, OntologyConfig
from dataeval_flow.config._schemas._preprocessor import PreprocessingStep, PreprocessorConfig
from dataeval_flow.config._schemas._stats import StatsMeasureConfig, StatsPolicyConfig
from dataeval_flow.config._schemas._task import AutoBinMethod, FactorSource, TaskConfig, TaskKind
from dataeval_flow.config._schemas._view import (
    ViewConfig,
    ViewOperation,
)

# -- discriminated-union aliases (internal) ---------------------------------

DatasetConfig = Annotated[
    HuggingFaceDatasetConfig | ImageFolderDatasetConfig | CocoDatasetConfig | YoloDatasetConfig | DemoDatasetConfig,
    Field(discriminator="format"),
]

__all__ = [
    # Dataset
    "CocoDatasetConfig",
    "DatasetConfig",
    "DatasetProtocolConfig",
    "DemoDatasetConfig",
    "HuggingFaceDatasetConfig",
    "ImageFolderDatasetConfig",
    "YoloDatasetConfig",
    # Task
    "AutoBinMethod",
    "MetadataPolicyConfig",
    "OntologyConfig",
    "OntologyConceptConfig",
    "FactorSource",
    "TaskConfig",
    "TaskKind",
    # Metadata policy — corrections and aggregation
    "AggregatorConfig",
    "ParseDateTimeCorrectionConfig",
    "ParseValueCorrectionConfig",
    "ReductionOptionsConfig",
    "RemapCorrectionConfig",
    "RemapRuleConfig",
    "RescaleCorrectionConfig",
    # Mixins
    "MetadataConfigMixin",
    "StatsConfigMixin",
    # Other
    "ExportConfig",
    "PreprocessingStep",
    "PreprocessorConfig",
    "StatsMeasureConfig",
    "StatsPolicyConfig",
    "ViewConfig",
    "ViewOperation",
]
