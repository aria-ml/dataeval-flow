"""Config layer — the pipeline config, and everything a pipeline file describes besides its workflows and evaluators.

The pipeline config and its plain sections' types are imported from here::

    from dataeval_flow.config import PipelineConfig, SourceConfig, TaskConfig, ...

A section whose entries are plugins has a subpackage of its own: ``dataeval_flow.config.extractors`` for
``extractors:``, and ``dataeval_flow.config.image_transforms`` for what a preprocessor's ``step:`` names.

``load_config`` is imported from ``dataeval_flow``; a workflow's or evaluator's config from its type's package under
``dataeval_flow.workflows`` or ``dataeval_flow.evaluators``.
"""

__all__ = [
    # The pipeline
    "PipelineConfig",
    # Dataset configs
    "CocoDatasetConfig",
    "DatasetConfig",
    "DatasetProtocolConfig",
    "DemoDatasetConfig",
    "HuggingFaceDatasetConfig",
    "ImageFolderDatasetConfig",
    "YoloDatasetConfig",
    # Task configs
    "TaskConfig",
    "TaskKind",
    # Composition
    "SourceConfig",
    # Logging
    "LoggingConfig",
    # Results
    "ResultConfig",
    # Metadata policy — corrections and aggregation
    "AggregatorConfig",
    "MetadataPolicyConfig",
    "ParseDateTimeCorrectionConfig",
    "ParseValueCorrectionConfig",
    "ReductionOptionsConfig",
    "RemapCorrectionConfig",
    "RemapRuleConfig",
    "RescaleCorrectionConfig",
    # Ontology
    "OntologyConceptConfig",
    "OntologyConfig",
    # Export / stats
    "ExportConfig",
    "StatsMeasureConfig",
    "StatsPolicyConfig",
    # Config mixins
    "MetadataConfigMixin",
    "StatsConfigMixin",
    # Other schemas
    "PreprocessingStep",
    "PreprocessorConfig",
    "ViewConfig",
    "ViewOperation",
]

from dataeval_flow.config._models import LoggingConfig, PipelineConfig, ResultConfig, SourceConfig
from dataeval_flow.config._schemas import (
    AggregatorConfig,
    CocoDatasetConfig,
    DatasetConfig,
    DatasetProtocolConfig,
    DemoDatasetConfig,
    ExportConfig,
    HuggingFaceDatasetConfig,
    ImageFolderDatasetConfig,
    MetadataPolicyConfig,
    OntologyConceptConfig,
    OntologyConfig,
    ParseDateTimeCorrectionConfig,
    ParseValueCorrectionConfig,
    PreprocessingStep,
    PreprocessorConfig,
    ReductionOptionsConfig,
    RemapCorrectionConfig,
    RemapRuleConfig,
    RescaleCorrectionConfig,
    StatsMeasureConfig,
    StatsPolicyConfig,
    TaskConfig,
    TaskKind,
    ViewConfig,
    ViewOperation,
    YoloDatasetConfig,
)
from dataeval_flow.config._schemas._mixins import MetadataConfigMixin, StatsConfigMixin
