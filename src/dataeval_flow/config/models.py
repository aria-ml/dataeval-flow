"""Workflow configuration models."""

__all__ = [
    "BoVWExtractorConfig",
    "ExtractorConfig",
    "FlattenExtractorConfig",
    "ModelConfig",
    "OnnxExtractorConfig",
    "TorchExtractorConfig",
    "UncertaintyExtractorConfig",
    "WorkflowConfig",
]

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from dataeval_flow.config.schemas import (
    DatasetConfig,
    DatasetProtocolConfig,
    PreprocessorConfig,
    SelectionConfig,
    TaskConfig,
)
from dataeval_flow.workflows.cleaning.params import DataCleaningParameters
from dataeval_flow.workflows.drift.params import DriftMonitoringParameters


class OnnxExtractorConfig(BaseModel):
    """OnnxExtractor configuration."""

    type: Literal["onnx"] = "onnx"
    model_path: str = Field(description="Path to ONNX model file.")
    output_name: str | None = Field(default=None, description="Output layer name. None = first output.")
    flatten: bool = Field(default=True, description="Flatten output to (N, D) shape.")


class BoVWExtractorConfig(BaseModel):
    """BoVWExtractor configuration. No model file required."""

    type: Literal["bovw"] = "bovw"
    vocab_size: int = Field(default=2048, ge=256, le=4096, description="Visual word count.")


class FlattenExtractorConfig(BaseModel):
    """FlattenExtractor configuration. No parameters."""

    type: Literal["flatten"] = "flatten"


class TorchExtractorConfig(BaseModel):
    """TorchExtractor configuration.

    Note: model must be loadable from model_path (e.g., torchscript or
    state_dict + class). Full programmatic model support deferred.
    """

    type: Literal["torch"] = "torch"
    model_path: str = Field(description="Path to saved PyTorch model.")
    layer_name: str | None = Field(default=None, description="Layer for forward hook extraction.")
    use_output: bool = Field(default=True, description="Capture layer output (True) or input (False).")
    device: str | None = Field(default=None, description="Device (e.g., 'cpu', 'cuda:0').")


class UncertaintyExtractorConfig(BaseModel):
    """ClassifierUncertaintyExtractor configuration.

    Note: same model_path constraint as TorchExtractorConfig.
    """

    type: Literal["uncertainty"] = "uncertainty"
    model_path: str = Field(description="Path to saved classification model.")
    preds_type: Literal["probs", "logits"] = Field(default="probs", description="Model output format.")
    batch_size: int = Field(default=32, description="Inference batch size.")
    device: str | None = Field(default=None, description="Device (e.g., 'cpu', 'cuda:0').")


# Discriminated union — Pydantic selects the right model based on `type` field.
ExtractorConfig = Annotated[
    OnnxExtractorConfig
    | BoVWExtractorConfig
    | FlattenExtractorConfig
    | TorchExtractorConfig
    | UncertaintyExtractorConfig,
    Field(discriminator="type"),
]


class ModelConfig(BaseModel):
    """Configuration for a model used for embedding extraction."""

    name: str = Field(description="Identifier for the model")
    extractor: ExtractorConfig = Field(description="Extractor configuration")


class LoggingConfig(BaseModel):
    """Logging level configuration."""

    app_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"
    lib_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "WARNING"


class WorkflowConfig(BaseModel):
    """Unified workflow configuration.

    One optional field per registered workflow.
    """

    # Logging
    logging: LoggingConfig | None = None

    # Workflow-specific params (one field per workflow)
    data_cleaning: DataCleaningParameters | None = None
    drift_monitoring: DriftMonitoringParameters | None = None

    # Shared config
    models: list[ModelConfig] | None = Field(
        default=None,
        description="Optional list of models for embedding extraction",
    )
    datasets: list[DatasetConfig | DatasetProtocolConfig] | None = None
    preprocessors: list[PreprocessorConfig] | None = None
    selections: list[SelectionConfig] | None = None
    tasks: list[TaskConfig] | None = None
