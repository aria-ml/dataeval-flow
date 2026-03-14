"""Pipeline and workflow configuration models."""

__all__ = [
    "BoVWExtractorConfig",
    "DataCleaningWorkflowConfig",
    "DriftMonitoringWorkflowConfig",
    "ExtractorConfig",
    "FlattenExtractorConfig",
    "ModelConfig",
    "OnnxExtractorConfig",
    "PipelineConfig",
    "TorchExtractorConfig",
    "UncertaintyExtractorConfig",
    "WorkflowConfig",
]

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from dataeval_flow.config.schemas import (
    DataCleaningWorkflowConfig,
    DatasetConfig,
    DatasetProtocolConfig,
    DriftMonitoringWorkflowConfig,
    PreprocessorConfig,
    SelectionConfig,
    TaskConfig,
    WorkflowConfig,
)


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


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration.

    All sections use a define-once, reference-by-name pattern.
    Workflows bind a workflow type to specific parameters;
    tasks reference workflows and resource pools by name.
    """

    # Logging
    logging: LoggingConfig | None = None

    # Named resource pools
    datasets: Sequence[DatasetConfig | DatasetProtocolConfig] | None = None
    models: Sequence[ModelConfig] | None = Field(
        default=None,
        description="Named models for embedding extraction",
    )
    preprocessors: Sequence[PreprocessorConfig] | None = None
    selections: Sequence[SelectionConfig] | None = None
    workflows: Sequence[WorkflowConfig] | None = Field(
        default=None,
        description="Named workflow configurations (type + params), referenced by tasks",
    )
    tasks: Sequence[TaskConfig] | None = None
