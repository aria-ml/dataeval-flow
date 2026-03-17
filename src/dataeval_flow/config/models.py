"""Pipeline and workflow configuration models."""

__all__ = [
    "DataCleaningWorkflowConfig",
    "DriftMonitoringWorkflowConfig",
    "ExtractorConfig",
    "PipelineConfig",
    "SourceConfig",
    "WorkflowConfig",
]

from collections.abc import Sequence
from typing import Literal

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

# ---------------------------------------------------------------------------
# Source — dataset + optional selection
# ---------------------------------------------------------------------------


class SourceConfig(BaseModel):
    """Named source definition — bundles a dataset with an optional selection.

    YAML example::

        sources:
          - name: cifar_train_subset
            dataset: cifar10_train
            selection: first_5k
    """

    name: str = Field(description="Identifier for the source")
    dataset: str = Field(description="Reference to a dataset name")
    selection: str | None = Field(default=None, description="Reference to a selection name (optional)")


# ---------------------------------------------------------------------------
# Extractor — model type + params + optional preprocessor + batch_size
# ---------------------------------------------------------------------------


class ExtractorConfig(BaseModel):
    """Named extractor definition.

    Combines model type, model-specific parameters, optional preprocessor,
    and batch size into a single named unit. The ``model`` field acts as a
    discriminator for which model-specific parameters are relevant.

    YAML examples::

        extractors:
          - name: resnet_extractor
            model: onnx
            model_path: "./resnet50.onnx"
            output_name: "flatten0"
            preprocessor: resnet_preprocess
            batch_size: 64

          - name: flat_extractor
            model: flatten

          - name: bovw_extractor
            model: bovw
            vocab_size: 1024
            batch_size: 32
    """

    name: str = Field(description="Identifier for the extractor")
    model: Literal["onnx", "bovw", "flatten", "torch", "uncertainty"] = Field(
        description="Model/extractor type",
    )
    # Composition references
    preprocessor: str | None = Field(default=None, description="Reference to a preprocessor name (optional)")
    batch_size: int | None = Field(default=None, description="Batch size for embedding extraction")
    # onnx params
    model_path: str | None = Field(default=None, description="Path to model file (onnx/torch/uncertainty).")
    output_name: str | None = Field(default=None, description="Output layer name (onnx).")
    flatten: bool = Field(default=True, description="Flatten output to (N, D) shape (onnx).")
    # bovw params
    vocab_size: int | None = Field(default=None, ge=256, le=4096, description="Visual word count (bovw).")
    # torch params
    layer_name: str | None = Field(default=None, description="Layer for forward hook extraction (torch).")
    use_output: bool = Field(default=True, description="Capture layer output (True) or input (False) (torch).")
    # torch / uncertainty params
    device: str | None = Field(default=None, description="Device (e.g., 'cpu', 'cuda:0').")
    # uncertainty params
    preds_type: Literal["probs", "logits"] | None = Field(
        default=None, description="Model output format (uncertainty)."
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class LoggingConfig(BaseModel):
    """Logging level configuration."""

    app_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"
    lib_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "WARNING"


# ---------------------------------------------------------------------------
# Pipeline (top-level)
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration.

    All sections use a define-once, reference-by-name pattern.
    Sources compose datasets with optional selections; extractors
    compose model type/params with optional preprocessors.
    Tasks reference workflows, sources, and extractors by name.
    """

    # Logging
    logging: LoggingConfig | None = None

    # Named resource pools
    datasets: Sequence[DatasetConfig | DatasetProtocolConfig] | None = None
    preprocessors: Sequence[PreprocessorConfig] | None = None
    selections: Sequence[SelectionConfig] | None = None

    # Composition layers
    sources: Sequence[SourceConfig] | None = Field(
        default=None,
        description="Named source definitions (dataset + optional selection)",
    )
    extractors: Sequence[ExtractorConfig] | None = Field(
        default=None,
        description="Named extractor definitions (model type + params + optional preprocessor + batch_size)",
    )

    # Execution
    workflows: Sequence[WorkflowConfig] | None = Field(
        default=None,
        description="Named workflow configurations (type + params), referenced by tasks",
    )
    tasks: Sequence[TaskConfig] | None = None
