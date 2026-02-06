"""Workflow parameters - user-facing configuration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from dataeval_app.ingest.schemas import (
    DatasetConfig,
    PreprocessorConfig,
    SelectionConfig,
    TaskConfig,
)

logger: logging.Logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Base classes
# -----------------------------------------------------------------------------


class WorkflowParametersBase(BaseModel):
    """Base class for all workflow parameters."""

    mode: Literal["advisory", "preparatory"] = Field(
        default="advisory",
        description="advisory: report only, preparatory: modify dataset",
    )


# -----------------------------------------------------------------------------
# Data Cleaning workflow
# -----------------------------------------------------------------------------


class DataCleaningParameters(WorkflowParametersBase):
    """Parameters for data cleaning workflow.

    Required parameters must be explicitly set per CR-4.14-G-1
    (avoid application-specific defaults).
    """

    # REQUIRED - no defaults per CR-4.14-G-1
    outlier_method: Literal["zscore", "modzscore", "iqr"] = Field(
        description="Statistical method for outlier detection",
    )
    outlier_use_dimension: bool = Field(
        description="Use dimension statistics for outlier detection",
    )
    outlier_use_pixel: bool = Field(
        description="Use pixel statistics for outlier detection",
    )
    outlier_use_visual: bool = Field(
        description="Use visual statistics for outlier detection",
    )

    # Threshold - required, but None means "use DataEval default"
    outlier_threshold: float | None = Field(
        ge=0.0,
        description="Custom threshold (None = use DataEval default for chosen method)",
    )


# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Configuration for a single ONNX model.

    Pass-through configuration for DataEval's OnnxEncoder.
    """

    name: str = Field(description="Identifier for the model")
    path: str = Field(description="Path to the ONNX model file")
    embedding_layer: str = Field(description="Layer name for embedding extraction")


# -----------------------------------------------------------------------------
# Unified workflow configuration
# -----------------------------------------------------------------------------


class WorkflowConfig(BaseModel):
    """Unified workflow configuration.

    Contains all configuration sections for the workflow.
    Extended with new schemas (P1): datasets, preprocessors, selections, tasks.
    """

    # Existing fields (from PRs #1-5)
    data_cleaning: DataCleaningParameters | None = None
    models: list[ModelConfig] | None = Field(
        default=None,
        description="Optional list of ONNX models for embedding extraction",
    )

    # New fields (P1)
    datasets: list[DatasetConfig] | None = None
    preprocessors: list[PreprocessorConfig] | None = None
    selections: list[SelectionConfig] | None = None  # Named selection pipelines
    tasks: list[TaskConfig] | None = None


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

# Keep original path for backward compat (single-file loading)
DEFAULT_PARAMS_PATH = Path("/data/config/params.yaml")

# New default for multi-file folder loading
DEFAULT_CONFIG_FOLDER = Path("/data/config")


def load_config(config_path: Path | None = None) -> WorkflowConfig:
    """Load unified workflow configuration from YAML.

    Parameters
    ----------
    config_path : Path | None
        Path to config file. Uses /data/config/params.yaml if None.

    Returns
    -------
    WorkflowConfig
        Validated configuration with all sections.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    pydantic.ValidationError
        If required parameters are missing or invalid.
    """
    path = config_path or DEFAULT_PARAMS_PATH

    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return WorkflowConfig.model_validate(data)


def load_params(params_path: Path | None = None) -> DataCleaningParameters:
    """Load data cleaning parameters from 'data_cleaning' section.

    Parameters
    ----------
    params_path : Path | None
        Path to config file. Uses /data/config/params.yaml if None.

    Returns
    -------
    DataCleaningParameters
        Validated parameters instance.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If 'data_cleaning' section is missing.
    pydantic.ValidationError
        If required parameters are missing or invalid.
    """
    path = params_path or DEFAULT_PARAMS_PATH

    if not path.exists():
        msg = f"Parameters file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "data_cleaning" not in data:
        msg = (
            f"'data_cleaning' section not found in {path}. "
            f"Config must have 'data_cleaning:' section with required fields."
        )
        raise ValueError(msg)

    return DataCleaningParameters.model_validate(data["data_cleaning"])


def export_params_schema(output_path: Path) -> None:
    """Export JSON Schema for params.yaml (WorkflowConfig) for IDE validation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = WorkflowConfig.model_json_schema()
    output_path.write_text(json.dumps(schema, indent=2))


def load_config_folder(config_path: Path | None = None) -> WorkflowConfig:
    """Load and merge all YAML files from config folder.

    Replaces load_config() for multi-file support.

    Parameters
    ----------
    config_path : Path | None
        Path to config folder. Uses /data/config if None.

    Returns
    -------
    WorkflowConfig
        Validated configuration with all sections merged.

    Raises
    ------
    ValueError
        If the config path is not a directory.
    pydantic.ValidationError
        If merged configuration is invalid.
    """
    from dataeval_app.ingest.config_loader import merge_yaml_folder

    path = config_path or DEFAULT_CONFIG_FOLDER
    merged = merge_yaml_folder(path)
    return WorkflowConfig.model_validate(merged)
