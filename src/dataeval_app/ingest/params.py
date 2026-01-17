"""Workflow parameters - user-facing configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

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
# Utility functions
# -----------------------------------------------------------------------------

DEFAULT_PARAMS_PATH = Path("/data/config/params.yaml")


def load_params(params_path: Path | None = None) -> DataCleaningParameters:
    """Load and validate data cleaning parameters from YAML file.

    Parameters
    ----------
    params_path : Path | None
        Path to parameters file. Uses /data/config/params.yaml if None.

    Returns
    -------
    DataCleaningParameters
        Validated parameters instance.

    Raises
    ------
    FileNotFoundError
        If the parameters file does not exist.
    pydantic.ValidationError
        If required parameters are missing or invalid.
    """
    path = params_path or DEFAULT_PARAMS_PATH

    if not path.exists():
        msg = f"Parameters file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return DataCleaningParameters.model_validate(data)


def export_params_schema(output_path: Path) -> None:
    """Export JSON Schema for params.yaml IDE validation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = DataCleaningParameters.model_json_schema()
    output_path.write_text(json.dumps(schema, indent=2))
