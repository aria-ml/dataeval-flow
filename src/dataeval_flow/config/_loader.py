"""Config loading - YAML single-file, multi-file merge, schema export."""

import json
import logging
from pathlib import Path

import yaml

from dataeval_flow.config._models import PipelineConfig

__all__ = ["export_params_schema", "load_config", "load_config_folder"]

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_PARAMS_PATH = Path("/data/config/params.yaml")
DEFAULT_CONFIG_FOLDER = Path("/data/config")


def load_config(config_path: Path | None = None) -> PipelineConfig:
    """Load pipeline configuration from a single YAML file."""
    path = config_path or DEFAULT_PARAMS_PATH
    logger.debug("Loading config from %s", path)

    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return PipelineConfig.model_validate(data)


def load_config_folder(config_path: Path | None = None) -> PipelineConfig:
    """Load and merge all YAML files from config folder."""
    from dataeval_flow.config._merge import merge_yaml_folder

    path = config_path or DEFAULT_CONFIG_FOLDER
    logger.debug("Loading config folder %s", path)
    merged = merge_yaml_folder(path)
    return PipelineConfig.model_validate(merged)


def export_params_schema(output_path: Path) -> None:
    """Export JSON Schema for params.yaml (PipelineConfig) for IDE validation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = PipelineConfig.model_json_schema()
    output_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
