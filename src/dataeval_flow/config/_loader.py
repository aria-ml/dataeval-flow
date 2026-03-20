"""Config loading - YAML/JSON single-file, multi-file merge, schema export."""

import json
import logging
import os
from pathlib import Path

import yaml

from dataeval_flow.config._models import PipelineConfig

__all__ = [
    "export_params_schema",
    "get_data_dir",
    "load_config",
    "load_config_folder",
    "resolve_path",
]

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(".")
_DATAEVAL_DATA_ENV = "DATAEVAL_DATA"


def get_data_dir(data_dir: Path | None = None) -> Path:
    """Resolve the data root directory.

    Priority: explicit argument > ``DATAEVAL_DATA`` env var > current directory.
    """
    if data_dir is not None:
        return data_dir
    return Path(os.environ.get(_DATAEVAL_DATA_ENV, str(DEFAULT_DATA_DIR)))


def resolve_path(relative: str | Path, data_dir: Path | None = None) -> Path:
    """Resolve a user-provided path against *data_dir*.

    Absolute paths are returned as-is.  Relative paths are joined to
    *data_dir* (which itself defaults via :func:`get_data_dir`).
    """
    p = Path(relative)
    if p.is_absolute():
        return p
    root = data_dir if data_dir is not None else get_data_dir()
    return root / p


def load_config(config_path: Path) -> PipelineConfig:
    """Load pipeline configuration from a single YAML or JSON file."""
    logger.debug("Loading config from %s", config_path)

    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f) if config_path.suffix.lower() == ".json" else yaml.safe_load(f) or {}

    return PipelineConfig.model_validate(data)


def load_config_folder(config_path: Path) -> PipelineConfig:
    """Load and merge all YAML/JSON files from config folder."""
    from dataeval_flow.config._merge import merge_config_folder

    logger.debug("Loading config folder %s", config_path)
    merged = merge_config_folder(config_path)
    return PipelineConfig.model_validate(merged)


def export_params_schema(output_path: Path) -> None:
    """Export JSON Schema for params.yaml (PipelineConfig) for IDE validation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = PipelineConfig.model_json_schema()
    output_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
