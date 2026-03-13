"""Multi-file YAML configuration loader."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger: logging.Logger = logging.getLogger(__name__)


def merge_yaml_folder(config_path: Path) -> dict[str, Any]:
    """Scan folder, merge all YAML files alphabetically.

    Files are loaded in sorted order (00-base.yaml before 01-datasets.yaml).
    Later files override earlier ones for duplicate keys.

    Returns raw dict - use load_config_folder() for validated WorkflowConfig.
    """
    config: dict[str, Any] = {}

    if not config_path.is_dir():
        raise ValueError(f"Config path is not a directory: {config_path}")

    yaml_files = sorted(list(config_path.glob("*.yaml")) + list(config_path.glob("*.yml")))
    logger.debug("Found %d YAML file(s): %s", len(yaml_files), [f.name for f in yaml_files])

    for yaml_file in yaml_files:
        with open(yaml_file, encoding="utf-8") as f:
            file_config = yaml.safe_load(f) or {}
            _deep_merge(config, file_config)

    return config


def _deep_merge(base: dict, overlay: dict) -> None:
    """Recursively merge overlay into base.

    Rules: dicts merge recursively, lists extend, scalars replace.
    """
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        elif key in base and isinstance(base[key], list) and isinstance(value, list):
            base[key].extend(value)
        else:
            base[key] = value
