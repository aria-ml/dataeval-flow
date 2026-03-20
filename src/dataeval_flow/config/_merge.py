"""Multi-file YAML/JSON configuration loader with schema validation."""

import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger: logging.Logger = logging.getLogger(__name__)

_YAML_EXTS = frozenset({".yaml", ".yml"})
_JSON_EXTS = frozenset({".json"})
_CONFIG_EXTS = _YAML_EXTS | _JSON_EXTS


def _load_file(path: Path) -> dict[str, Any] | list[str]:
    """Load a single YAML or JSON file and return its contents as a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f) if path.suffix.lower() in _JSON_EXTS else yaml.safe_load(f) or []


def _is_valid_config(data: dict[str, Any]) -> bool:
    """Return True if *data* looks like a pipeline config fragment.

    A file is accepted when it contains at least one key that is a known
    ``PipelineConfig`` field **and** no keys that are completely unknown.
    This allows partial configs (e.g. only ``datasets:``) while rejecting
    unrelated files like JSON schemas.
    """
    from dataeval_flow.config._models import PipelineConfig

    known_keys = set(PipelineConfig.model_fields)
    file_keys = set(data.keys())
    return bool(file_keys) and file_keys <= known_keys


def merge_config_folder(config_path: Path) -> dict[str, Any]:
    """Scan folder, merge all valid config files alphabetically.

    Each candidate file is validated against the ``PipelineConfig`` schema
    individually.  Files that fail validation (e.g. JSON schemas, unrelated
    YAML) are silently skipped.  If no valid files are found, a
    ``FileNotFoundError`` is raised.

    Files are loaded in sorted order (00-base.yaml before 01-datasets.yaml).
    Later files override earlier ones for duplicate keys.

    Returns raw dict - use load_config_folder() for validated PipelineConfig.
    """
    config: dict[str, Any] = {}

    if not config_path.is_dir():
        raise ValueError(f"Config path is not a directory: {config_path}")

    candidates = sorted(f for f in config_path.iterdir() if f.is_file() and f.suffix.lower() in _CONFIG_EXTS)
    logger.debug("Found %d candidate file(s): %s", len(candidates), [f.name for f in candidates])

    accepted: list[Path] = []
    for config_file in candidates:
        try:
            file_config = _load_file(config_file)
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            logger.debug("Skipping %s (parse error: %s)", config_file.name, exc)
            continue

        if not isinstance(file_config, dict) or not _is_valid_config(file_config):
            logger.debug("Skipping %s (not a valid pipeline config)", config_file.name)
            continue

        _deep_merge(config, file_config)
        accepted.append(config_file)

    if not accepted:
        raise FileNotFoundError(f"No valid pipeline config files found in {config_path}")

    logger.debug("Accepted %d config file(s): %s", len(accepted), [f.name for f in accepted])
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
