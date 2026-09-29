"""Multi-file YAML/JSON configuration loader with schema validation."""

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from dataeval_flow._logging import LogMessage

_logger: logging.Logger = logging.getLogger(__name__)

_YAML_EXTS = frozenset({".yaml", ".yml"})
_JSON_EXTS = frozenset({".json"})
_CONFIG_EXTS = _YAML_EXTS | _JSON_EXTS


def _load_file(path: Path) -> dict[str, Any] | list[str]:
    """Load a single YAML or JSON file and return its contents as a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f) if path.suffix.lower() in _JSON_EXTS else yaml.safe_load(f) or []


def _is_config_fragment(data: dict[str, Any]) -> bool:
    """Whether *data* is a pipeline config fragment: it holds at least one top-level section.

    A file holding none, such as a JSON schema or a compose file, is unrelated and skipped. A fragment's other keys
    must all be sections too, which :func:`merge_config_folder` checks, so a misspelled section is refused rather than
    dropping its whole file.
    """
    from dataeval_flow.config._models import top_level_keys

    return not top_level_keys().isdisjoint(data)


def merge_config_folder(config_path: Path) -> dict[str, Any]:
    """Scan folder, merge all valid config files alphabetically.

    A file holding no top-level section (e.g. a JSON schema, unrelated YAML)
    is skipped. A file holding a section beside a key that is none is refused,
    naming the file and the key, since the key is most likely a misspelled
    section. If no config files are found, a ``FileNotFoundError`` is raised.

    Files are loaded in sorted order (00-base.yaml before 01-datasets.yaml).
    Later files override earlier ones for duplicate keys.

    Returns raw dict - use load_config() for validated PipelineConfig.
    """
    from dataeval_flow.config._models import unknown_keys_problem

    config: dict[str, Any] = {}

    if not config_path.is_dir():
        raise ValueError(f"Config path is not a directory: {config_path}")

    candidates = sorted(f for f in config_path.iterdir() if f.is_file() and f.suffix.lower() in _CONFIG_EXTS)
    _logger.debug(LogMessage(lambda: f"Found {len(candidates)} candidate file(s): {[f.name for f in candidates]}"))

    accepted: list[Path] = []
    for config_file in candidates:
        try:
            file_config = _load_file(config_file)
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            _logger.debug("Skipping %s (parse error: %s)", config_file.name, exc)
            continue

        if not isinstance(file_config, dict) or not _is_config_fragment(file_config):
            _logger.debug("Skipping %s (not a pipeline config)", config_file.name)
            continue
        problem = unknown_keys_problem(file_config)
        if problem is not None:
            raise ValueError(f"{config_file.name}: {problem}")

        _deep_merge(config, file_config)
        accepted.append(config_file)

    if not accepted:
        raise FileNotFoundError(f"No valid pipeline config files found in {config_path}")

    _logger.debug(LogMessage(lambda: f"Accepted {len(accepted)} config file(s): {[f.name for f in accepted]}"))
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
