"""Config loading - a YAML/JSON file, or a folder of them merged."""

import json
import logging
import os
from pathlib import Path

import yaml

from dataeval_flow.config._models import PipelineConfig
from dataeval_flow.config._paths import relativize_to_data_dir, validate_config_path

__all__ = [
    "get_data_dir",
    "load_config",
    "relativize_to_data_dir",
    "resolve_path",
    "validate_config_path",
]

_logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(".")
_DATAEVAL_DATA_ENV = "DATAEVAL_DATA"


def get_data_dir(data_dir: Path | None = None) -> Path:
    """Resolve the data root directory.

    Priority: explicit argument > ``DATAEVAL_DATA`` env var > current directory.
    """
    if data_dir is not None:
        return data_dir
    return Path(os.environ.get(_DATAEVAL_DATA_ENV, str(DEFAULT_DATA_DIR)))


def resolve_path(relative: str | Path, data_dir: Path | None = None, *, default_subdir: str | None = None) -> Path:
    """Resolve a user-provided path against *data_dir*.

    Absolute paths are returned as-is.  Relative paths are joined to
    *data_dir* (which itself defaults via :func:`get_data_dir`).

    ``default_subdir`` implements the mount-folder conventions: when a relative
    path is not found directly under the data root, the conventional subfolder
    is tried (``data`` for datasets, ``models`` for models). Explicit paths
    that resolve directly are unchanged, so existing configs keep working; new
    configs may use the conventional folders and reference them by bare name.
    """
    p = Path(relative)
    if p.is_absolute():
        return p
    root = data_dir if data_dir is not None else get_data_dir()
    direct = root / p
    if default_subdir is not None and not direct.exists():
        conventional = root / default_subdir / p
        if conventional.exists():
            return conventional
    return direct


def load_config(path: Path | str) -> PipelineConfig:
    """Load a pipeline config from a YAML or JSON file, or from a folder of them.

    Parameters
    ----------
    path : Path or str
        A ``.yaml``, ``.yml`` or ``.json`` file, or a folder. A folder's config files are merged in name
        order: mappings merge, lists extend and a later file's scalar replaces an earlier one's. Files in the
        folder that are not pipeline configs are skipped.

    Returns
    -------
    PipelineConfig
        The validated config.

    Raises
    ------
    FileNotFoundError
        When `path` does not exist, or names a folder holding no pipeline config file.
    pydantic.ValidationError
        When the config is invalid.

    Examples
    --------
    >>> from dataeval_flow import load_config
    >>> config = load_config("params.yaml")  # doctest: +SKIP
    >>> config = load_config("configs/")  # doctest: +SKIP
    """
    path = Path(path)
    if path.is_dir():
        from dataeval_flow.config._merge import merge_config_folder

        _logger.debug("Loading config folder %s", path)
        return PipelineConfig.model_validate(merge_config_folder(path))

    _logger.debug("Loading config from %s", path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with open(path, encoding="utf-8") as f:
        data = json.load(f) if path.suffix.lower() == ".json" else yaml.safe_load(f) or {}

    return PipelineConfig.model_validate(data)
