"""Config path validation and relativization utilities.

These are separated from ``_loader`` to avoid circular imports — the
schema modules (``_dataset``, ``_extractor``) need them at import time,
while ``_loader`` depends on ``_models`` which depends on schemas.
"""

import os
from pathlib import Path, PurePosixPath

__all__ = [
    "relativize_to_data_dir",
    "validate_config_path",
]


def validate_config_path(value: str) -> str:
    """Validate that a config path is relative and stays under the data root.

    Config paths (dataset ``path``, extractor ``model_path``) must be
    stored as relative paths so that configs remain portable across
    machines and into containers where the data root is mounted at an
    arbitrary location (e.g. ``/dataeval``).

    Raises
    ------
    ValueError
        If *value* is an absolute path or escapes upward via ``..``.
    """
    if not value:
        raise ValueError("Path must not be empty")

    p = PurePosixPath(value)
    if p.is_absolute():
        raise ValueError(
            f"Config paths must be relative (got absolute path '{value}'). "
            "Use a path relative to the data root instead."
        )

    # Normalize and check for upward escape via ".."
    normalized = os.path.normpath(value)
    if normalized.startswith(".."):
        raise ValueError(
            f"Config path '{value}' escapes the data root directory. "
            "All paths must resolve to locations under the data root."
        )

    return value


def relativize_to_data_dir(absolute_path: str | Path, data_dir: Path) -> str:
    """Convert an absolute path to a relative path under *data_dir*.

    Used by the TUI browse to convert OS-absolute paths into portable
    relative paths suitable for config storage.

    Parameters
    ----------
    absolute_path : str | Path
        The absolute path to relativize.
    data_dir : Path
        The data root to relativize against.

    Returns
    -------
    str
        A relative path string under *data_dir*.

    Raises
    ------
    ValueError
        If *absolute_path* is not under *data_dir*.
    """
    p = Path(absolute_path).resolve()
    root = data_dir.resolve()

    try:
        rel = p.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Path '{absolute_path}' is not under the data root '{root}'. "
            "Config paths must point to locations inside the data directory."
        ) from None

    return str(rel)
