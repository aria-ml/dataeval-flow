"""Environment variable parsing for CLI options.

Functions return ``None`` when the variable is unset or blank. Invalid values
raise ``ValueError``.
"""

import os
from pathlib import Path

__all__ = ["env_bool", "env_choice", "env_int", "env_list", "env_path"]

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def _raw(name: str) -> str | None:
    """Return the stripped value of *name*, or None when unset or blank."""
    value = os.environ.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def env_path(name: str) -> Path | None:
    """Read *name* as a filesystem path."""
    raw = _raw(name)
    return Path(raw) if raw is not None else None


def env_int(name: str) -> int | None:
    """Read *name* as an integer, raising ValueError if invalid."""
    raw = _raw(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def env_bool(name: str) -> bool | None:
    """Read *name* as a boolean, raising ValueError if unrecognized."""
    raw = _raw(name)
    if raw is None:
        return None
    lowered = raw.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    accepted = ", ".join((*_TRUE, *_FALSE))
    raise ValueError(f"{name} must be one of {accepted}; got {raw!r}")


def env_list(name: str) -> list[str] | None:
    """Read *name* as a comma-separated list, raising ValueError if empty."""
    raw = _raw(name)
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} was set but lists no values: {raw!r}")
    return items


def env_choice(name: str, choices: tuple[str, ...]) -> str | None:
    """Read *name* and validate that its value is in *choices*."""
    raw = _raw(name)
    if raw is None:
        return None
    if raw not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}; got {raw!r}")
    return raw
