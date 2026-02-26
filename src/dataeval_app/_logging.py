"""Logging configuration for the dataeval_app package."""

import logging
import os
import sys
import time
from pathlib import Path

_initialized: bool = False
_APP_LOGGERS: tuple[str, ...] = ("dataeval_app", "container_run")


def setup_logging(output_dir: Path) -> None:
    """Configure root logger with FileHandler (DEBUG) + StreamHandler (INFO).

    Called once at startup, before config loads.  The module-level
    ``_initialized`` flag prevents duplicate handler attachment.

    Parameters
    ----------
    output_dir : Path
        Directory for ``log.txt``.  Created if it does not exist.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    fmt = "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%SZ"
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    formatter.converter = time.gmtime

    root = logging.getLogger()
    # Root stays at WARNING — third-party loggers inherit this level,
    # suppressing their DEBUG/INFO messages by default.

    # --- FileHandler (DEBUG) ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(
            output_dir / "log.txt",
            mode="w",
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)
    except OSError:
        pass  # fallback — StreamHandler still works if dir is unwritable

    # --- StreamHandler (INFO) ---
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # --- App loggers at DEBUG ---
    for name in _APP_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG)


def configure_log_levels(
    app_level: str = "DEBUG",
    lib_level: str = "WARNING",
) -> None:
    """Apply config-driven log level overrides.

    Called after config loads so that user YAML settings take effect.

    Parameters
    ----------
    app_level : str
        Level for ``dataeval_app`` and ``container_run`` loggers.
    lib_level : str
        Level for root logger (controls third-party effective level).
    """
    level = getattr(logging, app_level, logging.DEBUG)
    for name in _APP_LOGGERS:
        logging.getLogger(name).setLevel(level)

    root_level = getattr(logging, lib_level, logging.WARNING)
    logging.getLogger().setLevel(root_level)


def flush_logs() -> None:
    """Flush all root-logger handlers.

    Call after important checkpoints (e.g. after each task) so that
    buffered log records are written even if the process is killed.
    """
    for handler in logging.getLogger().handlers:
        handler.flush()
