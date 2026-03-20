"""Logging configuration for the dataeval_flow package."""

import logging
import os
import sys
import time
from pathlib import Path

_initialized: bool = False
_APP_LOGGERS: tuple[str, ...] = ("dataeval_flow", "container_run")


def setup_logging(output_dir: Path | None = None, verbosity: int = 0) -> None:
    """Configure root logger with optional FileHandler + StreamHandler.

    Called once at startup, before config loads.  The module-level
    ``_initialized`` flag prevents duplicate handler attachment.

    Parameters
    ----------
    output_dir : Path | None
        Directory for the pipeline log file.  When ``None``, no file
        handler is created and output is console-only.
    verbosity : int
        Console verbosity level (0=quiet, 1=report, 2=+INFO, 3=+DEBUG).
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

    # --- FileHandler (DEBUG) — only when output_dir is provided ---
    if output_dir is not None:
        try:
            os.makedirs(output_dir, exist_ok=True)
            fh = logging.FileHandler(
                output_dir / "result.log",
                mode="w",
                encoding="utf-8",
            )
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            root.addHandler(fh)
        except OSError:
            pass  # fallback — StreamHandler still works if dir is unwritable

    # --- StreamHandler — level driven by verbosity ---
    sh = logging.StreamHandler(sys.stdout)
    if verbosity >= 3:
        sh.setLevel(logging.DEBUG)
    elif verbosity >= 2:
        sh.setLevel(logging.INFO)
    else:
        sh.setLevel(logging.WARNING)
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
        Level for ``dataeval_flow`` and ``container_run`` loggers.
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
