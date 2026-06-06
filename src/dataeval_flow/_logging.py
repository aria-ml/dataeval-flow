"""Logging configuration for the dataeval_flow package."""

import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

_initialized: bool = False
_APP_LOGGERS: tuple[str, ...] = ("dataeval_flow",)

# Marker attributes used to make setup_logging additive and idempotent: each
# handler we attach is tagged with its role so repeat calls never duplicate it
# and the file handler can be added on a later call than the console handler.
_CONSOLE_ROLE = "_dataeval_flow_console"
_FILE_ROLE = "_dataeval_flow_file"

# Detailed format for the file log — full timestamp, level, and logger name.
_FILE_FORMAT = "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"
_FILE_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"


class _ConsoleFormatter(logging.Formatter):
    """Clean console formatter for CLI/container output.

    Strips the library-style prefix (timestamp, logger name) so user-facing
    output reads like plain program output.  INFO/DEBUG records render as just
    the message; WARNING and above are tagged with ``LEVEL:`` so problems stay
    visible.  Tracebacks (``exc_info``) are appended via the standard machinery.
    """

    def __init__(self) -> None:
        super().__init__("%(message)s")
        self._warn_formatter = logging.Formatter("%(levelname)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno >= logging.WARNING:
            return self._warn_formatter.format(record)
        return super().format(record)


class LogMessage:
    """Deferred message callback for logging expensive messages.

    Wrap an expensive string construction in ``LogMessage`` so it is only
    evaluated when the record is actually emitted (i.e. the level is enabled
    and a handler will format it)::

        _logger.debug(LogMessage(lambda: f"resolved: {[f.name for f in files]}"))
    """

    def __init__(self, fn: Callable[..., str]) -> None:
        self._fn = fn
        self._str: str | None = None

    def __str__(self) -> str:
        if self._str is None:
            self._str = self._fn()
        return self._str


def setup_logging(output_dir: Path | None = None, verbosity: int = 0) -> None:
    """Configure root logger with a clean console handler and optional file log.

    Additive and idempotent: the console (stdout) handler and the file handler
    are tagged with role markers, so this may be called more than once — first
    by the CLI to enable console output early, then by the runner to add the
    file handler once ``output_dir`` is known — without ever duplicating a
    handler.

    The console handler uses :class:`_ConsoleFormatter` (bare messages, with a
    ``LEVEL:`` prefix only for warnings and above) so CLI/container output reads
    like plain program output.  The file handler keeps the full timestamped,
    named format at DEBUG for diagnostics.

    Parameters
    ----------
    output_dir : Path | None
        Directory for the pipeline log file.  When ``None``, no file
        handler is created and output is console-only.
    verbosity : int
        Console verbosity level (0=quiet, 1=report, 2=+INFO, 3=+DEBUG).
    """
    global _initialized
    _initialized = True

    root = logging.getLogger()
    # Root stays at WARNING — third-party loggers inherit this level,
    # suppressing their DEBUG/INFO messages by default.

    # --- Console StreamHandler (clean format) — level driven by verbosity ---
    if not any(getattr(h, _CONSOLE_ROLE, False) for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        if verbosity >= 3:
            sh.setLevel(logging.DEBUG)
        elif verbosity >= 2:
            sh.setLevel(logging.INFO)
        else:
            sh.setLevel(logging.WARNING)
        sh.setFormatter(_ConsoleFormatter())
        setattr(sh, _CONSOLE_ROLE, True)
        root.addHandler(sh)

    # --- FileHandler (DEBUG, full format) — only when output_dir is provided ---
    if output_dir is not None and not any(getattr(h, _FILE_ROLE, False) for h in root.handlers):
        try:
            os.makedirs(output_dir, exist_ok=True)
            fh = logging.FileHandler(
                output_dir / "result.log",
                mode="w",
                encoding="utf-8",
            )
            fh.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATEFMT)
            file_formatter.converter = time.gmtime
            fh.setFormatter(file_formatter)
            setattr(fh, _FILE_ROLE, True)
            root.addHandler(fh)
        except OSError:
            pass  # fallback — StreamHandler still works if dir is unwritable

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
        Level for ``dataeval_flow`` loggers.
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
