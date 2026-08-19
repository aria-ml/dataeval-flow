"""Logging configuration for the dataeval_flow package."""

import inspect
import logging
import os
import sys
import time
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

_initialized: bool = False
_APP_LOGGERS: tuple[str, ...] = ("dataeval_flow",)

# Third-party loggers carrying diagnostics we never want dropped as library
# noise.  DataEval reports several decisions it makes on the caller's behalf
# only as WARNING records — which factors it binned automatically and with what
# edges, and which arrays it could not resolve a ``value_range`` for (answered
# as NaN).  Both describe the results the run produced, so they are held at
# WARNING even when ``lib_level`` is raised to quiet genuinely noisy libraries.
_LIB_DIAGNOSTIC_LOGGERS: tuple[str, ...] = ("dataeval",)
_LIB_DIAGNOSTIC_CEILING = logging.WARNING

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

    # --- Library diagnostics pinned so they survive a raised lib_level ---
    for name in _LIB_DIAGNOSTIC_LOGGERS:
        logging.getLogger(name).setLevel(_LIB_DIAGNOSTIC_CEILING)


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

    # Keep DataEval's binning and value_range diagnostics at WARNING even when
    # lib_level is raised above it — they report what this run computed, not
    # library chatter.  A lib_level *below* WARNING still wins, so asking for
    # more detail works.
    for name in _LIB_DIAGNOSTIC_LOGGERS:
        logging.getLogger(name).setLevel(min(root_level, _LIB_DIAGNOSTIC_CEILING))


class _DiagnosticCollector(logging.Handler):
    """Collect library diagnostics as formatted strings, from logs and warnings alike.

    Two sources because DataEval uses two.  The per-factor detail stays on the logger;
    the advice a caller is meant to act on — which factors were binned with nobody's
    say-so, where a declared cut no longer fits the data, which encoding named a factor
    that is not one — is raised with :func:`warnings.warn`, because a ``NullHandler`` on
    the ``dataeval`` root logger means log records reach only callers who configured
    logging.  Capturing one and not the other would archive the footnotes and drop the
    finding.

    Deduplicates on the formatted message: DataEval already groups its per-datum
    warnings by message, but a pipeline running several workflows over one dataset
    re-raises the same diagnostic per workflow.
    """

    def __init__(self) -> None:
        super().__init__(level=_LIB_DIAGNOSTIC_CEILING)
        self.messages: list[str] = []
        self._seen: set[str] = set()

    def add(self, message: str) -> bool:
        """Record one message, answering whether it had not been seen before."""
        if message in self._seen:
            return False
        self._seen.add(message)
        self.messages.append(message)
        return True

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = f"{record.levelname}: {record.name}: {record.getMessage()}"
        except Exception:  # noqa: BLE001 - a handler must never propagate out of a log call
            return
        self.add(message)


def _library_root() -> str | None:
    """Directory the diagnostic library's own frames live in, or None if it is absent.

    Returned with a trailing separator, because the match below is a prefix test and this
    package installs *beside* the library it is testing for: without the separator
    ``site-packages/dataeval_flow/...`` matches the ``site-packages/dataeval`` root, and
    every warning raised anywhere under this package — which is all of them, since the
    matcher's own frame lives here — would be attributed to DataEval.
    """
    try:
        library = __import__(_LIB_DIAGNOSTIC_LOGGERS[0])
        return str(Path(library.__file__).parent) + os.sep if library.__file__ else None
    except Exception:  # noqa: BLE001 - no library is a reason to capture nothing, not to fail
        return None


def _raised_within(root: str) -> bool:
    """Whether any frame below this one belongs to *root*.

    The filename ``showwarning`` is handed is the *caller's*, not the library's: DataEval
    computes a ``stacklevel`` that points at the first frame outside itself, deliberately,
    so that "warn once" is keyed per calling site.  That makes the filename useless for
    telling its warnings from NumPy's, and the stack the only place the answer survives.
    ``warn`` calls ``showwarning`` synchronously, so the frames that raised it are still
    below this one.
    """
    frame = inspect.currentframe()
    while frame is not None:
        if frame.f_code.co_filename.startswith(root):
            return True
        frame = frame.f_back
    return False


@contextmanager
def capture_diagnostics() -> "Iterator[list[str]]":
    """Collect library diagnostics emitted inside the block.

    DataEval reports the decisions it makes on the caller's behalf — which factors it
    binned and with what edges, where a declared cut has stopped fitting the data, which
    arrays it could not resolve a ``value_range`` for — as log records and as warnings.
    Neither reaches the result envelope on its own, so a result archived alone cannot say
    whether a statistic came back ``NaN`` because the data said so or because nothing
    declared a range.  Capturing both here lets the envelope carry the answer.

    Warnings are re-emitted as they arrive, so a user watching the console still sees
    them — once each, which is what they saw before, for this library's warnings and for
    everybody else's alike.

    The inserted ``"always"`` filter is load-bearing rather than tidy.  :func:`warnings.warn`
    keeps its once-per-location bookkeeping in the globals of the frame its
    ``stacklevel`` selects, and DataEval points that at *its caller* — which is this
    package.  Two workflows building metadata through one line of flow therefore share a
    registry entry, and the second one's diagnostic would be suppressed before anything
    could record it, leaving its envelope quietly missing a finding that applies to it.
    Deduplication happens here instead, where it is per message rather than per line.

    Yields
    ------
    list[str]
        Messages collected so far.  The list is populated as records arrive and
        is complete once the block exits.
    """
    collector = _DiagnosticCollector()
    loggers = [logging.getLogger(name) for name in _LIB_DIAGNOSTIC_LOGGERS]
    for logger in loggers:
        logger.addHandler(collector)

    root = _library_root()

    with warnings.catch_warnings():
        # Restored on exit along with showwarning, both by catch_warnings itself.
        #
        # Inserted rather than `simplefilter`, and only for the category the diagnostics
        # use.  Replacing the whole filter list would also un-suppress every dependency's
        # DeprecationWarnings and neutralise a caller's `-W error` or pytest
        # `filterwarnings` for categories this block has no interest in — for the whole
        # duration of a workflow run.
        warnings.filterwarnings("always", category=UserWarning)
        previous = warnings.showwarning
        shown: set[tuple[str, str, int, str]] = set()

        def _show(message, category, filename, lineno, file=None, line=None) -> None:  # noqa: ANN001
            ours = root is not None and issubclass(category, UserWarning) and _raised_within(root)
            if ours:
                # Shown unless it is a repeat this block has already recorded, so the
                # console keeps the once-each behaviour the default filter gave it.
                if collector.add(f"{category.__name__}: {_LIB_DIAGNOSTIC_LOGGERS[0]}: {message}"):
                    previous(message, category, filename, lineno, file, line)
                return
            # Somebody else's warning. The filter above bypasses the once-per-location
            # registry for its category, so a library warning raised per sample would
            # otherwise print per sample; keyed here instead to restore what it had.
            key = (category.__name__, filename, lineno, str(message))
            if key not in shown:
                shown.add(key)
                previous(message, category, filename, lineno, file, line)

        warnings.showwarning = _show
        try:
            yield collector.messages
        finally:
            for logger in loggers:
                logger.removeHandler(collector)


def flush_logs() -> None:
    """Flush all root-logger handlers.

    Call after important checkpoints (e.g. after each task) so that
    buffered log records are written even if the process is killed.
    """
    for handler in logging.getLogger().handlers:
        handler.flush()
