"""Run the ``dataeval-flow`` command in this process, capturing its exit code and output."""

from __future__ import annotations

from typing import Any


class Invocation:
    """What one in-process command-line invocation left behind: its exit code and what it printed."""

    def __init__(self, code: int, stdout: str, stderr: str) -> None:
        self.code, self.stdout, self.stderr = code, stdout, stderr

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


def reset_logging() -> None:
    """Remove the console and file handlers a command-line run attached, so the next run attaches its own."""
    import logging

    import dataeval_flow._logging as flow_logging

    flow_logging._initialized = False
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_dataeval_flow_console", False) or getattr(handler, "_dataeval_flow_file", False):
            handler.close()
            root.removeHandler(handler)
    root.setLevel(logging.WARNING)
    logging.getLogger("dataeval_flow").setLevel(logging.NOTSET)


def invoke(monkeypatch: Any, capsys: Any, *args: Any, env: dict[str, str] | None = None) -> Invocation:
    """Run ``dataeval-flow <args>`` in this process with only the ``DATAEVAL_*`` variables in *env*.

    Running in-process spares each call the seconds it takes to import the stack; the real command is run in a
    subprocess by the tests that check the process exit code itself.
    """
    import os
    import sys

    import pytest

    from dataeval_flow.__main__ import main
    from dataeval_flow._cache import DatasetCache

    reset_logging()
    DatasetCache.clear_instances()
    for name in [name for name in os.environ if name.startswith("DATAEVAL_")]:
        monkeypatch.delenv(name)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["dataeval_flow", *map(str, args)])
    with pytest.raises(SystemExit) as stopped:
        main()
    captured = capsys.readouterr()
    code = stopped.value.code
    return Invocation(code if isinstance(code, int) else 0, captured.out, captured.err)
