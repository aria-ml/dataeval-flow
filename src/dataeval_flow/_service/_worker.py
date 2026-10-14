"""A queued run's process: the batch command, which ends its process group should the service that started it go."""

import contextlib
import os
import signal
import sys
import threading
import time
from pathlib import Path

from dataeval_flow.__main__ import main as run_command
from dataeval_flow._activity import recording


def _watch_parent(parent: int) -> None:
    """Terminate this run's process group once the service that started it is gone."""
    while os.getppid() == parent:
        time.sleep(0.5)
    os.killpg(os.getpgrp(), signal.SIGTERM)


def _activity_path(argv: list[str]) -> Path | None:
    """Take ``--activity <path>`` off `argv`: where to record the run, and no flag of the batch command."""
    if "--activity" not in argv:
        return None
    index = argv.index("--activity")
    path = Path(argv[index + 1])
    del argv[index : index + 2]
    return path


def main() -> None:
    """Run the batch command on this process's arguments, recording its activity, and watching the service that
    started it."""
    threading.Thread(target=_watch_parent, args=(os.getppid(),), daemon=True).start()
    path = _activity_path(sys.argv)
    with recording(path) if path is not None else contextlib.nullcontext():
        run_command()


if __name__ == "__main__":  # pragma: no cover
    main()
