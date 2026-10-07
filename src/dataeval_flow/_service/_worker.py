"""A queued run's process: the batch command, which ends its process group should the service that started it go."""

import os
import signal
import threading
import time

from dataeval_flow.__main__ import main as run_command


def _watch_parent(parent: int) -> None:
    """Terminate this run's process group once the service that started it is gone."""
    while os.getppid() == parent:
        time.sleep(0.5)
    os.killpg(os.getpgrp(), signal.SIGTERM)


def main() -> None:
    """Run the batch command on this process's arguments, watching the service that started it."""
    threading.Thread(target=_watch_parent, args=(os.getppid(),), daemon=True).start()
    run_command()


if __name__ == "__main__":  # pragma: no cover
    main()
