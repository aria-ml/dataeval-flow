"""Recorded `sweep.json` from the legacy parameter-sweep workflow, before its removal.

Commit c93bf6a ran it once, writing each combination's outlier and near-duplicate counts. A data-cleaning task's
matrix must give the same counts (task-matrix spec §11.3). The legacy workflow is gone, so it can't run again.
"""

if __name__ == "__main__":
    raise SystemExit(
        "generate_sweep recorded from the legacy parameter-sweep workflow, which has been removed: a data-cleaning "
        "task's matrix is tested against the golden it wrote, tests/golden/sweep.json."
    )
