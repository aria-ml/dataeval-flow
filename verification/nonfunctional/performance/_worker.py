"""Pipeline builder and subprocess worker for the performance tests (not a test module).

As a script it runs one pipeline over an ImageFolder in a fresh interpreter and prints a JSON line with the
elapsed time and peak resident memory, so peak memory is measured per run rather than across the session.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

from dataeval_flow import run_tasks
from dataeval_flow.config import PipelineConfig

_FLAGS = ["dimension", "pixel", "visual"]

# Two reference workloads, both the `quality` preset: image statistics and outlier detection over them, and the
# same with cluster-based outlier detection on top, which also embeds every image and clusters the embeddings.
WORKLOADS: dict[str, dict] = {
    "quality": {"outliers": {"flags": _FLAGS, "outlier_threshold": "zscore"}},
    "quality-clusters": {
        "outliers": {
            "flags": _FLAGS,
            "outlier_threshold": "zscore",
            "cluster_threshold": 3.0,
            "cluster_algorithm": "kmeans",
            "n_clusters": 4,
        }
    },
}


def build_config(workload: str, *, max_processes: int | None = None) -> PipelineConfig:
    """A one-task pipeline over ``<data_dir>/imgs`` running the *workload*."""
    return PipelineConfig.model_validate(
        {
            "max_processes": max_processes,
            "datasets": [{"name": "ds", "format": "image_folder", "path": "imgs", "infer_labels": True}],
            "sources": [{"name": "main", "dataset": "ds"}],
            "extractors": [{"name": "flat", "model": "flatten", "batch_size": 256}],
            "workflows": [{"name": "wf", "type": "quality", **WORKLOADS[workload]}],
            "tasks": [{"name": "task", "workflow": "wf", "sources": "main", "extractor": "flat"}],
        }
    )


def _status_kb(field: str) -> int | None:
    """A kilobyte figure from ``/proc/self/status`` (``VmRSS``: resident now, ``VmHWM``: resident peak), or ``None``.

    ``getrusage`` is not used: the peak of a child process starts at its parent's peak, so under a large test
    process it would hide the worker's own growth.
    """
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(f"{field}:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def _reset_peak() -> None:
    """Start the resident-memory peak again from the current size (Linux 4.0 and later)."""
    with contextlib.suppress(OSError):
        Path("/proc/self/clear_refs").write_text("5")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("workload", choices=sorted(WORKLOADS))
    parser.add_argument(
        "--warmup", type=Path, help="small ImageFolder root run first, to exclude one-off start-up cost"
    )
    parser.add_argument("--max-processes", type=int)
    args = parser.parse_args()

    cfg = build_config(args.workload, max_processes=args.max_processes)
    if args.warmup is not None:
        run_tasks(cfg, data_dir=args.warmup)
    _reset_peak()
    baseline_kb = _status_kb("VmRSS")

    start = time.perf_counter()
    result = run_tasks(cfg, data_dir=args.data_dir)["task"]
    seconds = time.perf_counter() - start

    print(
        json.dumps(
            {
                "success": result.success,
                "errors": list(result.errors),
                "seconds": seconds,
                "baseline_rss_kb": baseline_kb,
                "peak_rss_kb": _status_kb("VmHWM"),
                "max_processes": result.metadata.resolved_config.get("max_processes"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
