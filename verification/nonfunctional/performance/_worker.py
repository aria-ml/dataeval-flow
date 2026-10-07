"""Pipeline builder and subprocess worker for the performance tests (not a test module).

As a script it runs one pipeline over an ImageFolder in a fresh interpreter and prints a JSON line with the
elapsed time and peak resident memory, so peak memory is measured per run rather than across the session.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from dataeval_flow import (
    DataCleaningWorkflowConfig,
    FlattenExtractorConfig,
    ImageFolderDatasetConfig,
    PipelineConfig,
    SourceConfig,
    TaskConfig,
    run_tasks,
)
from dataeval_flow.config import DataAnalysisTaskConfig, DataAnalysisWorkflowConfig

WORKFLOWS = ("data-cleaning", "data-analysis")


def build_config(workflow: str, *, max_processes: int | None = None) -> PipelineConfig:
    """A one-task pipeline over ``<data_dir>/imgs`` running *workflow* with the full statistics set."""
    flags = ["dimension", "pixel", "visual"]
    if workflow == "data-cleaning":
        wf = DataCleaningWorkflowConfig(name="wf", type="data-cleaning", outlier_method="zscore", outlier_flags=flags)
        task: TaskConfig = TaskConfig(name="task", workflow="wf", sources="main", extractor="flat")
    else:
        wf = DataAnalysisWorkflowConfig(name="wf", type="data-analysis", outlier_method="zscore", outlier_flags=flags)  # type: ignore[assignment]
        task = DataAnalysisTaskConfig(name="task", workflow="wf", sources="main", extractor="flat")
    return PipelineConfig(
        max_processes=max_processes,
        datasets=[ImageFolderDatasetConfig(name="ds", path="imgs", infer_labels=True)],
        sources=[SourceConfig(name="main", dataset="ds")],
        extractors=[FlattenExtractorConfig(name="flat", model="flatten", batch_size=256)],
        workflows=[wf],
        tasks=[task],
    )


def _peak_rss_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # kilobytes on Linux


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("workflow", choices=WORKFLOWS)
    parser.add_argument(
        "--warmup", type=Path, help="small ImageFolder root run first, to exclude one-off start-up cost"
    )
    parser.add_argument("--max-processes", type=int)
    args = parser.parse_args()

    cfg = build_config(args.workflow, max_processes=args.max_processes)
    if args.warmup is not None:
        run_tasks(cfg, data_dir=args.warmup)
    baseline_kb = _peak_rss_kb()

    start = time.perf_counter()
    result = run_tasks(cfg, data_dir=args.data_dir)[0]
    seconds = time.perf_counter() - start

    print(
        json.dumps(
            {
                "success": result.success,
                "errors": list(result.errors),
                "seconds": seconds,
                "baseline_rss_kb": baseline_kb,
                "peak_rss_kb": _peak_rss_kb(),
                "result_bytes": len(result.export()),
                "max_processes": result.metadata.resolved_config.get("max_processes"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
