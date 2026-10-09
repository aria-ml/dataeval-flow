"""TC-35-1 — memory, warm-cache reuse, process-count timing, and reference-pipeline budgets.

Measured runs execute in a fresh interpreter (``_worker.py``) so peak memory belongs to that run alone.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import run_tasks
from verification.nonfunctional.performance._worker import build_config

pytestmark = [pytest.mark.required, pytest.mark.performance]

_WORKER = Path(__file__).with_name("_worker.py")

# Time budgets in seconds for 10,000 32x32 images. The reference measurements on a development machine were
# 7.7 s (quality) and 10.6 s (quality-clusters); the budgets are about eight times those, which is the margin the
# earlier releases kept over a shared CI runner. Regression guards, not performance promises.
BUDGET_S = {"quality": 60.0, "quality-clusters": 90.0}

# Reference peak-memory growth per 32x32 image for the `quality` workload (about 8 KB on the development machine).
# The gate allows three times that.
STORED_KB_PER_IMAGE = 8
MEMORY_GROWTH_FACTOR = 3


def _run_worker(
    data_root: Path, workload: str, *, warmup: Path | None = None, max_processes: int | None = None
) -> dict[str, Any]:
    cmd = [sys.executable, "-W", "ignore", str(_WORKER), str(data_root), workload]
    if warmup is not None:
        cmd += ["--warmup", str(warmup)]
    if max_processes is not None:
        cmd += ["--max-processes", str(max_processes)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("DATAEVAL_")}
    proc = subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, check=False, env=env
    )
    assert proc.returncode == 0, proc.stderr
    outcome = json.loads(proc.stdout.strip().splitlines()[-1])
    assert outcome["success"], outcome["errors"]
    return outcome


class TestScale:
    def test_peak_memory_follows_stored_results_not_dataset_size(
        self, image_root: Callable[[int, int], Path], record_property: Callable[[str, object], None]
    ) -> None:
        warmup = image_root(100, 32)

        # Growth over the post-warm-up baseline at two dataset sizes; the slope is the per-image cost.
        small = _run_worker(image_root(2_000, 32), "quality", warmup=warmup)
        if small["peak_rss_kb"] is None:
            pytest.skip("peak resident memory is read from /proc/self/status, which this platform does not have")
        large = _run_worker(image_root(8_000, 32), "quality", warmup=warmup)
        growth_small = small["peak_rss_kb"] - small["baseline_rss_kb"]
        growth_large = large["peak_rss_kb"] - large["baseline_rss_kb"]
        assert growth_large > growth_small > 0, "the measurement saw no growth, so it proves nothing"
        kb_per_image = (growth_large - growth_small) / (8_000 - 2_000)
        record_property("peak_growth_kb_per_32px_image", round(kb_per_image, 2))
        record_property("peak_growth_kb_2000_images", growth_small)
        record_property("peak_growth_kb_8000_images", growth_large)
        assert kb_per_image <= MEMORY_GROWTH_FACTOR * STORED_KB_PER_IMAGE

        # Datasets are read lazily: peak growth for larger images stays below the pixels themselves.
        n_large_images = 600
        big = _run_worker(image_root(n_large_images, 224), "quality", warmup=warmup)
        raw_pixel_kb = n_large_images * 224 * 224 * 3 / 1024
        growth_big = big["peak_rss_kb"] - big["baseline_rss_kb"]
        record_property("peak_growth_kb_600_images_224px", growth_big)
        assert 0 < growth_big < raw_pixel_kb

    def test_warm_cache_rerun_reuses_artifacts_and_is_faster(
        self,
        image_root: Callable[[int, int], Path],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        record_property: Callable[[str, object], None],
    ) -> None:
        data_root = image_root(1_000, 32)
        cache_dir = tmp_path / "cache"
        cfg = build_config("quality")
        caplog.set_level(logging.INFO, logger="dataeval_flow._cache")

        cold = run_tasks(cfg, data_dir=data_root, cache_dir=cache_dir)["task"]
        cold_logs = [r.getMessage() for r in caplog.records]
        cached_files = sorted(p.name for p in cache_dir.rglob("*") if p.is_file())
        caplog.clear()
        from dataeval_flow._cache import DatasetCache

        DatasetCache.clear_instances()  # a new process would start with empty memory caches too
        warm = run_tasks(cfg, data_dir=data_root, cache_dir=cache_dir)["task"]
        warm_logs = [r.getMessage() for r in caplog.records]

        assert cold.success
        assert warm.success
        assert cached_files
        assert not any(m.startswith("Cache hit") for m in cold_logs)
        assert any(m.startswith("Cache hit: stats") for m in warm_logs)
        assert any(m.startswith("Cache hit: metadata") for m in warm_logs)
        assert not any("Cache save" in m for m in warm_logs)
        cold_s, warm_s = cold.metadata.execution_time_s, warm.metadata.execution_time_s
        record_property("cold_s", cold_s)
        record_property("warm_s", warm_s)
        assert cold_s is not None
        assert warm_s is not None
        assert warm_s < cold_s
        # Cached statistics give the same findings.
        assert warm.findings == cold.findings

    def test_quality_time_at_one_and_four_processes_is_recorded(
        self, image_root: Callable[[int, int], Path], record_property: Callable[[str, object], None]
    ) -> None:
        """Reported, not gated: whether more processes help depends on the platform."""
        data_root = image_root(200, 224)
        warmup = image_root(100, 32)

        runs = {n: _run_worker(data_root, "quality", warmup=warmup, max_processes=n) for n in (1, 4)}

        for n, run in runs.items():
            record_property(f"quality_s_{n}_processes", round(run["seconds"], 3))
            assert run["max_processes"] == n
            print(f"quality, 200 images of 224x224, {n} process(es): {run['seconds']:.2f} s")
        record_property("ratio_4_to_1_processes", round(runs[4]["seconds"] / runs[1]["seconds"], 3))

    @pytest.mark.parametrize("workload", ["quality", "quality-clusters"])
    def test_reference_pipeline_completes_within_budget(
        self,
        image_root: Callable[[int, int], Path],
        record_property: Callable[[str, object], None],
        workload: str,
    ) -> None:
        run = _run_worker(image_root(10_000, 32), workload)
        record_property(f"{workload}_s", round(run["seconds"], 3))
        assert run["seconds"] <= BUDGET_S[workload], (
            f"{workload} took {run['seconds']:.1f} s (budget {BUDGET_S[workload]} s)"
        )
