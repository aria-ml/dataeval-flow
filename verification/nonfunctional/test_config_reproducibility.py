"""TC-11-1 (NFR-4) — configuration reproducibility.

Demonstrates that the splitting workflow is deterministic for a given
``PipelineConfig`` (same config → identical ``to_dict()`` output, ignoring
non-deterministic envelope fields like ``timestamp`` and ``execution_time_s``),
and that varying the configuration changes the output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import (
    ImageFolderDatasetConfig,
    PipelineConfig,
    SourceConfig,
    run_tasks,
)
from dataeval_flow.config.schemas import DataSplittingTaskConfig, DataSplittingWorkflowConfig
from verification.fixtures import write_image_folder


def _build_split_cfg(data_root: Path, test_frac: float) -> PipelineConfig:
    write_image_folder(data_root / "imgs", n_per_class=8, n_classes=2)
    return PipelineConfig(
        datasets=[
            ImageFolderDatasetConfig(
                name="main_ds",
                format="image_folder",
                path="imgs",
                infer_labels=True,
            ),
        ],
        sources=[SourceConfig(name="main", dataset="main_ds")],
        workflows=[
            DataSplittingWorkflowConfig(
                name="split_main",
                type="data-splitting",
                test_frac=test_frac,
                val_frac=0.25,
                num_folds=1,
                stratify=False,
            ),
        ],
        tasks=[
            DataSplittingTaskConfig(
                name="split_task",
                workflow="split_main",
                sources="main",
            ),
        ],
    )


def _strip_volatile_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove non-deterministic envelope fields (timestamp + duration)."""
    meta = dict(payload.get("metadata", {}))
    for key in ("timestamp", "execution_time_s"):
        meta.pop(key, None)
    return {**payload, "metadata": meta}


def _run(data_root: Path, test_frac: float) -> dict[str, Any]:
    """Run splitting and return its serialized result.

    The fold/test index determinism comes from ``KFold(shuffle=False)`` inherent
    to the splitter configuration. The pre-split balance/diversity MI analyses,
    however, use DataEval's MI estimators which sample from the global RNG, so a
    fixed seed is required for full ``to_dict()`` equality across runs.
    """
    from dataeval.config import set_seed

    set_seed(42)
    result = run_tasks(_build_split_cfg(data_root, test_frac=test_frac), data_dir=data_root)[0]
    return _strip_volatile_metadata(result.to_dict())


@pytest.mark.test_case("11-1")
class TestConfigReproducibility:
    def test_same_config_produces_identical_output(self, tmp_path: Path) -> None:
        out_a = _run(tmp_path / "a", test_frac=0.25)
        out_b = _run(tmp_path / "b", test_frac=0.25)
        assert out_a == out_b

    def test_different_config_produces_different_output(self, tmp_path: Path) -> None:
        out_a = _run(tmp_path / "a", test_frac=0.25)
        out_b = _run(tmp_path / "b", test_frac=0.5)
        assert out_a != out_b
