"""TC-5-1 — preprocessing & view pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import (
    DataCleaningWorkflowConfig,
    PreprocessorConfig,
    SourceConfig,
    TaskConfig,
    ViewConfig,
    ViewOperation,
    run_tasks,
)
from dataeval_flow.preprocessing import PreprocessingStep
from verification.fixtures import make_synthetic_dataset

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dataeval_flow import PipelineConfig


class TestPreprocessingView:
    def test_preprocessor_config_accepts_transforms(self) -> None:
        cfg = PreprocessorConfig(
            name="pp",
            steps=[PreprocessingStep(step="ToDtype", params={"dtype": "float32", "scale": True})],
        )
        assert len(cfg.steps) == 1

    def test_view_operation_constructs(self) -> None:
        op = ViewOperation(type="Limit", params={"size": 4})
        assert op.type == "Limit"

    def test_view_config_stacks_operations(self) -> None:
        cfg = ViewConfig(
            name="view",
            operations=[
                ViewOperation(type="Limit", params={"size": 4}),
                ViewOperation(type="Shuffle", params={}),
            ],
        )
        assert len(cfg.operations) == 2

    def test_view_limit_reduces_dataset_length(self) -> None:
        from dataeval.data import Limit, View

        ds = make_synthetic_dataset(n=8)
        limited = View(ds, [Limit(size=3)])
        assert len(limited) == 3

    def test_seeded_shuffle_view_pipeline_is_deterministic(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        """The same config and seed select the same samples; another seed selects others."""

        def selected_indices(seed: int) -> list[int]:
            cfg, data_dir = image_folder_pipeline_builder(
                n_per_class=8,
                workflows=[
                    DataCleaningWorkflowConfig(
                        name="clean", type="data-cleaning", outlier_method="zscore", outlier_flags=["pixel"]
                    )
                ],
                tasks=[TaskConfig(name="clean_task", workflow="clean", sources="main")],
            )
            views = [
                ViewConfig(
                    name="shuffled",
                    operations=[
                        ViewOperation(type="Shuffle", params={}),
                        ViewOperation(type="Limit", params={"size": 6}),
                    ],
                )
            ]
            sources = [SourceConfig(name="main", dataset="main_ds", view="shuffled")]
            cfg = cfg.model_copy(update={"seed": seed, "views": views, "sources": sources})
            result = run_tasks(cfg, data_dir=data_dir)[0]
            assert result.success
            return list(result.dataset.resolve_indices())  # type: ignore[union-attr]

        first = selected_indices(11)
        assert len(first) == 6
        assert selected_indices(11) == first
        assert selected_indices(12) != first
