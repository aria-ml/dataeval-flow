"""TC-21-1 — pipeline seeding makes stochastic runs reproducible [CR-7-S-1]."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from dataeval_flow import PipelineConfig, run_tasks
from dataeval_flow.config import DataCleaningTaskConfig, DataCleaningWorkflowConfig

pytestmark = pytest.mark.required

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _seeded_cleaning_pipeline(
    builder: Callable[..., tuple[PipelineConfig, Path]],
    *,
    seed: int | None,
) -> tuple[PipelineConfig, Path]:
    """A cleaning pipeline whose cluster-based detection is stochastic."""
    cfg, data_dir = builder(
        workflows=[
            DataCleaningWorkflowConfig(
                name="clean",
                type="data-cleaning",
                outlier_method="zscore",
                outlier_flags=["dimension", "pixel"],
                # Cluster-based detection is the stochastic part.
                outlier_cluster_threshold=3.0,
                outlier_cluster_algorithm="kmeans",
                outlier_n_clusters=2,
            ),
        ],
        tasks=[
            DataCleaningTaskConfig(
                name="clean_task",
                workflow="clean",
                sources="main",
                extractor="flat",
            ),
        ],
    )
    return cfg.model_copy(update={"seed": seed}), data_dir


@pytest.mark.test_case("21-1")
class TestSeedConfiguration:
    def test_seed_defaults_to_none(self) -> None:
        """An unseeded pipeline leaves randomness alone."""
        assert PipelineConfig().seed is None
        assert PipelineConfig().deterministic is False

    def test_seed_is_applied_to_dataeval(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        """Running a seeded pipeline pins DataEval's seed configuration."""
        from dataeval.config import get_seed, set_seed

        set_seed(None)
        assert get_seed() is None

        cfg, data_dir = _seeded_cleaning_pipeline(image_folder_pipeline_builder, seed=1234)
        result = run_tasks(cfg, data_dir=data_dir)[0]

        assert result.success
        assert get_seed() == 1234

    def test_seed_recorded_in_result_envelope(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        """The seed is part of the provenance, so the envelope alone can repeat the run."""
        cfg, data_dir = _seeded_cleaning_pipeline(image_folder_pipeline_builder, seed=7)
        result = run_tasks(cfg, data_dir=data_dir)[0]

        resolved = result.metadata.resolved_config
        assert resolved["seed"] == 7
        assert resolved["deterministic"] is False

    def test_unseeded_run_records_no_seed(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        """An unseeded run must not claim a seed it never applied."""
        cfg, data_dir = _seeded_cleaning_pipeline(image_folder_pipeline_builder, seed=None)
        result = run_tasks(cfg, data_dir=data_dir)[0]

        assert "seed" not in result.metadata.resolved_config

    def test_same_seed_reproduces_stochastic_output(
        self,
        image_folder_pipeline_builder: Callable[..., tuple[PipelineConfig, Path]],
    ) -> None:
        """Two seeded runs of a clustering workflow agree."""
        first_cfg, first_dir = _seeded_cleaning_pipeline(image_folder_pipeline_builder, seed=99)
        second_cfg, second_dir = _seeded_cleaning_pipeline(image_folder_pipeline_builder, seed=99)

        first = run_tasks(first_cfg, data_dir=first_dir)[0]
        second = run_tasks(second_cfg, data_dir=second_dir)[0]

        assert first.success
        assert second.success
        assert first.data.raw.model_dump(mode="json") == second.data.raw.model_dump(mode="json")
