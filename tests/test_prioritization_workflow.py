"""Tests for prioritization workflow — value_range wiring end-to-end."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval_flow._orchestrator import _run_target
from dataeval_flow.workflows import DatasetContext, WorkflowContext
from dataeval_flow.workflows.data_prioritization import (
    DataPrioritizationCleaningConfig,
    DataPrioritizationConfig,
    DataPrioritizationWorkflow,
)

pytestmark = pytest.mark.required


def _prioritization_params(**overrides: object) -> DataPrioritizationConfig:
    """Build DataPrioritizationConfig with defaults for testing.

    Every field on ``DataPrioritizationConfig`` has a default (see
    ``DataPrioritizationConfig.model_fields``), so nothing is strictly required — except
    that ``get_or_compute_stats`` is only reached when ``cleaning`` is configured, so tests
    that need the stats pass to run must set it.
    """
    defaults: dict[str, object] = {
        "cleaning": DataPrioritizationCleaningConfig(outlier_method="adaptive", outlier_flags=["dimension", "pixel"]),
    }
    defaults.update(overrides)
    return DataPrioritizationConfig(**defaults)  # type: ignore[arg-type]


class TestValueRangeReachesPrioritization:
    """The workflow with no metadata policy still gets the dataset's declared range."""

    def test_the_dataset_range_reaches_compute_stats(self, monkeypatch):
        from dataeval_flow import _cache as cache_module

        seen: list[tuple[float, float] | None] = []
        original = cache_module._do_compute_stats

        def _spy(dataset, desired_flags, per_image=True, per_target=True, value_range=None):
            seen.append(value_range)
            return original(dataset, desired_flags, per_image, per_target, value_range)

        monkeypatch.setattr(cache_module, "_do_compute_stats", _spy)

        # Embedding extraction is unrelated to value_range but runs unconditionally before
        # the optional cleaning step that calls get_or_compute_stats — stub it out so the
        # run reaches cleaning without a real extractor/model.
        from dataeval_flow.workflows.data_prioritization import _workflow as prioritization_workflow

        monkeypatch.setattr(
            prioritization_workflow,
            "_get_embeddings_for_context",
            lambda _dc, dataset: np.zeros((len(dataset), 4), dtype=np.float32),
        )

        from tests.test_metadata_injection import _ICDataset

        # DataPrioritizationConfig declares `SourceCount.TWO_OR_MORE`, checked when a task's
        # config loads — this context needs two entries only to match a real run's shape.
        context = WorkflowContext(
            dataset_contexts={
                "default": DatasetContext(
                    name="default",
                    dataset=_ICDataset(),
                    extractor=MagicMock(),
                    value_range=(0.0, 1.0),
                ),
                "extra": DatasetContext(
                    name="extra",
                    dataset=_ICDataset(),
                    extractor=MagicMock(),
                    value_range=(0.0, 1.0),
                ),
            },
        )
        # The mocked extractor fails the run after cleaning; only the stats pass before it matters here.
        _run_target(DataPrioritizationWorkflow(), _prioritization_params(), context)

        assert seen, "no stats pass ran"
        assert all(entry == (0.0, 1.0) for entry in seen), seen
