"""Tests for prioritization workflow — value_range wiring end-to-end."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval_flow.workflow import DatasetContext, WorkflowContext
from dataeval_flow.workflows.prioritization.params import CleaningConfig, DataPrioritizationParameters
from dataeval_flow.workflows.prioritization.workflow import DataPrioritizationWorkflow

pytestmark = pytest.mark.required


def _prioritization_params(**overrides: object) -> DataPrioritizationParameters:
    """Build DataPrioritizationParameters with defaults for testing.

    Every field on ``DataPrioritizationParameters`` has a default (see
    ``DataPrioritizationParameters.model_fields``), so nothing is strictly required — except
    that ``get_or_compute_stats`` is only reached when ``cleaning`` is configured, so tests
    that need the stats pass to run must set it.
    """
    defaults: dict[str, object] = {
        "cleaning": CleaningConfig(outlier_method="adaptive", outlier_flags=["dimension", "pixel"]),
    }
    defaults.update(overrides)
    return DataPrioritizationParameters(**defaults)  # type: ignore[arg-type]


class TestValueRangeReachesPrioritization:
    """The workflow with no metadata policy still gets the dataset's declared range."""

    def test_the_dataset_range_reaches_compute_stats(self, monkeypatch):
        from dataeval_flow import cache as cache_module

        seen: list[tuple[float, float] | None] = []
        original = cache_module._do_compute_stats

        def _spy(dataset, desired_flags, per_image=True, per_target=True, value_range=None):
            seen.append(value_range)
            return original(dataset, desired_flags, per_image, per_target, value_range)

        monkeypatch.setattr(cache_module, "_do_compute_stats", _spy)

        # Embedding extraction is unrelated to value_range but runs unconditionally before
        # the optional cleaning step that calls get_or_compute_stats — stub it out so the
        # run reaches cleaning without a real extractor/model.
        from dataeval_flow.workflows.prioritization import workflow as prioritization_workflow

        monkeypatch.setattr(
            prioritization_workflow,
            "_get_embeddings_for_context",
            lambda _dc, dataset: np.zeros((len(dataset), 4), dtype=np.float32),
        )

        from tests.test_metadata_injection import _ICDataset

        # DataPrioritizationWorkflow requires at least 2 dataset contexts (reference +
        # data to prioritize) — a single context never reaches _run_cleaning because the
        # workflow's own validation guard returns early first.
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
        DataPrioritizationWorkflow().execute(context, _prioritization_params())

        assert seen, "no stats pass ran"
        assert all(entry == (0.0, 1.0) for entry in seen), seen
