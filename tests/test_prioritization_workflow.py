"""Tests for prioritization workflow — value_range wiring end-to-end, and the findings it reports."""

from collections.abc import Sized
from unittest.mock import MagicMock

import numpy as np
import pytest

from dataeval_flow._blocks import Fields, ItemRef
from dataeval_flow._orchestrator import _run_target
from dataeval_flow.config import ViewOperation
from dataeval_flow.workflows import DatasetContext, WorkflowContext
from dataeval_flow.workflows.data_prioritization import (
    DataPrioritizationCleaningConfig,
    DataPrioritizationConfig,
    DataPrioritizationWorkflow,
)
from dataeval_flow.workflows.data_prioritization._outputs import (
    CleaningSummaryDict,
    DataPrioritizationRawOutput,
    PerDatasetPrioritizationDict,
)
from dataeval_flow.workflows.data_prioritization._report import build_findings
from tests.finding_blocks import blocks_of, column, rendered, sections, tables

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


class TestSources:
    def test_the_result_carries_the_views_it_ranked(self, monkeypatch):
        """Each item the report pictures is read from the view it was ranked in, not a view built anew."""
        from dataeval_flow.workflows.data_prioritization import _workflow as prioritization_workflow
        from tests.test_metadata_injection import _ICDataset

        embedded: dict[str, Sized] = {}

        def _embeddings(dc: DatasetContext, dataset: Sized) -> np.ndarray:
            embedded[dc.name] = dataset
            return np.random.default_rng(len(embedded)).random((len(dataset), 4)).astype(np.float32)

        monkeypatch.setattr(prioritization_workflow, "_get_embeddings_for_context", _embeddings)
        monkeypatch.setattr(prioritization_workflow, "build_extractor", lambda *_args, **_kwargs: None)
        head = [ViewOperation(type="Limit", params={"size": 30})]
        context = WorkflowContext(
            dataset_contexts={
                name: DatasetContext(name=name, dataset=_ICDataset(), extractor=MagicMock(), view_operations=head)
                for name in ("reference", "pool")
            }
        )
        result = DataPrioritizationWorkflow().run(DataPrioritizationConfig(method="knn", k=2), context)

        assert result.sources is not None
        assert result.sources.keys() == {"reference", "pool"}
        assert all(result.sources[name] is embedded[name] for name in ("reference", "pool"))


def _raw(*, cleaning: bool = True) -> DataPrioritizationRawOutput:
    """A two-source run: one source ranked with scores, one without."""
    return DataPrioritizationRawOutput(
        dataset_size=1500,
        reference_size=1000,
        method="knn",
        order="hard_first",
        policy="difficulty",
        cleaning_summary=(
            CleaningSummaryDict(total_combined=1500, outliers_flagged=37, duplicates_flagged=12, total_removed=49)
            if cleaning
            else None
        ),
        prioritizations=[
            PerDatasetPrioritizationDict(
                source_name="pool",
                original_size=500,
                cleaned_size=468,
                prioritized_indices=list(range(467, -1, -1)),
                scores=[0.9 - i * 0.001 for i in range(468)],
            ),
            PerDatasetPrioritizationDict(
                source_name="batch", original_size=200, cleaned_size=200, prioritized_indices=[3, 1, 2], scores=None
            ),
        ],
    )


class TestFindings:
    """Pruning and each source's prioritization, as report blocks."""

    def test_one_pruning_finding_then_one_per_source(self):
        titles = [f.title for f in build_findings(_raw(), DataPrioritizationConfig())]
        assert titles == ["Pruning", "Prioritization: pool", "Prioritization: batch"]

    def test_no_pruning_finding_without_cleaning(self):
        titles = [f.title for f in build_findings(_raw(cleaning=False), DataPrioritizationConfig())]
        assert titles == ["Prioritization: pool", "Prioritization: batch"]

    def test_pruning_counts_are_labelled_fields_in_order(self):
        pruning = build_findings(_raw(), DataPrioritizationConfig())[0]
        assert pruning.brief == "49 items (3.3%)"
        assert pruning.description == "Pruning removed 49/1500 items (3.3%): 37 outliers, 12 duplicates"
        (block,) = blocks_of(pruning, Fields)
        assert block.items == [
            ("Total combined", 1500),
            ("Outliers flagged", 37),
            ("Duplicates flagged", 12),
            ("Total removed", 49),
            ("Removed", "3.3%"),
        ]

    def test_prioritization_run_is_labelled_fields_in_order(self):
        pool = build_findings(_raw(), DataPrioritizationConfig())[1]
        assert pool.severity == "info"
        assert pool.brief == "468 items"
        assert pool.description == "pool: 468 items prioritized via knn (hard_first, difficulty)"
        (block,) = blocks_of(pool, Fields)
        assert block.items == [
            ("Source", "pool"),
            ("Original size", 500),
            ("Cleaned size", 468),
            ("Method", "knn"),
            ("Order", "hard_first"),
            ("Policy", "difficulty"),
        ]

    def test_the_ranking_s_first_and_last_25_are_listed_with_their_ranks_and_scores(self):
        pool = build_findings(_raw(), DataPrioritizationConfig())[1]
        assert [section.title for section in sections(pool)] == ["Highest priority", "Lowest priority"]
        highest, lowest = tables(pool)
        assert [c.header for c in highest.columns] == ["Rank", "", "Item", "Score"]
        assert column(highest, "rank") == list(range(1, 26))
        assert column(highest, "item") == list(range(467, 442, -1))
        assert column(highest, "image")[:2] == [ItemRef(source="pool", index=467), ItemRef(source="pool", index=466)]
        assert column(highest, "score")[:2] == [0.9, 0.899]
        assert column(lowest, "rank") == list(range(444, 469))
        assert column(lowest, "item") == list(range(24, -1, -1))
        assert highest.preview == lowest.preview == 10

    def test_a_short_ranking_is_listed_once_without_scores_where_the_method_gives_none(self):
        batch = build_findings(_raw(), DataPrioritizationConfig())[2]
        (table,) = tables(batch)
        assert [section.title for section in sections(batch)] == ["Highest priority"]
        assert [c.header for c in table.columns] == ["Rank", "", "Item"]
        assert [(row["rank"], row["item"]) for row in table.rows] == [(1, 3), (2, 1), (3, 2)]

    def test_a_ranking_of_30_lists_its_last_5_as_its_lowest(self):
        raw = _raw()
        raw.prioritizations[1]["prioritized_indices"] = list(range(30))
        (_, lowest) = tables(build_findings(raw, DataPrioritizationConfig())[2])
        assert column(lowest, "rank") == [26, 27, 28, 29, 30]

    def test_pruning_renders_as_aligned_fields(self):
        pruning = build_findings(_raw(), DataPrioritizationConfig())[0]
        assert rendered(pruning).splitlines() == [
            "=" * 80,
            "  PRUNING" + "49 items (3.3%)".rjust(71),
            "=" * 80,
            "  Pruning removed 49/1500 items (3.3%): 37 outliers, 12 duplicates",
            "",
            "  Total combined:     1500",
            "  Outliers flagged:   37",
            "  Duplicates flagged: 12",
            "  Total removed:      49",
            "  Removed:            3.3%",
        ]
