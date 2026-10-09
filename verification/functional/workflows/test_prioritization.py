"""TC-10-2 — the prioritization preset: pools ranked against a reference, and the top of each ranking kept."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow.workflows.prioritization import PrioritizationWorkflowConfig
from verification.functional.workflows._synthetic import Concat, Images, pipeline, run_preset

pytestmark = pytest.mark.required

PRIORITIZATION: dict[str, Any] = {"type": "prioritization", "prioritization": {"method": "knn", "k": 3}}


def sources(**more: Any) -> dict[str, Any]:
    """A reference, and a pool whose first 20 images resemble it and whose last 20 have a colour cast it lacks."""
    return {
        "reference": Images(40, seed=0),
        "pool": Concat(Images(20, seed=5), Images(20, seed=6, tint=True)),
        **more,
    }


def ranking(result: Any, pool: str = "pool") -> list[int]:
    return [int(i) for i in result.steps["prioritization"].elements[pool].output.indices]


def selected_ids(result: Any, pool: str = "pool") -> list[int]:
    kept = result.steps["selected"].elements[pool].output
    return [kept[i][2]["id"] for i in range(len(kept))]


class TestPrioritizationPreset:
    def test_hard_first_ranks_the_items_least_like_the_reference_first(self) -> None:
        result = run_preset(PRIORITIZATION, sources(), extractor=True)

        assert result.success, result.errors
        assert result.type == "prioritization"
        assert list(result.steps) == ["prioritization", "selected"]
        order = ranking(result)
        assert sorted(order) == list(range(40))  # every pool item is ranked once
        assert set(order[:20]) == set(range(20, 40))  # the colour-cast half comes first
        assert result.findings == []  # the preset has no checks
        assert "Highest priority" in result.report()

    def test_easy_first_reverses_the_order(self) -> None:
        entry = {"type": "prioritization", "prioritization": {"method": "knn", "k": 3, "order": "easy_first"}}

        result = run_preset(entry, sources(), extractor=True)

        assert set(ranking(result)[:20]) == set(range(20))

    def test_selected_keeps_every_item_in_ranked_order_unless_told_otherwise(self) -> None:
        result = run_preset(PRIORITIZATION, sources(), extractor=True)

        assert selected_ids(result) == ranking(result)

    def test_select_n_keeps_that_many_of_the_top_items(self) -> None:
        result = run_preset({**PRIORITIZATION, "select": {"n": 5}}, sources(), extractor=True)

        assert selected_ids(result) == ranking(result)[:5]

    def test_select_n_larger_than_the_pool_keeps_the_whole_pool(self) -> None:
        result = run_preset({**PRIORITIZATION, "select": {"n": 100}}, sources(), extractor=True)

        assert len(selected_ids(result)) == 40

    def test_select_fraction_keeps_its_share_rounded_up(self) -> None:
        result = run_preset({**PRIORITIZATION, "select": {"fraction": 0.21}}, sources(), extractor=True)

        assert selected_ids(result) == ranking(result)[:9]  # 0.21 of 40 is 8.4

    def test_n_and_fraction_together_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="takes `n:` or `fraction:`, not both"):
            PrioritizationWorkflowConfig.model_validate({"select": {"n": 5, "fraction": 0.5}})

    def test_each_pool_is_ranked_on_its_own_against_the_one_reference(self) -> None:
        data = sources(second=Images(12, seed=7, tint=True))

        result = run_preset({**PRIORITIZATION, "select": {"n": 5}}, data, extractor=True)

        assert result.steps["prioritization"].inputs == ["pools", "reference"]
        assert list(result.steps["prioritization"].elements) == ["pool", "second"]
        assert len(selected_ids(result, "pool")) == 5
        assert len(selected_ids(result, "second")) == 5
        assert sorted(ranking(result, "second")) == list(range(12))

    @pytest.mark.parametrize(
        "ranking_settings",
        [
            {"policy": "stratified", "num_bins": 4},
            {"policy": "class_balanced"},
            {"method": "kmeans_distance", "c": 4},
            {"method": "hdbscan_distance"},
        ],
        ids=["stratified", "class-balanced", "kmeans-distance", "hdbscan-distance"],
    )
    def test_other_methods_and_policies_rank_the_whole_pool(self, ranking_settings: dict[str, Any]) -> None:
        entry = {"type": "prioritization", "prioritization": {"method": "knn", "k": 3, **ranking_settings}}

        result = run_preset(entry, sources(), extractor=True)

        assert result.success, result.errors
        assert sorted(ranking(result)) == list(range(40))

    @pytest.mark.parametrize(
        "bad",
        [{"policy": "bogus"}, {"method": "bogus"}, {"order": "sideways"}],
        ids=["policy", "method", "order"],
    )
    def test_an_unsupported_policy_method_or_order_is_refused(self, bad: dict[str, Any]) -> None:
        with pytest.raises(ValidationError, match="Input should be"):
            PrioritizationWorkflowConfig.model_validate({"prioritization": bad})

    def test_a_task_needs_a_reference_and_a_pool_and_an_extractor(self) -> None:
        with pytest.raises(ValidationError, match="takes two or more sources, but the task names 1"):
            run_preset(PRIORITIZATION, {"reference": Images(10)}, extractor=True)
        with pytest.raises(ValidationError, match="extractor"):
            run_preset(PRIORITIZATION, sources())

    def test_cleaning_each_pool_first_ranks_only_clean_data(self) -> None:
        shape = {"shape": (3, 16, 16), "value_range": (100, 150)}
        workflows = [
            {"name": "cleaning", "type": "quality", "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}},
            {"name": "prio", **PRIORITIZATION},
            {
                "name": "clean-then-rank",
                "inputs": ["reference", {"name": "pools", "list": True}],
                "steps": [
                    {"name": "pool-clean", "workflow": "cleaning", "input": "pools"},
                    {"name": "rank", "workflow": "prio", "input": ["reference", "pool-clean.clean"]},
                ],
            },
        ]
        config = pipeline(
            workflows=workflows,
            tasks=[{"name": "t", "workflow": "clean-then-rank", "sources": ["reference", "pool"], "extractor": "flat"}],
            datasets={"reference": Images(40, seed=0, **shape), "pool": Images(40, seed=5, planted=True, **shape)},
            extractor=True,
            extra={"seed": 0},
        )

        result = run_tasks(config)["t"]

        assert result.success, result.errors
        ranked = result.steps["rank/selected"].elements["pool"].output
        ids = {ranked[i][2]["id"] for i in range(len(ranked))}
        assert ids == set(range(40)) - {5, 7}  # the duplicate of item 0 and the white image never reach the ranking
