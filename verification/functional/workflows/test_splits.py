"""TC-10-1 — the splits preset: train, validation and test sets, or k folds, and their stratification."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow.workflows.splits import SplitsConfig
from verification.functional.workflows._synthetic import Images, by_title, pipeline, run_preset

pytestmark = pytest.mark.required

SPLITS: dict[str, Any] = {"type": "splits"}


def scenes() -> Images:
    """100 items, 50 cat, 30 dog, 20 bird, in ten scenes of ten: every scene holds every class."""
    return Images(100, extra={"scene": lambda index: f"scene-{index // 10}"})


def indices(result: Any) -> dict[str, Any]:
    return result.steps["split"].details["indices"]


class TestSplitsPreset:
    def test_the_parts_are_disjoint_and_together_hold_every_item(self) -> None:
        result = run_preset(SPLITS, scenes())

        assert result.success, result.errors
        assert result.type == "splits"
        parts = indices(result)
        assert set(parts) == {"train", "val", "test"}
        everything = [i for part in parts.values() for i in part]
        assert sorted(everything) == list(range(100))
        assert len(parts["test"]) == 20  # test_frac 0.2
        assert 0 < len(parts["val"]) < len(parts["train"])  # val_frac unset holds out a tenth of the rest
        assert result.report().strip()

    def test_the_same_settings_give_the_same_split_and_other_settings_give_another(self) -> None:
        first = indices(run_preset(SPLITS, scenes()))
        again = indices(run_preset(SPLITS, scenes()))
        more_test = indices(run_preset({**SPLITS, "test_frac": 0.25}, scenes()))
        unstratified = indices(run_preset({**SPLITS, "stratify": False}, scenes()))

        assert again == first
        assert len(more_test["test"]) == 25
        assert more_test != first
        assert unstratified != first

    def test_each_part_s_class_shares_are_judged_against_the_whole(self) -> None:
        stratified = run_preset({**SPLITS, "test_frac": 0.25, "val_frac": 0.25}, scenes())
        unstratified = run_preset({**SPLITS, "test_frac": 0.25, "val_frac": 0.25, "stratify": False}, scenes())

        assert by_title(stratified)["Class Stratification"].brief == "max deviation 3.3pp (class 'cat' in split.val)"
        assert by_title(stratified)["Class Stratification"].severity == "info"  # over 2.0 points, under 10.0
        assert by_title(unstratified)["Class Stratification"].brief == "max deviation 16.7pp (class 'cat' in split.val)"
        assert by_title(unstratified)["Class Stratification"].severity == "warning"

    def test_thresholds_in_checks_decide_whether_a_deviation_warns(self) -> None:
        strict = {"class-stratification": {"info": 1.0, "warning": 3.0}}

        result = run_preset({**SPLITS, "test_frac": 0.25, "val_frac": 0.25, "checks": strict}, scenes())

        assert by_title(result)["Class Stratification"].severity == "warning"  # 3.3 points, over 3.0

    def test_folds_make_that_many_train_and_validation_sets_and_one_shared_test(self) -> None:
        result = run_preset({**SPLITS, "folds": 4}, scenes())

        assert result.success, result.errors
        parts = indices(result)
        assert len(parts["train"]) == len(parts["val"]) == 4
        assert len(parts["test"]) == 20
        validation = [set(fold) for fold in parts["val"].values()]
        assert set().union(*validation) == set(range(100)) - set(parts["test"])  # the folds tile what is left
        assert sum(len(fold) for fold in validation) == 80  # and never overlap
        for key, fold in parts["train"].items():
            assert not set(fold) & set(parts["val"][key])
            assert not set(fold) & set(parts["test"])
        steps = [f.step for f in result.findings if f.title == "Class Stratification"]
        assert steps == [f"class-stratification[{k}]" for k in range(4)]  # each fold is judged

    def test_split_on_keeps_each_value_of_a_factor_in_one_part(self) -> None:
        result = run_preset({**SPLITS, "split_on": ["scene"]}, scenes())

        assert result.success, result.errors
        parts = indices(result)
        scene_sets = {name: {i // 10 for i in part} for name, part in parts.items()}
        assert not scene_sets["train"] & scene_sets["val"]
        assert not scene_sets["train"] & scene_sets["test"]
        assert not scene_sets["val"] & scene_sets["test"]
        assert sorted(i for part in parts.values() for i in part) == list(range(100))

    def test_rebalancing_makes_the_trained_on_part_more_even_than_the_split_one(self) -> None:
        result = run_preset({**SPLITS, "rebalance": "interclass"}, scenes())

        assert result.success, result.errors
        split_counts = result.steps["label-health-train"].output.data()["label_counts_per_class"]
        rebalanced = result.steps["label-health-rebalanced"].output.data()["label_counts_per_class"]
        assert max(split_counts.values()) / min(split_counts.values()) > 2.0  # cat 37, dog 22, bird 15
        assert max(rebalanced.values()) / min(rebalanced.values()) < 1.2

    def test_a_custom_workflow_reads_the_parts_as_the_steps_train_val_and_test(self) -> None:
        mine = {
            "name": "mine",
            "inputs": ["data"],
            "steps": [
                {"name": "parts", "workflow": "splitter", "input": "data"},
                {"name": "train-health", "evaluator": "labels", "input": "parts.train"},
            ],
        }
        config = pipeline(
            workflows=[{"name": "splitter", **SPLITS}, mine],
            evaluators=[{"name": "labels", "type": "label-health"}],
            tasks=[{"name": "t", "workflow": "mine", "sources": ["data"]}],
            datasets={"data": scenes()},
            extra={"seed": 0},
        )

        result = run_tasks(config)["t"]

        assert result.success, result.errors
        assert result.steps["train-health"].output.data()["item_count"] == len(
            result.steps["parts/split"].output["train"]
        )

    def test_a_split_that_holds_nothing_out_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="`split` holds nothing out"):
            run_preset({**SPLITS, "test_frac": 0.0, "val_frac": 0.0}, scenes())

    def test_val_frac_with_folds_and_no_folds_are_refused(self) -> None:
        with pytest.raises(ValidationError, match="applies with `folds: 1` only"):
            SplitsConfig.model_validate({"folds": 3, "val_frac": 0.2})
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            SplitsConfig.model_validate({"folds": 0})

    def test_too_few_items_to_stratify_fail_the_split_step(self) -> None:
        result = run_preset({**SPLITS, "folds": 3}, Images(6))

        assert not result.success
        assert result.steps["split"].status == "failed"
        assert any("stratify" in error for error in result.steps["split"].errors)
