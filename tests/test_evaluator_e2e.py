"""Evaluator tasks end to end: config in, EvaluatorResult out, alongside workflows and sharing their cache."""

from unittest.mock import patch

import pytest

import dataeval_flow.cache as cache_module
from dataeval_flow.config import (
    DataCleaningWorkflowConfig,
    DuplicatesEvaluatorConfig,
    EvaluatorTaskConfig,
    OutliersEvaluatorConfig,
    TaskConfig,
)
from dataeval_flow.evaluator.result import EvaluatorResult
from dataeval_flow.workflow import WorkflowResult, run_task, run_tasks
from tests.evaluator_toys import ToyImages, exact_groups, toy_pipeline

_CLEAN = DataCleaningWorkflowConfig(
    name="clean", outlier_method="zscore", outlier_flags=["dimension", "pixel", "visual"]
)


def _dupes_task(sources: str | list[str] = "src") -> EvaluatorTaskConfig:
    return EvaluatorTaskConfig(name="dupes_task", workflow="dupes", sources=sources)


class TestRunTask:
    def test_an_evaluator_task_runs_end_to_end(self):
        task = _dupes_task()
        config = toy_pipeline(evaluators=[DuplicatesEvaluatorConfig(name="dupes")], tasks=[task])
        result = run_task(task, config)
        assert isinstance(result, EvaluatorResult)
        assert result.success, result.errors
        assert exact_groups(result.output["rows"]) == {(0, 5)}
        meta = result.metadata
        assert meta.evaluator == "quality.duplicates"
        assert meta.dataset_id == "toy"
        assert meta.execution_time_s is not None
        assert meta.source_descriptions
        assert meta.resolved_config["evaluator"]["type"] == "quality.duplicates"
        assert "workflow" not in meta.resolved_config
        assert result.dataset is not None

    def test_two_sources_read_as_one_duplicate_search(self):
        task = _dupes_task(["a", "b"])
        config = toy_pipeline(evaluators=[DuplicatesEvaluatorConfig(name="dupes")], tasks=[task], sources=("a", "b"))
        result = run_task(task, config)
        assert result.success, result.errors
        assert result.sources is not None
        assert set(result.sources) == {"a", "b"}

    def test_an_empty_source_never_raises(self):
        """Review Focus 3: nothing to evaluate is a result, never a crash — for either evaluator."""
        dupes_task = _dupes_task()
        outliers_task = EvaluatorTaskConfig(name="outliers_task", workflow="outliers", sources="src")
        config = toy_pipeline(
            evaluators=[DuplicatesEvaluatorConfig(name="dupes"), OutliersEvaluatorConfig(name="outliers")],
            tasks=[dupes_task, outliers_task],
            dataset=ToyImages(count=0),
        )
        dupes_result = run_task(dupes_task, config)
        outliers_result = run_task(outliers_task, config)

        assert dupes_result.success, dupes_result.errors
        assert outliers_result.success, outliers_result.errors
        assert dupes_result.report()
        assert outliers_result.report()
        assert dupes_result.output["rows"] == []
        assert outliers_result.output["rows"] == []

    def test_run_task_checks_a_task_outside_config_tasks(self):
        """A task passed to `run_task` need not be in `config.tasks`, so it skips the
        config-load checks; `_run_single_task` must run the same check itself, before
        cluster-mode code reads two sources it can only unpack one of."""
        task = EvaluatorTaskConfig(name="dupes_task2", workflow="dupes", sources=["a", "b"])
        config = toy_pipeline(
            evaluators=[DuplicatesEvaluatorConfig(name="dupes", cluster_sensitivity=1.0)],
            sources=("a", "b"),
        )
        match = r"Task 'dupes_task2' runs evaluator 'dupes' \(quality\.duplicates\), which"
        with pytest.raises(ValueError, match=match):
            run_task(task, config)


class TestAlongsideWorkflows:
    def test_one_run_holds_both_kinds(self):
        tasks = [TaskConfig(name="clean_task", workflow="clean", sources="src"), _dupes_task()]
        config = toy_pipeline(evaluators=[DuplicatesEvaluatorConfig(name="dupes")], workflows=[_CLEAN], tasks=tasks)
        results = run_tasks(config)
        assert isinstance(results[0], WorkflowResult)
        assert isinstance(results[1], EvaluatorResult)
        assert [r.to_dict()["kind"] for r in results] == ["workflow", "evaluator"]

    def test_duplicates_agree_with_data_cleaning(self):
        tasks = [TaskConfig(name="clean_task", workflow="clean", sources="src"), _dupes_task()]
        config = toy_pipeline(
            evaluators=[DuplicatesEvaluatorConfig(name="dupes")],
            workflows=[_CLEAN],
            tasks=tasks,
            dataset=ToyImages(near_duplicate=True),
        )
        cleaning, evaluator = run_tasks(config)
        assert isinstance(cleaning, WorkflowResult)
        assert isinstance(evaluator, EvaluatorResult)

        cleaning_exact = {tuple(sorted(group)) for group in cleaning.data.raw.duplicates["items"].get("exact", [])}
        assert exact_groups(evaluator.output["rows"]) == cleaning_exact

        # Hash mode too: near-duplicate groups must agree, not just exact ones.
        cleaning_near = {
            tuple(sorted(group["indices"])) for group in cleaning.data.raw.duplicates["items"].get("near", [])
        }
        evaluator_near = {
            tuple(sorted(row["item_indices"]))
            for row in evaluator.output["rows"]
            if row["dup_type"] == "near" and row["level"] == "item"
        }
        assert evaluator_near == cleaning_near == {(3, 9)}

    def test_outliers_agree_with_data_cleaning(self):
        outliers = OutliersEvaluatorConfig(
            name="outliers", flags=["dimension", "pixel", "visual"], outlier_threshold="zscore", per_target=True
        )
        tasks = [
            TaskConfig(name="clean_task", workflow="clean", sources="src"),
            EvaluatorTaskConfig(name="outliers_task", workflow="outliers", sources="src"),
        ]
        config = toy_pipeline(evaluators=[outliers], workflows=[_CLEAN], tasks=tasks)
        cleaning, evaluator = run_tasks(config)
        assert isinstance(cleaning, WorkflowResult)
        assert isinstance(evaluator, EvaluatorResult)
        cleaning_flags = {(i["item_index"], i["metric_name"]) for i in cleaning.data.raw.img_outliers["issues"]}
        evaluator_flags = {(r["item_index"], r["metric_name"]) for r in evaluator.output["rows"]}
        assert evaluator_flags == cleaning_flags

    def test_stats_cached_by_data_cleaning_serve_the_evaluator(self, tmp_path):
        tasks = [TaskConfig(name="clean_task", workflow="clean", sources="src"), _dupes_task()]
        config = toy_pipeline(evaluators=[DuplicatesEvaluatorConfig(name="dupes")], workflows=[_CLEAN], tasks=tasks)
        with patch.object(cache_module, "_do_compute_stats", wraps=cache_module._do_compute_stats) as compute:
            run_task(tasks[0], config, cache_dir=tmp_path)
            computed_by_cleaning = compute.call_count
            run_task(tasks[1], config, cache_dir=tmp_path)
        assert computed_by_cleaning >= 1
        assert compute.call_count == computed_by_cleaning
