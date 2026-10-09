"""TC-6-3 — running tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import dataeval_flow
from dataeval_flow import Result, load_config, run, run_task, run_tasks
from dataeval_flow.config import DatasetProtocolConfig, PipelineConfig, SourceConfig, TaskConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig, DuplicatesResult, LabelHealthResult
from dataeval_flow.steps import ChainResult
from verification.functional.orchestration.support import (
    QUALITY,
    BrightnessEvaluator,
    InMemoryImages,
    write_project,
)

pytestmark = [pytest.mark.required, pytest.mark.usefixtures("fresh_caches")]

TASKS = [
    {"name": "clean", "workflow": "q", "sources": "main"},
    {"name": "dupes", "evaluator": "dup", "sources": "main"},
    {"name": "labels", "evaluator": "labels", "sources": "main"},
    {"name": "off", "evaluator": "dup", "sources": "main", "enabled": False},
]


@pytest.fixture
def project(tmp_path: Path) -> tuple[PipelineConfig, Path]:
    """A pipeline over an image folder with a workflow task, two evaluator tasks and a disabled one."""
    path, _ = write_project(
        tmp_path,
        workflows=[QUALITY],
        evaluators=[{"name": "dup", "type": "duplicates"}, {"name": "labels", "type": "label-health"}],
        tasks=TASKS,
    )
    return load_config(path), tmp_path


@pytest.fixture
def failing_project(tmp_path: Path, example_plugin: dict[str, Any]) -> tuple[PipelineConfig, Path]:
    """A pipeline whose second and fourth tasks raise when they run, and whose others succeed."""
    path, _ = write_project(
        tmp_path,
        workflows=[QUALITY, {"name": "count_boom", "type": "example.count", "explode": True}],
        evaluators=[
            {"name": "dup", "type": "duplicates"},
            {"name": "bright", "type": "example.brightness"},
            {"name": "boom", "type": "example.brightness", "explode": True},
        ],
        tasks=[
            {"name": "first", "workflow": "q", "sources": "main"},
            {"name": "evaluator_fails", "evaluator": "boom", "sources": "main"},
            {"name": "second", "evaluator": "dup", "sources": "main"},
            {"name": "workflow_fails", "workflow": "count_boom", "sources": "main"},
            {"name": "third", "evaluator": "bright", "sources": "main"},
        ],
    )
    return load_config(path), tmp_path


class TestRunTasks:
    def test_run_tasks_returns_one_result_per_enabled_task_keyed_by_name(
        self, project: tuple[PipelineConfig, Path]
    ) -> None:
        config, root = project
        results = run_tasks(config, data_dir=root)

        assert list(results) == ["clean", "dupes", "labels"]  # config order; the disabled task is left out
        assert all(isinstance(result, Result) for result in results.values())
        assert all(result.success for result in results.values()), [r.errors for r in results.values()]

    def test_each_result_is_of_the_kind_and_type_its_task_ran(self, project: tuple[PipelineConfig, Path]) -> None:
        config, root = project
        results = run_tasks(config, data_dir=root)

        assert isinstance(results["clean"], ChainResult)
        assert (results["clean"].kind, results["clean"].type) == ("workflow", "quality")
        assert isinstance(results["dupes"], DuplicatesResult)
        assert (results["dupes"].kind, results["dupes"].type) == ("evaluator", "duplicates")
        assert isinstance(results["labels"], LabelHealthResult)
        assert results["labels"].output.data()["item_count"] == 10

    def test_every_result_carries_the_metadata_envelope(self, project: tuple[PipelineConfig, Path]) -> None:
        config, root = project
        for name, result in run_tasks(config, data_dir=root).items():
            assert result.metadata.tool == "dataeval-flow", name
            assert result.metadata.tool_version == dataeval_flow.__version__, name
            assert result.metadata.dataset_id == "ds", name
            assert result.metadata.source_descriptions == ["main (ds)"], name

    def test_a_result_reports_as_text(self, project: tuple[PipelineConfig, Path]) -> None:
        config, root = project
        results = run_tasks(config, data_dir=root)
        assert "QUALITY" in results["clean"].report()
        assert "DUPLICATES" in results["dupes"].report().upper()

    def test_relative_paths_in_the_config_resolve_against_data_dir(
        self, project: tuple[PipelineConfig, Path], tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        config, root = project
        with pytest.raises(FileNotFoundError, match="Image folder not found"):
            run_tasks(config, data_dir=tmp_path_factory.mktemp("elsewhere"))
        assert run_tasks(config, "dupes", data_dir=root)["dupes"].success

    def test_a_run_of_a_config_is_repeatable(self, project: tuple[PipelineConfig, Path]) -> None:
        config, root = project
        first = run_tasks(config, data_dir=root)
        second = run_tasks(config, data_dir=root)
        assert first["dupes"].to_dict()["output"] == second["dupes"].to_dict()["output"]
        assert first["labels"].output.data() == second["labels"].output.data()


class TestSelectingTasks:
    def test_naming_one_task_runs_only_it(self, project: tuple[PipelineConfig, Path]) -> None:
        config, root = project
        assert list(run_tasks(config, "labels", data_dir=root)) == ["labels"]

    def test_naming_several_runs_them_in_the_order_given_once_each(self, project: tuple[PipelineConfig, Path]) -> None:
        config, root = project
        results = run_tasks(config, ["labels", "clean", "labels"], data_dir=root)
        assert list(results) == ["labels", "clean"]

    def test_naming_a_task_runs_it_though_it_is_disabled(self, project: tuple[PipelineConfig, Path]) -> None:
        config, root = project
        results = run_tasks(config, "off", data_dir=root)
        assert list(results) == ["off"]
        assert results["off"].success

    def test_an_unknown_task_name_is_refused_naming_the_tasks(self, project: tuple[PipelineConfig, Path]) -> None:
        config, root = project
        with pytest.raises(ValueError, match=r"Unknown task: 'nope'. Available: \['clean', 'dupes', 'labels', 'off'\]"):
            run_tasks(config, ["clean", "nope"], data_dir=root)

    def test_a_pipeline_with_no_tasks_is_refused(self) -> None:
        with pytest.raises(ValueError, match="No tasks defined in pipeline config"):
            run_tasks(PipelineConfig())

    def test_a_pipeline_whose_tasks_are_all_disabled_is_refused(self, tmp_path: Path) -> None:
        path, _ = write_project(
            tmp_path,
            evaluators=[{"name": "dup", "type": "duplicates"}],
            tasks=[{"name": "t", "evaluator": "dup", "sources": "main", "enabled": False}],
        )
        with pytest.raises(ValueError, match="All tasks are disabled"):
            run_tasks(load_config(path), data_dir=tmp_path)

    def test_run_task_runs_one_task_and_returns_its_result_not_a_mapping(
        self, project: tuple[PipelineConfig, Path]
    ) -> None:
        config, root = project
        result = run_task(config, "dupes", data_dir=root)
        assert isinstance(result, DuplicatesResult)
        assert result.success

    def test_run_task_takes_a_task_the_config_does_not_hold(self, project: tuple[PipelineConfig, Path]) -> None:
        config, root = project
        extra = TaskConfig.model_validate({"name": "extra", "evaluator": "labels", "sources": "main"})
        assert config.tasks is not None
        assert "extra" not in [task.name for task in config.tasks]
        assert isinstance(run_task(config, extra, data_dir=root), LabelHealthResult)

    def test_run_task_runs_a_task_though_it_is_disabled(self, project: tuple[PipelineConfig, Path]) -> None:
        config, root = project
        assert run_task(config, "off", data_dir=root).success

    def test_run_task_with_an_unknown_name_is_refused(self, project: tuple[PipelineConfig, Path]) -> None:
        config, root = project
        with pytest.raises(ValueError, match="Unknown task: 'nope'"):
            run_task(config, "nope", data_dir=root)


class TestOnResult:
    def test_on_result_is_called_with_each_task_as_soon_as_it_finishes(
        self, failing_project: tuple[PipelineConfig, Path]
    ) -> None:
        config, root = failing_project
        seen: list[tuple[str, bool, int]] = []

        def record(name: str, result: Result[Any, Any]) -> None:
            seen.append((name, result.success, BrightnessEvaluator.calls))

        BrightnessEvaluator.calls = 0
        results = run_tasks(config, data_dir=root, on_result=record)

        # `boom` and `bright` are the two tasks that run the plug-in evaluator: the callback for a task runs before
        # the next one starts, so it sees the count at the time.
        assert [(name, success) for name, success, _ in seen] == [
            ("first", True),
            ("evaluator_fails", False),
            ("second", True),
            ("workflow_fails", False),
            ("third", True),
        ]
        assert [calls for _, _, calls in seen] == [0, 1, 1, 1, 2]
        assert list(results) == [name for name, _, _ in seen]

    def test_on_result_gets_the_same_objects_run_tasks_returns(self, project: tuple[PipelineConfig, Path]) -> None:
        config, root = project
        handed: dict[str, Result[Any, Any]] = {}
        results = run_tasks(config, data_dir=root, on_result=handed.__setitem__)
        assert list(handed) == list(results)
        assert all(handed[name] is results[name] for name in results)

    def test_what_finished_is_kept_when_a_later_task_cannot_be_set_up(self, tmp_path: Path) -> None:
        """A task that names a source the pipeline does not define raises; the callback has the tasks before it."""
        path, _ = write_project(
            tmp_path,
            evaluators=[{"name": "labels", "type": "label-health"}],
            tasks=[
                {"name": "good", "evaluator": "labels", "sources": "main"},
                {"name": "bad", "evaluator": "labels", "sources": "missing"},
            ],
        )
        kept: dict[str, Result[Any, Any]] = {}
        with pytest.raises(ValueError, match=r"Unknown source: 'missing'. Available: \['main'\]"):
            run_tasks(load_config(path), data_dir=tmp_path, on_result=kept.__setitem__)
        assert list(kept) == ["good"]
        assert kept["good"].success


class TestFailureIsolation:
    def test_a_failing_task_is_recorded_in_its_result_and_the_remaining_tasks_run(
        self, failing_project: tuple[PipelineConfig, Path]
    ) -> None:
        config, root = failing_project
        results = run_tasks(config, data_dir=root)

        assert list(results) == ["first", "evaluator_fails", "second", "workflow_fails", "third"]
        assert {name: result.success for name, result in results.items()} == {
            "first": True,
            "evaluator_fails": False,
            "second": True,
            "workflow_fails": False,
            "third": True,
        }

    def test_a_failed_evaluator_result_names_the_error_and_refuses_to_give_output(
        self, failing_project: tuple[PipelineConfig, Path]
    ) -> None:
        config, root = failing_project
        failed = run_tasks(config, "evaluator_fails", data_dir=root)["evaluator_fails"]

        assert failed.errors == ["RuntimeError: example.brightness was told to explode"]
        assert (failed.kind, failed.type) == ("evaluator", "example.brightness")
        with pytest.raises(RuntimeError, match="explode"):
            _ = failed.output
        assert "FAILED" in failed.report()
        assert "was told to explode" in failed.report()

    def test_a_failed_workflow_result_names_the_step_that_failed(
        self, failing_project: tuple[PipelineConfig, Path]
    ) -> None:
        config, root = failing_project
        failed = run_tasks(config, "workflow_fails", data_dir=root)["workflow_fails"]

        assert isinstance(failed, ChainResult)
        assert failed.errors == ["items: RuntimeError: example.items was told to explode"]
        assert (failed.kind, failed.type) == ("workflow", "example.count")
        with pytest.raises(RuntimeError, match="did not complete"):
            _ = failed.output

    def test_a_failed_result_still_carries_the_envelope(self, failing_project: tuple[PipelineConfig, Path]) -> None:
        config, root = failing_project
        failed = run_tasks(config, "evaluator_fails", data_dir=root)["evaluator_fails"]

        assert failed.metadata.tool == "dataeval-flow"
        assert failed.metadata.tool_version == dataeval_flow.__version__
        assert failed.metadata.dataset_id == "ds"
        assert failed.metadata.resolved_config["evaluator"]["name"] == "boom"
        assert failed.to_dict()["kind"] == "evaluator"

    def test_a_task_after_a_failure_gives_the_result_it_gives_alone(
        self, failing_project: tuple[PipelineConfig, Path]
    ) -> None:
        config, root = failing_project
        after_failures = run_tasks(config, data_dir=root)["second"]
        alone = run_tasks(config, "second", data_dir=root)["second"]
        assert after_failures.to_dict()["output"] == alone.to_dict()["output"]

    def test_a_failure_in_the_first_task_does_not_stop_the_next(
        self, failing_project: tuple[PipelineConfig, Path]
    ) -> None:
        config, root = failing_project
        results = run_tasks(config, ["evaluator_fails", "first"], data_dir=root)
        assert list(results) == ["evaluator_fails", "first"]
        assert not results["evaluator_fails"].success
        assert results["first"].success


class TestConfigErrorsRaise:
    """What the pipeline gets wrong, rather than what a run does wrong, is raised and not recorded in a result."""

    @pytest.mark.parametrize(
        ("task", "message"),
        [
            ({"sources": "missing"}, r"Unknown source: 'missing'. Available: \['main'\]"),
            ({"sources": "main", "extractor": "missing"}, r"Unknown extractor: 'missing'. Available: \['flat'\]"),
        ],
    )
    def test_a_task_naming_a_source_or_extractor_the_pipeline_lacks_raises(
        self, tmp_path: Path, task: dict[str, Any], message: str
    ) -> None:
        path, _ = write_project(
            tmp_path,
            evaluators=[{"name": "dup", "type": "duplicates"}],
            tasks=[{"name": "t", "evaluator": "dup", **task}],
        )
        with pytest.raises(ValueError, match=message):
            run_tasks(load_config(path), data_dir=tmp_path)

    def test_a_source_naming_a_dataset_that_does_not_load_raises_the_loaders_error(self, tmp_path: Path) -> None:
        path, _ = write_project(
            tmp_path,
            datasets=[{"name": "ds", "format": "image_folder", "path": "gone", "infer_labels": True}],
            evaluators=[{"name": "dup", "type": "duplicates"}],
            tasks=[{"name": "t", "evaluator": "dup", "sources": "main"}],
        )
        with pytest.raises(FileNotFoundError, match="Image folder not found"):
            run_tasks(load_config(path), data_dir=tmp_path)


class TestRun:
    """`run(config, data)` runs one workflow or evaluator on datasets already in memory."""

    def test_run_returns_the_result_class_of_the_config(self) -> None:
        result = run(DuplicatesConfig(), InMemoryImages(12))
        assert isinstance(result, DuplicatesResult)
        assert result.success, result.errors
        assert result.metadata.dataset_id == "dataset"

    def test_run_matches_the_same_task_run_in_a_pipeline(self) -> None:
        dataset = InMemoryImages(12)
        direct = run(DuplicatesConfig(), dataset)
        config = PipelineConfig(
            datasets=[DatasetProtocolConfig(name="ds", dataset=dataset)],
            sources=[SourceConfig(name="main", dataset="ds")],
            evaluators=[DuplicatesConfig(name="dup")],
            tasks=[TaskConfig(name="t", workflow="dup", kind="evaluator", sources="main")],
        )
        piped = run_tasks(config)["t"]
        assert isinstance(piped, DuplicatesResult)
        assert direct.to_dict()["output"] == piped.to_dict()["output"]
        assert [row["item_indices"] for row in direct.to_dict()["output"]["rows"]] == [[0, 5]]  # the planted copy

    def test_run_takes_a_workflow_config_and_returns_its_chain(self) -> None:
        from dataeval_flow.workflows.quality import QualityConfig

        result = run(QualityConfig(outliers={"flags": ["pixel"], "outlier_threshold": "zscore"}), InMemoryImages(12))  # type: ignore[arg-type]
        assert isinstance(result, ChainResult)
        assert (result.kind, result.type) == ("workflow", "quality")
        assert result.success, result.errors

    def test_run_names_the_sources_of_a_mapping_in_the_order_given(self) -> None:
        data = {"first": InMemoryImages(12), "second": InMemoryImages(12, seed=1)}
        result = run(DuplicatesConfig(), data)
        assert result.success, result.errors
        assert [description.split(" ")[0] for description in result.metadata.source_descriptions] == ["first", "second"]

    def test_run_refuses_no_datasets(self) -> None:
        with pytest.raises(ValueError, match="run\\(\\) needs at least one dataset"):
            run(DuplicatesConfig(), {})

    def test_a_run_that_raised_returns_a_failed_result_of_the_configs_class(
        self,
        example_plugin: dict[str, Any],
    ) -> None:
        from verification.functional.orchestration.support import BrightnessConfig, BrightnessResult

        result = run(BrightnessConfig(explode=True), InMemoryImages(12))
        assert isinstance(result, BrightnessResult)
        assert not result.success
        assert result.errors == ["RuntimeError: example.brightness was told to explode"]
