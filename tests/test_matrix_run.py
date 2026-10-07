"""Running a task's matrix: one result, one draw of each source, a fresh seed per run, shared work, and failures
kept (task-matrix spec §5, §6)."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

from dataeval_flow import MatrixResult, run_task, run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.config import ImageFolderDatasetConfig, SourceConfig, TaskConfig, ViewConfig, ViewOperation
from dataeval_flow.steps import ChainResult
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages

_CLEANING = {
    "name": "cleaning",
    "type": "quality",
    "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"},
}


@pytest.fixture(autouse=True)
def _fresh_cache() -> Any:
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _config(
    matrix: Any, *, seed: int | None = None, shuffled: bool = False, entry: dict[str, Any] | None = None, **task: Any
) -> Any:
    extra: dict[str, Any] = {"seed": seed}
    if shuffled:  # read the toy through an unseeded shuffle, cut to 8 items
        operations = [ViewOperation(type="Shuffle", params={}), ViewOperation(type="Limit", params={"size": 8})]
        extra["views"] = [ViewConfig(name="mixed", operations=operations)]
        extra["sources"] = [SourceConfig(name="src", dataset="src_data", view="mixed")]
    return chain_pipeline(
        workflows=[entry or _CLEANING],
        extractor=True,
        extra=extra,
        tasks=[{"name": "t", "workflow": "cleaning", "sources": "src", "matrix": matrix, **task}],
    )


def _matrix(config: Any) -> MatrixResult:
    """Task `t`'s result, which a matrix task returns as a :class:`MatrixResult`."""
    result = run_tasks(config)["t"]
    assert isinstance(result, MatrixResult)
    return result


def _drawn(result: Any) -> Any:
    """The dataset `result`'s run read from source `src`."""
    return result.sources["src"]


def test_a_matrix_task_returns_one_result_holding_every_run() -> None:
    result = _matrix(_config({"outliers.outlier_threshold": [2.0, 3.0]}))
    assert result.kind == "matrix"
    assert result.type == "quality"
    assert result.keys == ["outliers.outlier_threshold"]
    assert [(run.number, run.label) for run in result.runs] == [
        (1, "outliers.outlier_threshold=2.0"),
        (2, "outliers.outlier_threshold=3.0"),
    ]
    assert all(isinstance(run.result, ChainResult) and run.result.success for run in result.runs)
    assert result.success


def test_every_run_reads_one_draw_of_an_unseeded_shuffled_source() -> None:
    result = _matrix(_config({"outliers.outlier_threshold": [2.0, 3.0, 4.0]}, shuffled=True))
    drawn = [_drawn(run.result) for run in result.runs]
    assert all(dataset is drawn[0] for dataset in drawn)


def test_under_a_seed_a_run_reads_the_draw_a_lone_task_reads() -> None:
    def order(dataset: Any) -> list[int]:
        return [int(np.asarray(dataset[i][0]).sum()) for i in range(len(dataset))]

    matrix = _matrix(_config({"outliers.outlier_threshold": [3.0]}, seed=7, shuffled=True))
    DatasetCache.clear_instances()
    lone_config = _config({"outliers.outlier_threshold": [3.0]}, seed=7, shuffled=True)
    lone = run_task(lone_config, TaskConfig(name="t", workflow="cleaning", sources="src"))
    assert order(_drawn(matrix.runs[0].result)) == order(_drawn(lone))


def test_a_run_s_findings_equal_its_settings_run_as_a_lone_task() -> None:
    def seen(result: Any) -> list[tuple[str, str, str | None]]:
        return [(finding.title, finding.severity, finding.brief) for finding in result.findings]

    matrix = _matrix(_config({"outliers.outlier_threshold": [2.0]}, seed=1))
    DatasetCache.clear_instances()
    lone_config = _config(
        {"outliers.outlier_threshold": [3.0]},
        seed=1,
        entry={**_CLEANING, "outliers": {**_CLEANING["outliers"], "outlier_threshold": 2.0}},
    )
    lone = run_task(lone_config, TaskConfig(name="t", workflow="cleaning", sources="src"))
    assert seen(matrix.runs[0].result) == seen(lone)


def test_statistics_are_computed_once_across_a_threshold_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataeval_flow._cache as cache

    calls: list[int] = []
    real = cache._do_compute_stats
    monkeypatch.setattr(cache, "_do_compute_stats", lambda *a, **k: calls.append(1) or real(*a, **k))
    run_task(_config({"outliers.outlier_threshold": [3.0]}), TaskConfig(name="t", workflow="cleaning", sources="src"))
    lone = len(calls)
    assert lone >= 1, "spy on the function that computes statistics: this one was never called"
    DatasetCache.clear_instances()
    calls.clear()
    run_tasks(_config({"outliers.outlier_threshold": [2.0, 3.0, 4.0]}))
    assert len(calls) == lone


def test_runs_reading_one_sources_value_share_one_extractor_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataeval_flow._embeddings as embeddings

    made: list[object] = []
    real = embeddings.new_extractor_scope
    monkeypatch.setattr(embeddings, "new_extractor_scope", lambda: made.append(1) or real())
    config = chain_pipeline(
        workflows=[_CLEANING],
        datasets={"a": ToyImages(), "b": ToyImages(seed=1)},
        extractor=True,
        tasks=[
            {
                "name": "t",
                "workflow": "cleaning",
                "sources": "a",
                "matrix": {"outliers.outlier_threshold": [2.0, 3.0], "sources": ["a", "b"]},
            }
        ],
    )
    run_tasks(config)
    assert len(made) == 2


def test_an_inner_scope_joins_the_open_one() -> None:
    from dataeval_flow._embeddings import _task_scope, new_extractor_scope, shared_extractor_scope

    scope = new_extractor_scope()
    with shared_extractor_scope(scope), shared_extractor_scope():
        assert _task_scope.get() is scope


def test_a_failed_run_is_kept_while_the_others_finish_and_fails_the_matrix() -> None:
    # k-means can't make 50 clusters of 12 items: the outliers step fails in that run only. If DataEval accepts it,
    # pick another setting that validates at load and fails at run time, and say so in your report.
    entry = {
        **_CLEANING,
        "outliers": {**_CLEANING["outliers"], "cluster_threshold": 1.0, "cluster_algorithm": "kmeans"},
    }
    result = _matrix(_config({"outliers.n_clusters": [None, 50]}, entry=entry, extractor="flat"))
    assert [run.result.success for run in result.runs] == [True, False]
    assert not result.success
    assert result.health["status"] == "failed"
    assert result.health["failed_runs"] == [2]
    assert result.errors[0].startswith("run 2 (outliers.n_clusters=50): ")


def test_a_run_that_raises_becomes_a_failed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataeval_flow._orchestrator as orchestrator

    real = orchestrator._run_resolved

    def flaky(task: Any, config: Any, *args: Any, **kwargs: Any) -> Any:
        if config.workflows[0].outliers.outlier_threshold == 3.0:
            raise RuntimeError("boom")
        return real(task, config, *args, **kwargs)

    monkeypatch.setattr(orchestrator, "_run_resolved", flaky)
    result = _matrix(_config({"outliers.outlier_threshold": [2.0, 3.0]}))
    assert [run.result.success for run in result.runs] == [True, False]
    assert isinstance(result.runs[1].result, ChainResult)
    assert "boom" in result.runs[1].result.errors[0]


def test_a_source_that_does_not_resolve_raises_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataeval_flow._sources as sources

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("no such dataset")

    monkeypatch.setattr(sources, "resolve_source", broken)
    with pytest.raises(ValueError, match="no such dataset"):
        run_tasks(_config({"outliers.outlier_threshold": [2.0, 3.0]}))


def test_a_merge_of_conflicting_value_ranges_raises_once_with_no_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import dataeval_flow._orchestrator as orchestrator

    ran: list[int] = []
    real = orchestrator._run_resolved
    monkeypatch.setattr(orchestrator, "_run_resolved", lambda *a, **k: ran.append(1) or real(*a, **k))
    for name in ("a", "b"):  # an in-memory dataset declares no range, so each operand is a folder of two images
        (tmp_path / name).mkdir()
        for index in range(2):
            Image.new("RGB", (8, 8), color=(index * 90, 0, 0)).save(tmp_path / name / f"{index}.png")
    datasets = [
        ImageFolderDatasetConfig(name="a_data", path="a", value_range=(0.0, 255.0)),
        ImageFolderDatasetConfig(name="b_data", path="b", value_range=(0.0, 1.0)),
    ]
    sources = [
        SourceConfig(name="a", dataset="a_data"),
        SourceConfig(name="b", dataset="b_data"),
        SourceConfig(name="merged", merge=["a", "b"]),
    ]
    config = chain_pipeline(
        workflows=[_CLEANING],
        extra={"datasets": datasets, "sources": sources},
        tasks=[
            {
                "name": "t",
                "workflow": "cleaning",
                "sources": "merged",
                "matrix": {"outliers.outlier_threshold": [2.0, 3.0]},
            }
        ],
    )
    with pytest.raises(ValueError, match="merges datasets declaring different `value_range`s"):
        run_tasks(config, data_dir=tmp_path)
    assert ran == []


def test_a_run_that_raised_keeps_its_envelope_and_the_matrix_its_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataeval_flow._orchestrator as orchestrator

    real = orchestrator._run_resolved

    def flaky(task: Any, config: Any, *args: Any, **kwargs: Any) -> Any:
        if config.workflows[0].outliers.outlier_threshold == 2.0:
            raise RuntimeError("boom")
        return real(task, config, *args, **kwargs)

    monkeypatch.setattr(orchestrator, "_run_resolved", flaky)
    result = _matrix(_config({"outliers.outlier_threshold": [2.0, 3.0]}))
    raised = result.runs[0].result
    assert not raised.success
    assert raised.metadata.resolved_config["workflow"]["outliers"]["outlier_threshold"] == 2.0
    assert raised.metadata.source_descriptions == ["src (src_data)"]
    assert raised._entry == "cleaning"
    assert result._entry == "cleaning"
    assert "cleaning (quality), matrix of 2 runs" in result.report()


def test_the_json_holds_each_run_s_result_as_its_type_writes_it() -> None:
    result = _matrix(_config({"outliers.outlier_threshold": [2.0, 3.0]}))
    payload = cast("dict[str, Any]", result.to_dict())
    assert payload["kind"] == "matrix"
    assert payload["type"] == "quality"
    assert payload["keys"] == ["outliers.outlier_threshold"]
    assert payload["health"] == result.health
    assert [run["number"] for run in payload["runs"]] == [1, 2]
    assert payload["runs"][0]["values"] == {"outliers.outlier_threshold": 2.0}
    assert [run["result"] for run in payload["runs"]] == [run.result.to_dict() for run in result.runs]
    assert "assets" not in payload


def test_each_run_keeps_the_thumbnails_of_the_data_it_read() -> None:
    # A chain's thumbnails are keyed by node address, `data` here whichever source the run binds to it.
    datasets = {"a": ToyImages(), "b": ToyImages(seed=1)}

    def config(**task: Any) -> Any:
        return chain_pipeline(
            workflows=[_CLEANING], datasets=datasets, tasks=[{"name": "t", "workflow": "cleaning", **task}]
        )

    matrix = _matrix(config(sources="a", matrix={"sources": ["a", "b"]}))
    alone = [cast("dict[str, Any]", run_tasks(config(sources=name))["t"].to_dict())["assets"] for name in "ab"]
    payload = cast("dict[str, Any]", matrix.to_dict())
    assert [run["result"]["assets"] for run in payload["runs"]] == alone
    shared = {str(asset["item"]): asset["data"] for asset in alone[0]}
    assert any(shared.get(str(asset["item"]), asset["data"]) != asset["data"] for asset in alone[1])
    page = matrix.to_html()
    assert all(asset["data"] in page for assets in alone for asset in assets)


def test_the_envelope_names_the_task_as_written_and_the_sources_read() -> None:
    result = _matrix(_config({"outliers.outlier_threshold": [2.0, 3.0]}, seed=3))
    config = result.metadata.resolved_config
    assert config["task"]["matrix"] == {"outliers.outlier_threshold": [2.0, 3.0]}
    assert config["sources"] == ["src"]
    assert config["seed"] == 3
    assert result.metadata.execution_time_s is not None


def test_the_report_writes_the_matrix_as_yaml_writes_it() -> None:
    grids = [
        {"outliers.outlier_threshold": ["iqr", 2, 4]},
        {"duplicates.flags": [None]},
        {"outliers.outlier_threshold": {"from": 2.5, "to": 3.5, "step": 0.5}},
    ]
    report = _matrix(_config(grids)).report(detailed=False)
    assert "- {outliers.outlier_threshold: [iqr, 2, 4]}" in report  # not `range(2, 5, 2)` for 2 and 4
    assert "- {duplicates.flags: [null]}" in report  # not `None`
    assert "- {outliers.outlier_threshold: {from: 2.5, to: 3.5, step: 0.5}}" in report


def test_run_task_runs_a_matrix_task_the_config_does_not_hold() -> None:
    config = _config({"outliers.outlier_threshold": [3.0]})
    task = TaskConfig.model_validate(
        {
            "name": "other",
            "workflow": "cleaning",
            "sources": "src",
            "matrix": {"outliers.outlier_threshold": [2.0, 4.0]},
        }
    )
    result = run_task(config, task)
    assert isinstance(result, MatrixResult)
    assert len(result.runs) == 2


def test_run_task_with_a_matrix_task_naming_no_entry_says_which() -> None:
    config = _config({"outliers.outlier_threshold": [3.0]})
    task = TaskConfig.model_validate(
        {"name": "other", "workflow": "missing", "sources": "src", "matrix": {"outliers.outlier_threshold": [2.0, 4.0]}}
    )
    with pytest.raises(ValueError, match="Unknown workflow: 'missing'"):
        run_task(config, task)


def test_a_matrix_result_records_library_versions() -> None:
    from dataeval_flow._versions import library_versions

    result = _matrix(_config({"outliers.outlier_threshold": [2.0, 3.0]}))
    assert result.metadata.library_versions == library_versions()
