"""TC-11-1 — task matrices: one task run once per combination of values, and one result that compares the runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from dataeval_flow import MatrixResult, MatrixRun, load_config, run_tasks
from dataeval_flow.config import SourceConfig, ViewConfig, ViewOperation
from dataeval_flow.config.extractors import FlattenExtractorConfig
from dataeval_flow.steps import ChainResult
from verification.fixtures import write_image_folder
from verification.functional.workflows._synthetic import Detections, Images, pipeline

pytestmark = pytest.mark.required

CLEANING: dict[str, Any] = {
    "name": "cleaning",
    "type": "quality",
    "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"},
}


def planted() -> Images:
    """60 images with one white outlier (7) and one exact duplicate pair (0, 5) among values of 100 to 149."""
    return Images(60, planted=True, shape=(3, 16, 16), value_range=(100, 150))


def matrix_task(matrix: Any, **task: Any) -> dict[str, Any]:
    return {"name": "t", "workflow": "cleaning", "sources": ["src"], "matrix": matrix, **task}


def run_matrix(matrix: Any, **kwargs: Any) -> MatrixResult:
    datasets = kwargs.pop("datasets", {"src": planted()})
    config = pipeline(
        workflows=kwargs.pop("workflows", [CLEANING]),
        tasks=[matrix_task(matrix, **kwargs.pop("task", {}))],
        datasets=datasets,
        extractor=kwargs.pop("extractor", False),
        extra={"seed": 0, **kwargs.pop("extra", {})},
    )
    result = run_tasks(config, **kwargs)["t"]
    assert isinstance(result, MatrixResult)
    return result


class TestTaskMatrix:
    def test_a_list_runs_the_task_once_per_value_in_order(self) -> None:
        result = run_matrix({"checks.image-outliers.warning": [0.5, 5.0]})

        assert result.success, result.errors
        assert (result.kind, result.type, result.keys) == ("matrix", "quality", ["checks.image-outliers.warning"])
        assert all(isinstance(run, MatrixRun) for run in result.runs)
        assert [(run.number, run.label, run.values) for run in result.runs] == [
            (1, "checks.image-outliers.warning=0.5", {"checks.image-outliers.warning": 0.5}),
            (2, "checks.image-outliers.warning=5.0", {"checks.image-outliers.warning": 5.0}),
        ]
        assert all(isinstance(run.result, ChainResult) for run in result.runs)
        severities = [
            next(f.severity for f in run.result.findings if f.title == "Image Outliers") for run in result.runs
        ]  # type: ignore[union-attr]
        assert severities == ["warning", "info"]  # 1.7% of images flagged: over 0.5, under 5.0

    def test_each_run_gives_the_result_its_settings_give_as_a_lone_task(self) -> None:
        matrix = run_matrix({"outliers.outlier_threshold": [["zscore", 2.0], ["modzscore", 3.5]]})
        lone_entry = {**CLEANING, "outliers": {**CLEANING["outliers"], "outlier_threshold": ["modzscore", 3.5]}}
        lone = run_tasks(
            pipeline(
                workflows=[lone_entry],
                tasks=[{"name": "t", "workflow": "cleaning", "sources": ["src"]}],
                datasets={"src": planted()},
                extra={"seed": 0},
            )
        )["t"]

        def seen(result: Any) -> list[tuple[str, str, str | None]]:
            return [(f.title, f.severity, f.brief) for f in result.findings]

        assert seen(matrix.runs[1].result) == seen(lone)
        assert matrix.runs[1].result.metadata.resolved_config != matrix.runs[0].result.metadata.resolved_config

    def test_a_range_counts_up_by_its_step_and_includes_its_end(self) -> None:
        result = run_matrix({"checks.image-outliers.warning": {"from": 1.0, "to": 2.0, "step": 0.5}})

        assert [run.values["checks.image-outliers.warning"] for run in result.runs] == [1.0, 1.5, 2.0]

    def test_one_grid_crosses_its_keys_with_the_last_key_changing_fastest(self) -> None:
        result = run_matrix(
            {
                "outliers.outlier_threshold": [["zscore", 2.0], ["zscore", 3.0]],
                "checks.image-outliers.warning": [1.0, 2.0, 3.0],
            }
        )

        assert len(result.runs) == 6
        assert [
            (run.values["outliers.outlier_threshold"][1], run.values["checks.image-outliers.warning"])
            for run in result.runs
        ] == [
            (2.0, 1.0),
            (2.0, 2.0),
            (2.0, 3.0),
            (3.0, 1.0),
            (3.0, 2.0),
            (3.0, 3.0),
        ]
        assert result.runs[0].label == "outliers.outlier_threshold=[zscore, 2.0], checks.image-outliers.warning=1.0"

    def test_several_grids_run_in_turn_and_a_grid_need_not_set_every_key(self) -> None:
        result = run_matrix(
            [
                {"outliers.outlier_threshold": [["zscore", 2.0], ["zscore", 3.0]]},
                {"checks.image-outliers.warning": [1.0]},
            ]
        )

        assert [run.number for run in result.runs] == [1, 2, 3]
        assert [sorted(run.values) for run in result.runs] == [
            ["outliers.outlier_threshold"],
            ["outliers.outlier_threshold"],
            ["checks.image-outliers.warning"],
        ]
        assert result.keys == ["outliers.outlier_threshold", "checks.image-outliers.warning"]

    def test_the_tasks_sources_and_extractor_can_be_varied(self) -> None:
        by_source = run_matrix({"sources": ["src", "other"]}, datasets={"src": planted(), "other": Images(40, seed=3)})
        by_extractor = run_matrix(
            {"extractor": ["flat", "wide"]},
            extractor=True,
            extra={"extractors": [FlattenExtractorConfig(name="wide", batch_size=4)]},
        )

        assert [run.result.metadata.source_descriptions for run in by_source.runs] == [
            ["src (src_data)"],
            ["other (other_data)"],
        ]
        assert [run.result.metadata.model_id for run in by_extractor.runs] == ["flat (flatten)", "wide (flatten)"]

    def test_a_key_reaches_a_list_item_by_its_name(self) -> None:
        workflow = {
            "name": "cleaning",
            "type": "shift",
            "detectors": [{"name": "knn", "type": "ood-kneighbors", "k": 3}],
        }
        data = {"reference": Images(60, seed=0), "test": Images(60, seed=2, tint=True)}

        result = run_matrix(
            {"detectors.knn.k": [3, 5]},
            workflows=[workflow],
            datasets=data,
            task={"sources": ["reference", "test"], "extractor": "flat"},
            extractor=True,
        )

        assert [run.label for run in result.runs] == ["detectors.knn.k=3", "detectors.knn.k=5"]
        assert [run.result.steps["knn"].elements["test"].output.is_ood.all() for run in result.runs] == [True, True]  # type: ignore[union-attr]

    def test_an_evaluator_task_has_a_matrix_too(self) -> None:
        evaluator = {"name": "outl", "type": "outliers", "flags": ["pixel"], "outlier_threshold": "zscore"}
        config = pipeline(
            evaluators=[evaluator],
            tasks=[
                {
                    "name": "t",
                    "evaluator": "outl",
                    "sources": ["src"],
                    "matrix": {"outlier_threshold": [["zscore", 2.0], ["zscore", 3.0]]},
                }
            ],
            datasets={"src": planted()},
        )

        result = run_tasks(config)["t"]

        assert isinstance(result, MatrixResult)
        assert [run.status for run in result.runs] == ["ok", "ok"]  # an evaluator judges nothing, so it has no findings
        assert result.type == "outliers"
        assert "an evaluator has no findings to list" in result.report()


class TestMatrixReport:
    def test_the_report_opens_with_a_table_of_every_runs_findings(self) -> None:
        result = run_matrix({"checks.image-outliers.warning": [0.5, 5.0]})

        report = result.report()

        assert "matrix of 2 runs" in report
        assert "Health: 3 warnings [!!] across 2 runs" in report
        header = next(line for line in report.splitlines() if line.strip().startswith("#"))
        for column in (
            "checks.image-outliers.warning",
            "Health",
            "Image Outliers",
            "Class Outliers",
            "Image Duplicates",
        ):
            assert column in header
        assert result.health == {"status": "warning", "warnings": 3, "failed_runs": []}
        assert result.warning_count == 3

    def test_the_json_holds_each_run_under_runs(self) -> None:
        result = run_matrix({"checks.image-outliers.warning": [0.5, 5.0]})

        payload = result.to_dict()

        assert (payload["kind"], payload["type"], payload["keys"]) == (
            "matrix",
            "quality",
            ["checks.image-outliers.warning"],
        )
        runs = payload["runs"]
        assert [(run["number"], run["label"]) for run in runs] == [  # type: ignore[index]
            (1, "checks.image-outliers.warning=0.5"),
            (2, "checks.image-outliers.warning=5.0"),
        ]
        assert runs[0]["result"]["kind"] == "workflow"  # type: ignore[index]
        assert "errors" not in payload


class TestMatrixRuns:
    def test_a_failed_run_is_kept_and_fails_the_matrix_but_later_runs_and_tasks_still_run(self) -> None:
        workflow = {**CLEANING, "outliers": {**CLEANING["outliers"], "cluster_threshold": 2.0, "n_clusters": 2}}
        workflow["outliers"]["cluster_algorithm"] = "kmeans"
        config = pipeline(
            workflows=[workflow],
            tasks=[
                matrix_task({"outliers.n_clusters": [2, 1000, 3]}, extractor="flat"),
                {"name": "after", "workflow": "cleaning", "sources": ["src"], "extractor": "flat"},
            ],
            datasets={"src": planted()},
            extractor=True,
            extra={"seed": 0},
        )

        results = run_tasks(config)

        matrix = results["t"]
        assert isinstance(matrix, MatrixResult)
        assert not matrix.success
        assert [run.status for run in matrix.runs] == ["warning", "failed", "warning"]  # run 3 ran after run 2 failed
        assert matrix.health["status"] == "failed"
        assert matrix.health["failed_runs"] == [2]
        expected = (
            "run 2 (outliers.n_clusters=1000): outliers: ValueError: n_expected_clusters=1000 should be less than "
            "dataset size (60)"
        )
        assert matrix.errors == [expected]
        assert "Run 2 failed" in matrix.report()
        assert matrix.to_dict()["errors"] == matrix.errors
        assert results["after"].success  # the next task is not stopped

    def test_every_run_reads_one_draw_of_a_source_whose_view_shuffles(self) -> None:
        shuffled = ViewConfig(
            name="mixed",
            operations=[ViewOperation(type="Shuffle", params={}), ViewOperation(type="Limit", params={"size": 30})],
        )

        result = run_matrix(
            {"checks.image-outliers.warning": [1.0, 2.0, 3.0]},
            extra={"views": [shuffled], "sources": [SourceConfig(name="src", dataset="src_data", view="mixed")]},
        )

        digests = {run.result.metadata.lineage[0].digest for run in result.runs}  # type: ignore[union-attr]
        assert len(digests) == 1  # a shuffle that drew afresh in each run would give three
        assert result.runs[0].result.metadata.lineage[0].items == 30  # type: ignore[union-attr]

    def test_an_export_step_writes_each_run_under_its_own_directory(self, tmp_path: Path) -> None:
        workflows = [
            CLEANING,
            {
                "name": "tidy",
                "inputs": ["data"],
                "steps": [
                    {"name": "cleaning", "workflow": "cleaning", "input": "data"},
                    {"name": "dataset", "transform": "export", "input": "cleaning.clean", "format": "coco"},
                ],
            },
        ]

        result = run_matrix(
            {"workflows.cleaning.outliers.outlier_threshold": [["zscore", 2.0], ["zscore", 3.5]]},
            workflows=workflows,
            datasets={"src": Detections(20)},
            task={"name": "t", "workflow": "tidy"},
            output_dir=tmp_path,
        )

        assert result.success, result.errors
        written = sorted(path.name for path in (tmp_path / "datasets" / "t.dataset").iterdir())
        assert written == ["run-1", "run-2"]
        assert all((tmp_path / "datasets" / "t.dataset" / run / "annotations").is_dir() for run in written)


class TestMatrixConfiguration:
    @pytest.mark.parametrize(
        ("matrix", "message"),
        [
            ({"outliers.flags": [["pixel"], ["bogus"]]}, "run 2 (outliers.flags=[bogus]): Input should be"),
            (
                {"checks.image-outliers.warning": [1.0, 1.0]},
                "runs 1 and 2 (checks.image-outliers.warning=1.0) set the same settings",
            ),
            ({"outliers.n_clusters": 3}, "takes a list of values, such as `[3]`, or a range `{from, to, step}`; got 3"),
            ({"nonsense": [1]}, "quality has no setting `nonsense`"),
            ({"type": ["shift"]}, "a matrix varies settings, not identities"),
            ({"checks": [{}], "checks.image-outliers.warning": [1.0]}, "vary them in separate grids"),
            ({"sources": ["nowhere"]}, "nowhere"),
        ],
        ids=[
            "invalid-value",
            "duplicate-runs",
            "bare-value",
            "unknown-setting",
            "identity-key",
            "overlapping-keys",
            "unknown-source",
        ],
    )
    def test_a_matrix_that_cannot_run_is_refused_when_the_config_loads(self, matrix: Any, message: str) -> None:
        with pytest.raises(ValidationError) as caught:
            pipeline(workflows=[CLEANING], tasks=[matrix_task(matrix)], datasets={"src": planted()})

        assert message in str(caught.value)

    def test_a_matrix_in_a_yaml_file_runs_from_load_config(self, tmp_path: Path) -> None:
        write_image_folder(tmp_path / "imgs", n_per_class=10, n_classes=2)
        text = """
datasets:
  - {name: ds, format: image_folder, path: imgs, infer_labels: true}
sources:
  - {name: train, dataset: ds}
workflows:
  - name: cleaning
    type: quality
    outliers: {flags: [pixel], outlier_threshold: zscore}
tasks:
  - name: tune
    workflow: cleaning
    sources: train
    matrix:
      checks.image-outliers.warning: {from: 1, to: 5, step: 2}
      outliers.outlier_threshold: [zscore, [modzscore, 3.5]]
"""
        (tmp_path / "pipeline.yaml").write_text(text)
        assert yaml.safe_load(text)["tasks"][0]["matrix"]["checks.image-outliers.warning"] == {
            "from": 1,
            "to": 5,
            "step": 2,
        }

        result = run_tasks(load_config(tmp_path / "pipeline.yaml"), data_dir=tmp_path)["tune"]

        assert isinstance(result, MatrixResult)
        assert result.success, result.errors
        assert len(result.runs) == 6  # 1, 3 and 5, each with two thresholds
        assert [run.values["checks.image-outliers.warning"] for run in result.runs] == [1, 1, 3, 3, 5, 5]

    def test_the_parameter_sweep_workflow_type_no_longer_exists(self) -> None:
        with pytest.raises(ValidationError, match="Unknown workflow: 'parameter-sweep'"):
            pipeline(
                workflows=[{"name": "sweep", "type": "parameter-sweep"}],
                tasks=[{"name": "t", "workflow": "sweep", "sources": ["src"]}],
                datasets={"src": planted()},
            )
