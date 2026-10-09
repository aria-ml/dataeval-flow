"""TC-19-1 and TC-19-6 — running a custom workflow: inputs, steps in order, the result, and what happens on failure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dataeval_flow import run_tasks
from dataeval_flow.steps import ChainResult
from verification.fixtures import plant_duplicate_and_outlier, write_image_folder
from verification.functional.chains._toys import Images, yaml_pipeline
from verification.helpers import run_cli

pytestmark = pytest.mark.required

_CLEAN_AND_CHECK = """
evaluators:
  - {name: dupes, type: duplicates}
  - {name: labels, type: label-health}
workflows:
  - name: judged
    inputs: [data]
    steps:
      - {name: dupes, evaluator: dupes, input: data}
      - {name: labels, evaluator: labels, input: data}
      - {name: duplicates, check: image-duplicates, input: dupes}
      - {name: imbalance, check: class-imbalance, input: labels}
      - {name: clean, transform: remove, input: data, plans: {dupes: {keep: first}}}
      - {name: after, evaluator: dupes, input: clean}
tasks:
  - {name: t, workflow: judged, sources: [src]}
"""


def _run(text: str, datasets: dict, **kwargs) -> ChainResult:
    result = run_tasks(yaml_pipeline(text, datasets, **kwargs))["t"]
    assert isinstance(result, ChainResult)
    return result


class TestRunningAChain:
    def test_a_workflow_with_inputs_and_steps_and_no_type_runs_as_a_task(self) -> None:
        result = _run(_CLEAN_AND_CHECK, {"src": Images()})
        assert result.success, result.errors
        assert result.kind == "workflow"
        assert result.type == "judged"
        assert result.metadata.workflow == "judged"

    def test_every_step_runs_in_the_order_written_and_is_recorded(self) -> None:
        result = _run(_CLEAN_AND_CHECK, {"src": Images()})
        assert list(result.steps) == ["dupes", "labels", "duplicates", "imbalance", "clean", "after"]
        kinds = {name: (record.kind, record.type, record.status) for name, record in result.steps.items()}
        assert kinds["dupes"] == ("evaluator", "duplicates", "ok")
        assert kinds["duplicates"] == ("check", "image-duplicates", "ok")
        assert kinds["clean"] == ("transform", "remove", "ok")
        assert result.steps["clean"].inputs == ["data", "dupes"]

    def test_a_step_reads_what_an_earlier_step_made(self) -> None:
        result = _run(_CLEAN_AND_CHECK, {"src": Images()})
        # The planted exact duplicate is removed, so the second duplicates run finds none.
        assert len(result.steps["clean"].output) == 11
        assert result.steps["clean"].details["removed"]["items"] == 1
        assert result.steps["after"].output.data().is_empty()
        assert len(result.steps["dupes"].output.data()) == 1

    def test_the_sources_of_a_task_bind_to_the_inputs_in_order(self) -> None:
        text = """
evaluators:
  - {name: lh, type: label-health}
workflows:
  - name: two
    inputs: [first, second]
    steps:
      - {name: on_first, evaluator: lh, input: first}
      - {name: on_second, evaluator: lh, input: second}
tasks:
  - {name: t, workflow: two, sources: [big, small]}
"""
        result = _run(text, {"big": Images(20), "small": Images(8, seed=1)})
        assert result.steps["on_first"].output.data()["item_count"] == 20
        assert result.steps["on_second"].output.data()["item_count"] == 8

    def test_the_result_json_lists_each_step_the_findings_and_the_health(self) -> None:
        payload = _run(_CLEAN_AND_CHECK, {"src": Images()}).to_dict()
        assert payload["kind"] == "workflow"
        assert set(payload["steps"]) == {"dupes", "labels", "duplicates", "imbalance", "clean", "after"}  # type: ignore[arg-type]
        assert payload["health"] == {"status": "warning", "warnings": 1, "findings": 2, "failed_steps": []}
        assert [finding["step"] for finding in payload["findings"]] == ["duplicates", "imbalance"]  # type: ignore[union-attr]
        # A Dataset a step made is written as its size and digest, never as data.
        assert set(payload["steps"]["clean"]["output"]) == {"items", "digest"}  # type: ignore[index]

    def test_the_lineage_records_each_dataset_with_its_step_inputs_and_size(self) -> None:
        lineage = {record.name: record for record in _run(_CLEAN_AND_CHECK, {"src": Images()}).metadata.lineage}
        assert set(lineage) == {"data", "clean"}
        assert (lineage["data"].source, lineage["data"].items) == ("src", 12)
        assert (lineage["clean"].type, lineage["clean"].inputs, lineage["clean"].items) == ("remove", ["data"], 11)
        assert len(lineage["clean"].digest) == 12

    def test_the_same_run_gives_the_same_lineage_digests(self) -> None:
        first = _run(_CLEAN_AND_CHECK, {"src": Images()}).metadata.lineage
        second = _run(_CLEAN_AND_CHECK, {"src": Images()}).metadata.lineage
        assert [(r.name, r.digest) for r in first] == [(r.name, r.digest) for r in second]

    def test_the_report_gives_each_step_without_a_finding_a_section_and_a_steps_table(self) -> None:
        report = _run(_CLEAN_AND_CHECK, {"src": Images()}).report()
        assert "Workflow:  judged (custom workflow)" in report
        assert "Kept 11 of 12 images. Removed 1 image: 1 named by `dupes`." in report
        assert "REMOVE · CLEAN" in report
        assert "STEPS" in report

    def test_one_workflow_runs_on_different_sources_from_different_tasks(self) -> None:
        text = _CLEAN_AND_CHECK.replace(
            "tasks:\n  - {name: t, workflow: judged, sources: [src]}",
            "tasks:\n  - {name: t, workflow: judged, sources: [src]}\n"
            "  - {name: t2, workflow: judged, sources: [other]}",
        )
        results = run_tasks(yaml_pipeline(text, {"src": Images(), "other": Images(10, seed=3, planted=False)}))
        assert results["t"].steps["clean"].details["removed"]["items"] == 1  # type: ignore[union-attr]
        assert results["t2"].steps["clean"].details["removed"]["items"] == 0  # type: ignore[union-attr]


_FAILING = """
evaluators:
  - {name: dupes, type: duplicates}
  - {name: labels, type: label-health}
workflows:
  - name: w
    inputs: [a, b]
    steps:
      - {name: ok_first, evaluator: labels, input: a}
      - {name: bad, transform: split, input: a, test_frac: 0.2, split_on: [nope]%s}
      - {name: dependent, evaluator: dupes, input: bad.train}
      - {name: dependent_of_dependent, evaluator: labels, input: bad.test}
      - {name: independent, evaluator: dupes, input: b}
tasks:
  - {name: t, workflow: w, sources: [a, b]}
"""


class TestFailures:
    @staticmethod
    def _run(optional: str = "") -> ChainResult:
        datasets = {"a": Images(12, planted=False), "b": Images(12, seed=1, planted=False)}
        return _run(_FAILING % optional, datasets, extra={"seed": 0})

    def test_a_failed_step_fails_the_task_and_says_why(self) -> None:
        result = self._run()
        assert not result.success
        assert result.steps["bad"].status == "failed"
        assert "split_on" in result.steps["bad"].errors[0]
        assert result.errors[0].startswith("bad: ValueError")
        assert result.health["status"] == "failed"
        assert result.health["failed_steps"] == ["bad"]

    def test_a_step_that_reads_a_failed_step_is_skipped_with_the_reason(self) -> None:
        result = self._run()
        for name in ("dependent", "dependent_of_dependent"):
            assert result.steps[name].status == "skipped"
            assert result.steps[name].reason.startswith("needs `bad.")
            assert "which failed" in result.steps[name].reason

    def test_steps_that_do_not_depend_on_the_failure_still_run(self) -> None:
        result = self._run()
        assert result.steps["ok_first"].status == "ok"
        assert result.steps["independent"].status == "ok"
        assert result.steps["independent"].output is not None

    def test_the_steps_that_ran_stay_in_the_result_but_the_output_is_not_available(self) -> None:
        result = self._run()
        assert list(result.steps) == ["ok_first", "bad", "dependent", "dependent_of_dependent", "independent"]
        with pytest.raises(RuntimeError, match="did not complete"):
            result.output  # noqa: B018

    def test_an_optional_step_that_fails_is_a_skip_and_the_task_still_succeeds(self) -> None:
        result = self._run(", optional: true")
        assert result.success, result.errors
        assert result.health["status"] == "ok"
        assert result.steps["bad"].status == "skipped"
        assert "split_on" in result.steps["bad"].reason
        # The steps that read it are still skipped.
        assert result.steps["dependent"].status == "skipped"
        assert result.steps["independent"].status == "ok"

    def test_the_failure_is_in_the_report_and_the_json(self) -> None:
        result = self._run()
        assert "Health: failed" in result.report()
        assert "step `bad` failed" in result.report()
        payload = result.to_dict()
        assert payload["health"]["status"] == "failed"  # type: ignore[index]
        assert payload["errors"] == result.errors

    def test_a_task_whose_steps_fail_does_not_stop_the_next_task(self) -> None:
        text = _FAILING % ""
        text = text.replace(
            "tasks:\n  - {name: t, workflow: w, sources: [a, b]}",
            "tasks:\n  - {name: t, workflow: w, sources: [a, b]}\n  - {name: ok, evaluator: labels, sources: [a]}",
        )
        datasets = {"a": Images(12, planted=False), "b": Images(12, seed=1, planted=False)}
        results = run_tasks(yaml_pipeline(text, datasets, extra={"seed": 0}))
        assert list(results) == ["t", "ok"]
        assert not results["t"].success
        assert results["ok"].success


class TestChainOnTheCommandLine:
    @staticmethod
    def _project(root: Path, *, failing: bool = False) -> Path:
        write_image_folder(root / "imgs", n_per_class=10, n_classes=2)
        plant_duplicate_and_outlier(root / "imgs")
        steps = [
            {"name": "dupes", "evaluator": "dupes", "input": "data"},
            {"name": "clean", "transform": "remove", "input": "data", "plans": {"dupes": {"keep": "first"}}},
            {"name": "found", "check": "image-duplicates", "input": "dupes"},
        ]
        if failing:
            steps.append(
                {"name": "bad", "transform": "split", "input": "clean", "test_frac": 0.2, "split_on": ["nope"]}
            )
        config = {
            "datasets": [{"name": "ds", "format": "image_folder", "path": "imgs", "infer_labels": True}],
            "sources": [{"name": "main", "dataset": "ds"}],
            "evaluators": [{"name": "dupes", "type": "duplicates"}],
            "workflows": [{"name": "judged", "inputs": ["data"], "steps": steps}],
            "tasks": [{"name": "prep", "workflow": "judged", "sources": ["main"]}],
        }
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        return path

    def test_a_chain_run_from_the_command_line_writes_its_steps_lineage_and_findings(self, tmp_path: Path) -> None:
        config = self._project(tmp_path)
        proc = run_cli("-c", str(config), "-d", str(tmp_path), "-o", str(tmp_path / "out"))
        assert proc.returncode == 0, proc.stdout + proc.stderr
        entry = json.loads((tmp_path / "out" / "results" / "result.json").read_text())["prep"]
        assert entry["kind"] == "workflow"
        assert list(entry["steps"]) == ["dupes", "clean", "found"]
        assert entry["steps"]["clean"]["details"]["removed"]["items"] == 1
        assert [item["name"] for item in entry["metadata"]["lineage"]] == ["data", "clean"]
        assert [finding["step"] for finding in entry["findings"]] == ["found"]
        assert entry["health"]["status"] == "warning"
        assert "judged (custom workflow)" in (tmp_path / "out" / "results" / "result.txt").read_text()

    def test_a_chain_with_a_failed_step_exits_1_and_the_steps_that_ran_are_still_written(self, tmp_path: Path) -> None:
        config = self._project(tmp_path, failing=True)
        proc = run_cli("-c", str(config), "-d", str(tmp_path), "-o", str(tmp_path / "out"))
        assert proc.returncode == 1, proc.stdout + proc.stderr
        entry = json.loads((tmp_path / "out" / "results" / "result.json").read_text())["prep"]
        assert entry["health"]["status"] == "failed"
        assert entry["health"]["failed_steps"] == ["bad"]
        assert entry["steps"]["clean"]["status"] == "ok"
