"""Tests for _resolve_config in runner.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from dataeval_flow._runner import _resolve_config
from dataeval_flow.config import PipelineConfig

pytestmark = pytest.mark.required


@pytest.fixture
def dummy_config() -> PipelineConfig:
    return PipelineConfig()


_LOADER = "dataeval_flow.config._loader"


class TestResolveConfig:
    def test_explicit_absolute_file(self, tmp_path: Path, dummy_config: PipelineConfig):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("tasks: []")

        with patch(f"{_LOADER}.load_config", return_value=dummy_config) as mock_load:
            result = _resolve_config(cfg_file, tmp_path)

        mock_load.assert_called_once_with(cfg_file)
        assert result is dummy_config

    def test_explicit_relative_file(self, tmp_path: Path, dummy_config: PipelineConfig):
        cfg_file = tmp_path / "sub" / "config.yaml"
        cfg_file.parent.mkdir()
        cfg_file.write_text("tasks: []")

        with patch(f"{_LOADER}.load_config", return_value=dummy_config) as mock_load:
            result = _resolve_config(Path("sub/config.yaml"), tmp_path)

        mock_load.assert_called_once_with(tmp_path / "sub" / "config.yaml")
        assert result is dummy_config

    def test_explicit_directory(self, tmp_path: Path, dummy_config: PipelineConfig):
        cfg_dir = tmp_path / "conf"
        cfg_dir.mkdir()

        with patch(f"{_LOADER}.load_config", return_value=dummy_config) as mock_load:
            result = _resolve_config(cfg_dir, tmp_path)

        mock_load.assert_called_once_with(cfg_dir)
        assert result is dummy_config

    def test_none_config_uses_data_dir(self, tmp_path: Path, dummy_config: PipelineConfig):
        with patch(f"{_LOADER}.load_config", return_value=dummy_config) as mock_load:
            result = _resolve_config(None, tmp_path)

        mock_load.assert_called_once_with(tmp_path)
        assert result is dummy_config

    def test_missing_path_raises(self, tmp_path: Path):
        missing = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="Config path not found"):
            _resolve_config(missing, tmp_path)

    def test_string_config_arg(self, tmp_path: Path, dummy_config: PipelineConfig):
        cfg_file = tmp_path / "my_config.yaml"
        cfg_file.write_text("tasks: []")

        with patch(f"{_LOADER}.load_config", return_value=dummy_config) as mock_load:
            result = _resolve_config("my_config.yaml", tmp_path)

        mock_load.assert_called_once_with(tmp_path / "my_config.yaml")
        assert result is dummy_config


class TestWriteEncodingDescriptor:
    """A run writes the descriptor its results were computed under, beside them."""

    @staticmethod
    def _record(digest: str, edges: list) -> dict:
        return {
            "encoding_digest": digest,
            "descriptor_version": 1,
            "factors": {
                "temp_c": {"encoding": {"kind": "bins", "edges": edges, "provenance": "edges", "method": None}}
            },
        }

    def test_writes_one_when_the_tasks_agree(self, tmp_path: Path):
        from dataeval_flow._runner import _write_encoding_descriptor

        record = self._record("abc", ["-inf", 0.0, "inf"])
        _write_encoding_descriptor({"a": record, "b": record}, tmp_path)

        import json

        written = json.loads((tmp_path / "encoding.json").read_text())
        assert written["factors"]["temp_c"]["edges"] == ["-inf", 0.0, "inf"]
        assert written["version"] == 1

    def test_writes_nothing_when_the_tasks_disagree(self, tmp_path: Path, caplog):
        """Writing one of them would hand somebody a policy nobody chose."""
        from dataeval_flow._runner import _write_encoding_descriptor

        _write_encoding_descriptor(
            {"a": self._record("abc", ["-inf", 0.0, "inf"]), "b": self._record("def", ["-inf", 5.0, "inf"])},
            tmp_path,
        )

        assert not (tmp_path / "encoding.json").exists()
        assert "dataeval-flow encoding" in caplog.text

    def test_writes_nothing_when_no_task_built_metadata(self, tmp_path: Path):
        from dataeval_flow._runner import _write_encoding_descriptor

        _write_encoding_descriptor({}, tmp_path)
        assert not (tmp_path / "encoding.json").exists()

    def test_a_record_with_no_encodings_is_skipped_quietly(self, tmp_path: Path):
        """Never fatal: result.json already carries every record it is built from."""
        from dataeval_flow._runner import _write_encoding_descriptor

        _write_encoding_descriptor({"a": {"factors": {}}}, tmp_path)
        assert not (tmp_path / "encoding.json").exists()


# ---------------------------------------------------------------------------
# run() — task selection, result pairing, and the health gate
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, *, disable: str | None = None, extra: str = "") -> Path:
    """A two-task config, optionally with one task disabled, and *extra* YAML appended."""
    lines = [
        "datasets:",
        "  - name: ds",
        "    format: huggingface",
        "    path: ./d",
        "    task: image_classification",
        "sources:",
        "  - name: src",
        "    dataset: ds",
        "workflows:",
        "  - name: wf",
        "    type: quality",
        "    outliers: {flags: [dimension], outlier_threshold: modzscore}",
        "tasks:",
    ]
    for name in ("task_a", "task_b"):
        lines += [f"  - name: {name}", "    workflow: wf", "    sources: src"]
        if name == disable:
            lines.append("    enabled: false")
    path = tmp_path / "config.yaml"
    path.write_text("\n".join(lines) + "\n" + extra)
    return path


def _fake_result(*, warnings: int = 0):
    """A stand-in workflow result with a controllable warning count."""
    from unittest.mock import MagicMock

    from dataeval_flow._blocks import Section
    from dataeval_flow.steps import ChainResult

    result = MagicMock(spec=ChainResult)
    result.success = True
    result.report.return_value = "report"
    result._html_reports.return_value = [(Section(title="report"), [])]
    result.to_dict.return_value = {"metadata": {}}
    result.metadata = MagicMock(metadata_binning=None)
    result.warning_count = warnings
    result.assets = []
    result.steps = {}
    result.failed_steps = []
    result.findings = []
    result.verdict = None
    result.health = {"status": "warning" if warnings else "ok"}
    return result


class TestReportImages:
    """The runner's switch reaches every task it runs."""

    @pytest.mark.parametrize("report_images", [True, False])
    def test_the_switch_reaches_each_task(self, tmp_path: Path, report_images: bool):
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_fake_result()) as run_one:
            run(config, None, data_dir=tmp_path, report_images=report_images)
        assert {call.args[1].result.max_images != 0 for call in run_one.call_args_list} == {report_images}


class TestOutputDir:
    """The runner's output directory reaches every task it runs, for the export steps a chain holds."""

    def test_the_output_dir_reaches_each_task(self, tmp_path: Path):
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_fake_result()) as run_one:
            run(config, tmp_path / "out", data_dir=tmp_path)
        assert {call.kwargs["output_dir"] for call in run_one.call_args_list} == {tmp_path / "out"}


class TestRunTaskPairing:
    """A disabled task must not misalign results against the tasks that produced them."""

    def test_disabled_task_does_not_break_the_run(self, tmp_path: Path):
        """Regression: run() paired results against every task, not the executed ones."""
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path, disable="task_b")
        with patch.object(orch, "_run_single_task", return_value=_fake_result()):
            assert run(config, tmp_path / "out", data_dir=tmp_path) == 0

        import json

        merged = json.loads((tmp_path / "out" / "results" / "result.json").read_text())
        assert list(merged) == ["task_a"]

    def test_results_are_keyed_by_the_task_that_produced_them(self, tmp_path: Path):
        """A result names its workflow type, so keying has to come from the selection."""
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_fake_result()):
            assert run(config, tmp_path / "out", data_dir=tmp_path) == 0

        import json

        merged = json.loads((tmp_path / "out" / "results" / "result.json").read_text())
        assert list(merged) == ["task_a", "task_b"]


class TestRunTaskSelection:
    def test_names_select_a_subset(self, tmp_path: Path):
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_fake_result()):
            assert run(config, tmp_path / "out", data_dir=tmp_path, tasks=["task_b"]) == 0

        import json

        merged = json.loads((tmp_path / "out" / "results" / "result.json").read_text())
        assert list(merged) == ["task_b"]

    def test_naming_a_disabled_task_runs_it(self, tmp_path: Path):
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path, disable="task_b")
        with patch.object(orch, "_run_single_task", return_value=_fake_result()):
            assert run(config, tmp_path / "out", data_dir=tmp_path, tasks="task_b") == 0

        import json

        merged = json.loads((tmp_path / "out" / "results" / "result.json").read_text())
        assert list(merged) == ["task_b"]

    def test_a_task_named_twice_runs_and_counts_once(self, tmp_path: Path, caplog):
        import logging

        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path)
        with (
            patch.object(orch, "_run_single_task", return_value=_fake_result()) as run_single,
            caplog.at_level(logging.INFO),
        ):
            assert run(config, tmp_path / "out", data_dir=tmp_path, tasks=["task_b", "task_b"]) == 0

        assert run_single.call_count == 1
        assert "Done. 1/1 succeeded." in caplog.text

    def test_unknown_task_name_raises(self, tmp_path: Path):
        from dataeval_flow._runner import run

        config = _write_config(tmp_path)
        with pytest.raises(ValueError, match="Unknown task: 'nope'"):
            run(config, tmp_path / "out", data_dir=tmp_path, tasks="nope")


class TestFailOnWarning:
    def test_warnings_are_not_fatal_by_default(self, tmp_path: Path, caplog):
        """A warning is a prompt to look, so it is reported without failing the run."""
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_fake_result(warnings=2)):
            assert run(config, tmp_path / "out", data_dir=tmp_path) == 0

        assert "Health warnings raised by: task_a, task_b" in caplog.text

    def test_flag_makes_warnings_fatal(self, tmp_path: Path):
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_fake_result(warnings=1)):
            assert run(config, tmp_path / "out", data_dir=tmp_path, fail_on_warning=True) == 3

    def test_flag_is_a_no_op_without_warnings(self, tmp_path: Path):
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_fake_result(warnings=0)):
            assert run(config, tmp_path / "out", data_dir=tmp_path, fail_on_warning=True) == 0

    def test_results_are_still_written_when_warnings_are_fatal(self, tmp_path: Path):
        """The gate decides the exit code, not whether the run's artifacts survive."""
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_fake_result(warnings=1)):
            run(config, tmp_path / "out", data_dir=tmp_path, fail_on_warning=True)

        assert (tmp_path / "out" / "results" / "result.json").exists()
        assert (tmp_path / "out" / "results" / "result.txt").exists()
        assert (tmp_path / "out" / "results" / "result.html").exists()


class TestNothingSucceeded:
    def test_a_run_where_every_task_fails_writes_no_results_and_exits_1(self, tmp_path: Path):
        """With no report to show, no result file is written, json, text or html, and the run fails."""
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_failed_result()):
            assert run(config, tmp_path / "out", data_dir=tmp_path) == 1

        assert not (tmp_path / "out" / "results").exists()

    def test_it_says_which_files_it_left_unwritten(self, tmp_path: Path, caplog):
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path, extra="result:\n  fail_on: never\n")
        with patch.object(orch, "_run_single_task", side_effect=[_failed_result(), _failed_result()]):
            assert run(config, tmp_path / "out", data_dir=tmp_path) == 0
        assert "No task succeeded, so no file was written for: json, text, html." in caplog.text


def _with_exports(path: Path, source: str) -> Path:
    """Append an `exports:` block naming *source* to an existing config file."""
    path.write_text(path.read_text() + f"exports:\n  - name: dataset\n    source: {source}\n")
    return path


class TestRunnerExports:
    """The runner writes declared exports beside the results, and answers for a failure."""

    def test_exports_are_written_beside_the_results(self, tmp_path: Path):
        import dataeval_flow._export as export_mod
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _with_exports(_write_config(tmp_path), "src")
        out = tmp_path / "out"
        with (
            patch.object(orch, "_run_single_task", return_value=_fake_result()),
            patch.object(export_mod, "write_exports", return_value=0) as write_exports,
        ):
            assert run(config, out, data_dir=tmp_path) == 0

        write_exports.assert_called_once()
        assert write_exports.call_args.args[1] == out
        assert write_exports.call_args.kwargs["data_dir"] == tmp_path

    def test_nothing_is_exported_without_an_output_directory(self, tmp_path: Path):
        """No output directory means no file artifacts, exports included."""
        import dataeval_flow._export as export_mod
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _with_exports(_write_config(tmp_path), "src")
        with (
            patch.object(orch, "_run_single_task", return_value=_fake_result()),
            patch.object(export_mod, "write_exports") as write_exports,
        ):
            assert run(config, None, data_dir=tmp_path) == 0

        write_exports.assert_not_called()

    def test_a_failing_export_makes_the_run_non_zero(self, tmp_path: Path, caplog):
        """An export the config asked for and did not get is a failure the caller must see."""
        import logging

        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _with_exports(_write_config(tmp_path), "absent")
        with patch.object(orch, "_run_single_task", return_value=_fake_result()), caplog.at_level(logging.ERROR):
            assert run(config, tmp_path / "out", data_dir=tmp_path) == 1

        assert "export" in caplog.text.lower()
        # The tasks still ran and their results were written.
        assert (tmp_path / "out" / "results" / "result.json").exists()

    def test_an_export_is_written_when_the_config_declares_no_tasks(self, tmp_path: Path):
        """An export names a source, so a config that runs nothing still writes its dataset."""
        import dataeval_flow._export as export_mod
        from dataeval_flow._runner import run

        path = tmp_path / "config.yaml"
        path.write_text("datasets: []\nsources: []\n")
        _with_exports(path, "src")
        with patch.object(export_mod, "write_exports", return_value=0) as write_exports:
            assert run(path, tmp_path / "out", data_dir=tmp_path) == 0

        write_exports.assert_called_once()


def _fake_evaluator_result(*, success: bool = True):
    """A real evaluator result: evaluators never warn, so there is nothing to stub."""
    from dataeval_flow.evaluators import EvaluatorResult
    from dataeval_flow.evaluators._result import EvaluatorMetadata

    return EvaluatorResult(
        type="duplicates",
        success=success,
        output=object() if success else None,
        serialized={"shape": "table", "columns": [], "rows": []} if success else None,
        metadata=EvaluatorMetadata(evaluator="duplicates"),
        errors=[] if success else ["RuntimeError: boom"],
    )


class TestEvaluatorResultsCarryNoVerdict:
    def test_fail_on_warning_passes_a_run_of_evaluators(self, tmp_path: Path):
        """Review Focus 4: evaluators make determinations, never warnings."""
        import json

        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_fake_evaluator_result()):
            assert run(config, tmp_path / "out", data_dir=tmp_path, fail_on_warning=True) == 0

        merged = json.loads((tmp_path / "out" / "results" / "result.json").read_text())
        assert {entry["kind"] for entry in merged.values()} == {"evaluator"}
        assert not any("health" in entry for entry in merged.values())

    def test_a_failed_evaluator_still_fails_the_run(self, tmp_path: Path):
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path)
        with patch.object(orch, "_run_single_task", return_value=_fake_evaluator_result(success=False)):
            assert run(config, tmp_path / "out", data_dir=tmp_path) == 1


class TestResultFiles:
    """The pipeline's ``result:`` block chooses the files ``--output`` writes, their detail and their width."""

    @staticmethod
    def _run(tmp_path: Path, extra: str, **kwargs: object) -> tuple[Path, object]:
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        result = _fake_result()
        config = _write_config(tmp_path, extra=extra)
        with patch.object(orch, "_run_single_task", return_value=result):
            assert run(config, tmp_path / "out", data_dir=tmp_path, **kwargs) == 0  # type: ignore[arg-type]
        return tmp_path / "out" / "results", result

    def test_all_three_formats_under_one_name_by_default(self, tmp_path: Path):
        results, _ = self._run(tmp_path, "")
        assert sorted(path.name for path in results.glob("result*")) == ["result.html", "result.json", "result.txt"]

    def test_the_name_and_formats_choose_the_files(self, tmp_path: Path):
        results, _ = self._run(tmp_path, "result:\n  name: audit\n  formats: [text]\n")
        assert sorted(path.name for path in results.iterdir() if path.name.startswith(("audit", "result"))) == [
            "audit.txt"
        ]

    def test_per_task_writes_each_task_s_own_files(self, tmp_path: Path):
        results, _ = self._run(tmp_path, "result:\n  per_task: true\n  formats: [json, html]\n")
        assert sorted(path.name for path in results.glob("result*")) == [
            "result-task_a.html",
            "result-task_a.json",
            "result-task_b.html",
            "result-task_b.json",
        ]
        assert list(json.loads((results / "result-task_b.json").read_text())) == ["task_b"]

    def test_summary_detail_writes_the_summary(self, tmp_path: Path):
        _, result = self._run(tmp_path, "result:\n  detail: summary\n")
        assert result.report.call_args_list[-1].kwargs["detailed"] is False  # type: ignore[attr-defined]
        assert {call.kwargs["detailed"] for call in result._html_reports.call_args_list} == {False}  # type: ignore[attr-defined]

    def test_the_config_sets_the_width(self, tmp_path: Path):
        _, result = self._run(tmp_path, "result:\n  width: 100\n")
        assert {call.kwargs["width"] for call in result.report.call_args_list} == {100}  # type: ignore[attr-defined]

    def test_the_command_line_width_beats_the_config(self, tmp_path: Path):
        _, result = self._run(tmp_path, "result:\n  width: 100\n", report_width=72)
        assert {call.kwargs["width"] for call in result.report.call_args_list} == {72}  # type: ignore[attr-defined]


def _findings_result(*, warnings: bool) -> object:
    """A fake workflow result with real findings: a warning where *warnings*, and one passing finding."""
    from dataeval_flow.steps import Finding

    result = _fake_result(warnings=1 if warnings else 0)
    passing = Finding(severity="ok", title="Label Balance", brief="2 classes")
    flagged = Finding(severity="warning", title="Duplicates", brief="2 groups | 4 images", description="Look at them.")
    result.findings = [flagged, passing] if warnings else [passing]
    result.metadata.execution_time_s = 1.5
    return result


def _failed_result() -> object:
    """A failed task, which no file but the CI reports holds."""
    result = _fake_evaluator_result(success=False)
    result.errors = ["ValueError: boom", "and more"]
    return result


class TestGate:
    """What makes the run's exit code non-zero: 1 for a failed task, 3 for health warnings, as the config says."""

    @staticmethod
    def _exit(tmp_path: Path, extra: str, results: list[object], **kwargs: object) -> int:
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path, extra=extra)
        with patch.object(orch, "_run_single_task", side_effect=results):
            return run(config, tmp_path / "out", data_dir=tmp_path, **kwargs)  # type: ignore[arg-type]

    def test_a_failed_task_exits_1(self, tmp_path: Path):
        assert self._exit(tmp_path, "", [_failed_result(), _fake_result()]) == 1

    def test_the_config_can_gate_on_warnings(self, tmp_path: Path):
        assert self._exit(tmp_path, "result:\n  fail_on: warning\n", [_fake_result(warnings=1), _fake_result()]) == 3

    def test_a_failed_task_outranks_a_warning(self, tmp_path: Path):
        results = [_failed_result(), _fake_result(warnings=1)]
        assert self._exit(tmp_path, "result:\n  fail_on: warning\n", results) == 1

    def test_never_reports_without_failing(self, tmp_path: Path):
        assert self._exit(tmp_path, "result:\n  fail_on: never\n", [_failed_result(), _fake_result(warnings=1)]) == 0

    def test_never_holds_when_the_config_runs_no_task(self, tmp_path: Path):
        import dataeval_flow._export as export_mod
        from dataeval_flow._runner import run

        path = tmp_path / "config.yaml"
        path.write_text("result:\n  fail_on: never\ndatasets: []\nsources: []\n")
        _with_exports(path, "src")
        with patch.object(export_mod, "write_exports", return_value=1):
            assert run(path, tmp_path / "out", data_dir=tmp_path) == 0

    def test_the_command_line_beats_the_config(self, tmp_path: Path):
        extra = "result:\n  fail_on: warning\n"
        assert self._exit(tmp_path, extra, [_fake_result(warnings=1), _fake_result()], fail_on_warning=False) == 0


class TestCIFiles:
    """A JUnit report for CI's test views, and a Markdown summary for a job summary or a merge-request comment."""

    @staticmethod
    def _results(tmp_path: Path, formats: str) -> Path:
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path, extra=f"result:\n  formats: [{formats}]\n")
        with patch.object(orch, "_run_single_task", side_effect=[_findings_result(warnings=True), _failed_result()]):
            run(config, tmp_path / "out", data_dir=tmp_path)
        return tmp_path / "out" / "results"

    def test_junit_has_a_suite_per_task_and_a_case_per_finding(self, tmp_path: Path):
        import xml.etree.ElementTree as ET

        root = ET.parse(self._results(tmp_path, "junit") / "result.xml").getroot()  # noqa: S314 - the run's own file
        assert (root.tag, root.get("tests"), root.get("failures"), root.get("errors")) == ("testsuites", "3", "1", "1")
        task_a, task_b = root.findall("testsuite")
        assert (task_a.get("name"), task_a.get("time")) == ("task_a", "1.500")
        duplicates, balance = task_a.findall("testcase")
        assert (duplicates.get("time"), balance.get("time")) == ("1.500", None), "a CI sums its cases' times"
        assert (duplicates.get("name"), duplicates.get("classname")) == ("Duplicates", "task_a")
        failure = duplicates.find("failure")
        assert failure is not None
        assert (failure.get("message"), failure.text) == ("2 groups | 4 images", "Look at them.")
        assert list(balance) == []
        (run_case,) = task_b.findall("testcase")
        error = run_case.find("error")
        assert error is not None
        assert (run_case.get("name"), error.get("message"), error.text) == (
            "run",
            "ValueError: boom",
            "ValueError: boom\nand more",
        )

    def test_markdown_has_each_task_s_findings_and_failures(self, tmp_path: Path):
        text = (self._results(tmp_path, "markdown") / "result.md").read_text()
        assert "## task\\_a" in text
        assert "| warning | Duplicates | 2 groups \\| 4 images |" in text
        assert "| ok | Label Balance | 2 classes |" in text
        assert "## task\\_b: failed" in text
        assert "```\nValueError: boom\nand more\n```" in text

    def test_a_run_where_every_task_fails_still_reports_them_in_junit(self, tmp_path: Path):
        import dataeval_flow._orchestrator as orch
        from dataeval_flow._runner import run

        config = _write_config(tmp_path, extra="result:\n  formats: [json, junit]\n")
        with patch.object(orch, "_run_single_task", side_effect=[_failed_result(), _failed_result()]):
            assert run(config, tmp_path / "out", data_dir=tmp_path) == 1
        assert sorted(path.name for path in (tmp_path / "out" / "results").iterdir()) == ["result.xml"]


class TestCIReports:
    """The JUnit and Markdown files hold whatever a task's text or errors hold, and stay readable."""

    def test_junit_stays_valid_xml_whatever_an_error_holds(self):
        import xml.etree.ElementTree as ET

        from dataeval_flow._ci_reports import junit_report

        failed = _failed_result()
        failed.errors = ["RuntimeError: \x1b[31mboom\x00"]  # type: ignore[attr-defined]
        root = ET.fromstring(junit_report({"task\x07": failed}))  # type: ignore[dict-item]  # noqa: S314 - our own output
        assert root.find("testsuite").get("name") == "task"  # type: ignore[union-attr]
        assert root.find("testsuite/testcase/error").get("message") == "RuntimeError: [31mboom"  # type: ignore[union-attr]

    def test_junit_tells_a_task_s_findings_of_one_title_apart(self):
        import xml.etree.ElementTree as ET

        from dataeval_flow._ci_reports import junit_report
        from dataeval_flow.steps import Finding

        result = _findings_result(warnings=False)
        result.findings = [Finding(severity="ok", title="Outliers", brief=b) for b in "ab"]  # type: ignore[attr-defined]
        root = ET.fromstring(junit_report({"task": result}))  # type: ignore[dict-item]  # noqa: S314 - our own output
        assert [case.get("name") for case in root.iter("testcase")] == ["Outliers", "Outliers (2)"]

    def test_junit_times_a_long_run_in_seconds(self):
        import xml.etree.ElementTree as ET

        from dataeval_flow._ci_reports import junit_report

        result = _findings_result(warnings=True)
        result.metadata.execution_time_s = 1234567.891  # type: ignore[attr-defined]
        root = ET.fromstring(junit_report({"task": result}))  # type: ignore[dict-item]  # noqa: S314 - our own output
        assert root.find("testsuite").get("time") == "1234567.891"  # type: ignore[union-attr]

    def test_markdown_keeps_a_failed_task_s_errors_verbatim(self):
        from dataeval_flow._ci_reports import markdown_summary

        failed = _failed_result()
        failed.errors = ["SyntaxError: ```x```\n\n# field"]  # type: ignore[attr-defined]
        assert "````\nSyntaxError: ```x```\n\n# field\n````" in markdown_summary({"task": failed})  # type: ignore[dict-item]

    def test_markdown_shows_names_and_findings_as_written(self):
        from dataeval_flow._ci_reports import markdown_summary
        from dataeval_flow.steps import Finding

        result = _findings_result(warnings=False)
        result.findings = [Finding(severity="ok", title="Missing <NA>", brief="*none*")]  # type: ignore[attr-defined]
        text = markdown_summary({"clean_*train*": result})  # type: ignore[dict-item]
        assert "## clean\\_\\*train\\*" in text
        assert "| ok | Missing \\<NA\\> | \\*none\\* |" in text

    def test_markdown_names_an_evaluator_without_pointing_at_a_file(self):
        from dataeval_flow._ci_reports import markdown_summary

        text = markdown_summary({"dups": _fake_evaluator_result()})
        assert "`duplicates` ran; an evaluator has no findings to list." in text
