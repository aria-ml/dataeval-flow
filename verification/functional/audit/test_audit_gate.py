"""TC-18-5 — gating on an audit's verdict: `--require`, `result: require` and the exit code 4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from verification.fixtures import write_image_folder
from verification.functional.audit._helpers import audit, audit_pipeline, clean_pair
from verification.functional.chains._cli import invoke
from verification.helpers import run_cli

pytestmark = pytest.mark.required


def _output(proc) -> str:
    return proc.stdout + proc.stderr


def _project(root: Path, *, leaky: bool, require: str | None = None, extra_task: bool = False) -> Path:
    """Two image-folder splits and an `audit` task over them; *leaky* gives both the same images."""
    write_image_folder(root / "train", n_per_class=4, n_classes=2, seed=0)
    write_image_folder(root / "test", n_per_class=4, n_classes=2, seed=0 if leaky else 1)
    config: dict[str, Any] = {
        "datasets": [
            {"name": "train_ds", "format": "image_folder", "path": "train", "infer_labels": True},
            {"name": "test_ds", "format": "image_folder", "path": "test", "infer_labels": True},
        ],
        "sources": [{"name": "train", "dataset": "train_ds"}, {"name": "test", "dataset": "test_ds"}],
        "workflows": [
            {"name": "release", "type": "audit", "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}},
            {"name": "tidy", "type": "quality", "outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}},
        ],
        "tasks": [{"name": "audit_task", "workflow": "release", "sources": ["train", "test"]}],
    }
    if extra_task:
        config["tasks"].append({"name": "quality_task", "workflow": "tidy", "sources": ["train"]})
    if require is not None:
        config["result"] = {"require": require}
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """``run(root, config, *flags, env=...)``: the command in this process, for its exit code and output."""

    def _run(root: Path, config: Path, *args: str, env: dict[str, str] | None = None):
        return invoke(monkeypatch, capsys, "-c", config, "-d", root, "-o", root / "out", *args, env=env)

    return _run


class TestRequireGate:
    def test_the_command_process_exits_4_when_the_verdict_is_worse_than_required(self, tmp_path: Path) -> None:
        config = _project(tmp_path, leaky=True)
        proc = run_cli("-c", str(config), "-d", str(tmp_path), "--require", "ready-with-caveats")
        assert proc.returncode == 4, _output(proc)

    def test_a_not_ready_audit_exits_4_under_require(self, tmp_path: Path, run) -> None:
        config = _project(tmp_path, leaky=True)
        proc = run(tmp_path, config, "--require", "ready-with-caveats")
        assert proc.code == 4, _output(proc)
        entry = json.loads((tmp_path / "out" / "results" / "result.json").read_text())["audit_task"]
        assert entry["verdict"]["level"] == "not-ready"
        assert [item["check"] for item in entry["verdict"]["blocking"]] == ["leakage"]

    def test_without_require_the_exit_code_does_not_read_the_verdict(self, tmp_path: Path, run) -> None:
        config = _project(tmp_path, leaky=True)
        proc = run(tmp_path, config)
        assert proc.code == 0, _output(proc)

    def test_a_verdict_at_or_above_the_required_level_exits_0(self, tmp_path: Path, run) -> None:
        config = _project(tmp_path, leaky=False)
        proc = run(tmp_path, config, "--require", "ready-with-caveats")
        assert proc.code == 0, _output(proc)
        verdict = json.loads((tmp_path / "out" / "results" / "result.json").read_text())["audit_task"]["verdict"]
        assert verdict["level"] == "ready-with-caveats"

    def test_a_stricter_level_refuses_the_same_audit(self, tmp_path: Path, run) -> None:
        config = _project(tmp_path, leaky=False)
        assert run(tmp_path, config, "--require", "ready").code == 4
        # Not assessed checks (no extractor) are caveats that are not accepted risks.
        assert run(tmp_path, config, "--require", "ready-with-accepted-risks").code == 4

    def test_require_can_be_set_in_the_config_and_the_flag_overrides_it(self, tmp_path: Path, run) -> None:
        config = _project(tmp_path, leaky=False, require="ready-with-caveats")
        assert run(tmp_path, config).code == 0
        assert run(tmp_path, config, "--require", "ready").code == 4

    def test_require_can_be_set_in_the_environment(self, tmp_path: Path, run) -> None:
        config = _project(tmp_path, leaky=True)
        proc = run(tmp_path, config, env={"DATAEVAL_REQUIRE": "ready-with-caveats"})
        assert proc.code == 4, _output(proc)

    def test_a_level_that_is_not_one_of_the_three_is_refused(self, tmp_path: Path, run) -> None:
        config = _project(tmp_path, leaky=False)
        proc = run(tmp_path, config, "--require", "nope")
        assert proc.code != 0
        assert "ready-with-caveats" in _output(proc)

    def test_fail_on_warning_exits_3_and_a_verdict_short_of_require_takes_precedence(self, tmp_path: Path, run) -> None:
        config = _project(tmp_path, leaky=True)
        assert run(tmp_path, config, "--fail-on-warning").code == 3
        proc = run(tmp_path, config, "--fail-on-warning", "--require", "ready-with-caveats")
        assert proc.code == 4, _output(proc)


class TestTasksThatGiveNoVerdict:
    def test_a_run_in_which_no_task_gives_a_verdict_is_refused_before_any_task_starts(
        self, tmp_path: Path, run
    ) -> None:
        config = _project(tmp_path, leaky=False)
        proc = run(tmp_path, config, "--task", "audit_task", "--require", "ready")
        assert proc.code == 4  # sanity: the audit task alone is judged
        quality_only = yaml.safe_load(config.read_text())
        quality_only["tasks"] = [{"name": "quality_task", "workflow": "tidy", "sources": ["train"]}]
        config.write_text(yaml.safe_dump(quality_only))
        proc = run(tmp_path, config, "--require", "ready")
        assert proc.code == 1
        assert "gates on a verdict, and no task this run runs gives one" in _output(proc)
        assert not (tmp_path / "out" / "results" / "result.json").exists() or "quality_task" not in json.loads(
            (tmp_path / "out" / "results" / "result.json").read_text()
        )

    def test_a_task_that_gives_no_verdict_is_not_judged_beside_one_that_does(self, tmp_path: Path, run) -> None:
        config = _project(tmp_path, leaky=False, extra_task=True)
        proc = run(tmp_path, config, "--require", "ready-with-caveats")
        assert proc.code == 0, _output(proc)
        results = json.loads((tmp_path / "out" / "results" / "result.json").read_text())
        assert "verdict" in results["audit_task"]
        assert "verdict" not in results["quality_task"]

    def test_a_failed_task_exits_1_before_the_verdict_gate(self, tmp_path: Path, run) -> None:
        config = _project(tmp_path, leaky=False)
        data = yaml.safe_load(config.read_text())
        data["workflows"][0]["factor-leakage"] = {"factors": ["missing"]}
        config.write_text(yaml.safe_dump(data))
        proc = run(tmp_path, config, "--require", "ready-with-caveats")
        assert proc.code == 1, _output(proc)
        entry = json.loads((tmp_path / "out" / "results" / "result.json").read_text())["audit_task"]
        assert "verdict" not in entry
        assert entry["health"]["status"] == "failed"


class TestAcceptedRisksGate:
    """`ready-with-accepted-risks` lets through an audit whose only caveats are accepted warnings.

    Data with metadata factors and an extractor is needed for an audit with nothing left unassessed, which a config
    file cannot name (its datasets are files on disk), so these tests hand the command an in-memory pipeline in
    place of its config file.
    """

    @staticmethod
    def _exit(monkeypatch, capsys, tmp_path: Path, config, level: str) -> int:
        with patch("dataeval_flow._runner._resolve_config", return_value=config):
            return invoke(monkeypatch, capsys, "-c", "pipeline.yaml", "-d", tmp_path, "--require", level).code

    def test_an_audit_whose_only_caveat_is_an_accepted_warning_passes_the_middle_gate_only(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        # Default thresholds: dimensional-completeness warns on this data. Accept it; nothing else is wrong.
        accepted = {"accepted": {"dimensional-completeness": "Flattened pixels fill few dimensions by design."}}
        config = audit_pipeline(clean_pair(), accepted, extractor=True)
        result = audit(clean_pair(), accepted, extractor=True)
        verdict = result.verdict
        assert verdict is not None
        assert (verdict.level, verdict.warnings, verdict.not_assessed) == ("ready-with-caveats", [], [])
        assert [a.state for a in verdict.accepted] == ["warned"]

        assert self._exit(monkeypatch, capsys, tmp_path, config, "ready-with-caveats") == 0
        assert self._exit(monkeypatch, capsys, tmp_path, config, "ready-with-accepted-risks") == 0
        assert self._exit(monkeypatch, capsys, tmp_path, config, "ready") == 4

    def test_an_unaccepted_warning_fails_the_middle_gate(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        config = audit_pipeline(clean_pair(), None, extractor=True)  # the same warning, not accepted
        assert self._exit(monkeypatch, capsys, tmp_path, config, "ready-with-caveats") == 0
        assert self._exit(monkeypatch, capsys, tmp_path, config, "ready-with-accepted-risks") == 4
