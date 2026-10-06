"""`result: require:` and `--require`: the command's exit code reads an audit's verdict (follow-ons spec §3)."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dataeval_flow._cache import DatasetCache
from dataeval_flow._runner import run
from dataeval_flow.config import PipelineConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import Items, ToyImages
from tests.test_audit_preset import _OUTLIERS
from tests.test_audit_run import _leaky


@pytest.fixture(autouse=True)
def _fresh_caches():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _audit(datasets: dict[str, Any], *, require: str | None = None, matrix: dict | None = None) -> PipelineConfig:
    task: dict[str, Any] = {"name": "t", "workflow": "w", "sources": list(datasets)[:2]}
    if matrix is not None:
        task["matrix"] = matrix
    return chain_pipeline(
        workflows=[{"name": "w", "type": "audit", **_OUTLIERS}],
        tasks=[task],
        datasets=datasets,
        extra={"result": {"require": require}} if require is not None else None,
    )


def _exit(config: PipelineConfig, tmp_path: Path, **kwargs: Any) -> int:
    with patch("dataeval_flow._runner._resolve_config", return_value=config):
        return run("pipeline.yaml", None, data_dir=tmp_path, **kwargs)


def test_a_not_ready_audit_exits_4_under_require(tmp_path: Path) -> None:
    assert _exit(_audit(_leaky(), require="ready-with-caveats"), tmp_path) == 4


def test_a_not_ready_audit_exits_0_without_require(tmp_path: Path) -> None:
    assert _exit(_audit(_leaky()), tmp_path) == 0


def test_a_one_split_audit_passes_ready_with_caveats_and_falls_short_of_accepted_risks(tmp_path: Path) -> None:
    # One split leaves the split checks not assessed: a caveat that is not an accepted risk.
    assert _exit(_audit({"train": ToyImages()}, require="ready-with-caveats"), tmp_path) == 0
    assert _exit(_audit({"train": ToyImages()}, require="ready-with-accepted-risks"), tmp_path) == 4


def test_the_flag_overrides_the_config(tmp_path: Path) -> None:
    # The config alone passes this audit (above); the stricter flag refuses it.
    assert _exit(_audit({"train": ToyImages()}, require="ready-with-caveats"), tmp_path, require="ready") == 4


def test_a_verdict_short_of_require_exits_4_over_fail_on_warning_s_3(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _audit(_leaky(), require="ready-with-caveats")
    config.result.fail_on = "warning"
    assert _exit(config, tmp_path) == 4
    assert "Health warnings raised by: t" in caplog.text
    config.result.require = None
    assert _exit(config, tmp_path) == 3


def test_a_matrix_run_that_failed_falls_short(tmp_path: Path) -> None:
    datasets = {"train": ToyImages(), "test": ToyImages(seed=1), "empty": Items([])}
    config = _audit(datasets, require="ready-with-caveats", matrix={"sources": [["train", "test"], ["train", "empty"]]})
    # The failed run alone would exit 1; `fail_on: never` lets the verdict gate speak for it.
    config.result.fail_on = "never"
    assert _exit(config, tmp_path) == 4


def test_require_with_no_task_that_gives_a_verdict_is_refused_before_any_task_runs(tmp_path: Path) -> None:
    config = chain_pipeline(
        evaluators=[DuplicatesConfig(name="d")],
        tasks=[{"name": "t", "workflow": "d", "sources": ["src"], "kind": "evaluator"}],
        extra={"result": {"require": "ready"}},
    )
    with (
        patch("dataeval_flow._orchestrator._run_single_task") as ran,
        pytest.raises(ValueError, match="gates on a verdict, and no task this run runs gives one"),
    ):
        _exit(config, tmp_path)
    ran.assert_not_called()


def test_a_failed_task_still_exits_1_before_the_verdict_gate(tmp_path: Path) -> None:
    # A matrix run that fails is a failed result; a plain task's empty split raises before it has one.
    datasets = {"train": ToyImages(), "test": ToyImages(seed=1), "empty": Items([])}
    config = _audit(datasets, require="ready", matrix={"sources": [["train", "test"], ["train", "empty"]]})
    assert _exit(config, tmp_path) == 1


def test_the_cli_reads_require_from_the_flag_and_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataeval_flow.__main__ import _build_parser

    assert _build_parser().parse_args(["--require", "ready"]).require == "ready"
    monkeypatch.setenv("DATAEVAL_REQUIRE", "ready-with-accepted-risks")
    assert _build_parser().parse_args([]).require == "ready-with-accepted-risks"
    monkeypatch.setenv("DATAEVAL_REQUIRE", "nope")
    with pytest.raises(ValueError, match="DATAEVAL_REQUIRE must be one of"):
        _build_parser()
