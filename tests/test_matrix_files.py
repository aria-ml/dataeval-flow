"""A matrix result through the runner: printed and written even when a run failed, gating on its warnings, a JUnit
suite per run, a Markdown table, and its encoding (task-matrix spec §6, §7.2, §7.3)."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dataeval import Metadata

from dataeval_flow import run_tasks
from dataeval_flow._binning import describe_binning
from dataeval_flow._cache import DatasetCache
from dataeval_flow._ci_reports import junit_report, markdown_summary
from dataeval_flow._encoding_cli import agreed
from tests.chain_toys import chain_pipeline

_CLEANING = {"name": "cleaning", "type": "data-cleaning", "outlier_method": "zscore", "outlier_flags": ["pixel"]}


@pytest.fixture(autouse=True)
def _fresh_cache() -> Any:
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _results() -> dict[str, Any]:
    config = chain_pipeline(
        workflows=[{**_CLEANING, "health_thresholds": {"image_outliers": 0.0}}],
        tasks=[{"name": "t", "workflow": "cleaning", "sources": "src", "matrix": {"outlier_threshold": [1.0, 3.0]}}],
    )
    return run_tasks(config)


def test_junit_writes_a_suite_per_run() -> None:
    root = ET.fromstring(junit_report(_results()))  # noqa: S314 - our own output
    assert [suite.get("name") for suite in root] == ["t · run 1", "t · run 2"]


def test_markdown_writes_the_comparison_table() -> None:
    text = markdown_summary(_results())
    assert "## t" in text
    # Markdown's punctuation is escaped in a header, as everywhere in the summary.
    assert "| \\# | outlier\\_threshold | Health |" in text


def test_the_runner_writes_a_matrix_and_gates_on_its_warnings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataeval_flow import _runner

    config = chain_pipeline(
        workflows=[{**_CLEANING, "health_thresholds": {"image_outliers": 0.0}}],
        tasks=[{"name": "t", "workflow": "cleaning", "sources": "src", "matrix": {"outlier_threshold": [1.0, 3.0]}}],
        extra={"result": {"formats": ["json", "text"], "fail_on": "warning"}},
    )
    # The pipeline holds in-memory datasets, which no config file can name: hand it to the runner as loaded.
    monkeypatch.setattr(_runner, "_resolve_config", lambda *_args, **_kwargs: config)
    assert _runner.run(None, output_dir=tmp_path, data_dir=tmp_path) == 3
    files = {path.suffix: path for path in (tmp_path / "results").iterdir() if path.suffix in (".json", ".txt")}
    assert json.loads(files[".json"].read_text(encoding="utf-8"))["t"]["kind"] == "matrix"
    assert "outlier_threshold" in files[".txt"].read_text(encoding="utf-8")


def test_the_runner_prints_and_writes_a_failed_matrix_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from dataeval_flow import _runner

    # k-means can't make 50 clusters of 12 items, so run 2's outliers step fails and run 1 finishes.
    entry = {**_CLEANING, "outlier_cluster_threshold": 1.0, "outlier_cluster_algorithm": "kmeans"}
    config = chain_pipeline(
        workflows=[entry],
        extractor=True,
        tasks=[
            {
                "name": "t",
                "workflow": "cleaning",
                "sources": "src",
                "extractor": "flat",
                "matrix": {"outlier_n_clusters": [None, 50]},
            }
        ],
        extra={"result": {"formats": ["json", "text", "html", "markdown", "junit"]}},
    )
    monkeypatch.setattr(_runner, "_resolve_config", lambda *_args, **_kwargs: config)
    assert _runner.run(None, output_dir=tmp_path, data_dir=tmp_path) == 1
    assert "Run 2 failed:" in capsys.readouterr().out
    results = tmp_path / "results"
    payload = json.loads((results / "result.json").read_text(encoding="utf-8"))["t"]
    assert (payload["health"]["status"], payload["health"]["failed_runs"]) == ("failed", [2])
    assert [run["result"]["kind"] for run in payload["runs"]] == ["workflow", "workflow"]
    assert "Run 2 failed:" in (results / "result.txt").read_text(encoding="utf-8")
    assert "Run 2 · outlier_n_clusters=50" in (results / "result.html").read_text(encoding="utf-8")
    assert "**Health:** failed" in (results / "result.md").read_text(encoding="utf-8")
    suites = ET.fromstring((results / "result.xml").read_text(encoding="utf-8"))  # noqa: S314 - our own output
    assert [suite.get("name") for suite in suites] == ["t · run 1", "t · run 2"]


def test_agreed_compares_rendered_descriptors_and_skips_records_without_one() -> None:
    assert agreed([("run 1", None), ("run 2", None)]) == (None, [])
    record = _record(0.0)
    assert agreed([("run 1", record), ("run 2", None), ("run 3", _record(0.0))]) == (record, [])
    assert agreed([("run 1", record), ("run 2", _record(10.0))]) == (None, ["run 1", "run 2"])


def _record(edge: float) -> dict[str, Any]:
    """A binning record whose one continuous factor is cut at `edge`."""
    rng = np.random.default_rng(0)
    metadata = Metadata.from_factors(
        {"temp_c": rng.normal(20.0, 3.0, 200)}, continuous_factor_bins={"temp_c": [-np.inf, edge, np.inf]}
    )
    return describe_binning(metadata)


def _matrix_file(tmp_path: Path, *edges: float) -> Path:
    runs = [
        {
            "number": number,
            "label": f"k={number}",
            "values": {"k": number},
            "result": {"metadata": {"metadata_binning": _record(edge)}},
        }
        for number, edge in enumerate(edges, start=1)
    ]
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"t": {"kind": "matrix", "runs": runs}}), encoding="utf-8")
    return path


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_matrix_whose_runs_agree_gives_its_encoding(tmp_path: Path) -> None:
    from dataeval_flow._encoding_cli import write_encoding

    assert write_encoding(_matrix_file(tmp_path, 0.0, 0.0), output=tmp_path / "encoding.json", task="t") == 0
    assert "temp_c" in json.loads((tmp_path / "encoding.json").read_text(encoding="utf-8"))["factors"]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_a_matrix_whose_runs_were_encoded_differently_is_refused_naming_them(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from dataeval_flow._encoding_cli import write_encoding

    assert write_encoding(_matrix_file(tmp_path, 0.0, 10.0)) == 1
    assert "run 1, run 2 were encoded differently" in caplog.text
