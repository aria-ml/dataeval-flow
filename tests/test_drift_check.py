"""The `drift` check: a warning on drift, or, chunked, when enough chunks drift or enough drift in a row (§10.11)."""

from types import SimpleNamespace
from typing import Any

import polars as pl

from dataeval_flow import run_task
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators.shift import DriftKNeighborsConfig
from dataeval_flow.steps import ChainResult, CheckContext
from dataeval_flow.steps.checks import DriftCheck, DriftConfig
from tests.chain_toys import chain_pipeline
from tests.drift_toys import ClassImages
from tests.evaluator_toys import ToyImages

_SHIFTED = {"reference": ToyImages(40), "cam1": ToyImages(40, seed=1, bright=True)}
_SAME = {"reference": ToyImages(40), "cam1": ToyImages(40, seed=1)}


def _finding(
    datasets: dict[str, Any], *, entry: dict[str, Any] | None = None, by: str | None = None, **check: Any
) -> Any:
    """The one finding a `drift` check makes on `knn`'s Output for `cam1`."""
    knn = DriftKNeighborsConfig.model_validate({"name": "knn", "type": "drift-kneighbors", "k": 3, **(entry or {})})
    step = {"name": "knn", "evaluator": "knn", "input": ["reference", "tests"]}
    judge = {"name": "judge", "check": "drift", "input": "knn", **check}
    if by is not None:
        step["by"] = judge["by"] = by
    workflow = {"name": "w", "inputs": ["reference", {"name": "tests", "list": True}], "steps": [step, judge]}
    config = chain_pipeline(workflows=[workflow], evaluators=[knn], datasets=datasets, extractor=True)
    result = run_task(TaskConfig(name="t", workflow="w", sources=["reference", "cam1"], extractor="flat"), config)
    assert isinstance(result, ChainResult)
    elements = result.steps["judge"].elements
    assert elements is not None
    (finding,) = elements["cam1"].output
    return finding


def test_unchunked_drift_warns_and_briefs_drift():
    finding = _finding(_SHIFTED)
    assert (finding.severity, finding.brief) == ("warning", "drift")


def test_no_drift_is_ok():
    finding = _finding(_SAME)
    assert (finding.severity, finding.brief) == ("ok", "no drift")


def test_warn_on_drift_false_makes_drift_info():
    assert _finding(_SHIFTED, warn_on_drift=False).severity == "info"


def test_chunked_briefs_its_chunks_and_warns_at_chunk_percent():
    finding = _finding(_SHIFTED, entry={"chunking": {"chunk_count": 4}})
    drifted, total = (int(part) for part in finding.brief.split(" ")[0].split("/"))
    assert finding.brief.endswith("chunks drifted")
    assert total >= 1
    assert finding.severity == ("warning" if drifted / total > 0.10 else "info" if drifted else "ok")


def test_chunked_warns_past_consecutive_chunks_when_percent_is_off():
    finding = _finding(_SHIFTED, entry={"chunking": {"chunk_count": 4}}, chunk_percent=None, consecutive_chunks=1)
    longest = int(finding.description.rsplit("max consecutive: ", 1)[1])
    assert finding.severity == ("warning" if longest > 1 else "info" if longest else "ok")


def test_chunked_with_both_thresholds_off_is_info():
    finding = _finding(_SHIFTED, entry={"chunking": {"chunk_count": 4}}, chunk_percent=None, consecutive_chunks=None)
    assert finding.severity == "info"


def test_the_title_is_the_evaluators_and_its_differing_name():
    assert _finding(_SAME).title == "Drift (K-Neighbors) · knn"


def test_subject_overrides_the_title():
    assert _finding(_SAME, subject="Camera drift").title == "Camera drift"


def test_by_class_rolls_up_under_the_title():
    counts = {0: 15, 1: 15, 2: 15}
    datasets = {"reference": ClassImages(counts), "cam1": ClassImages(counts, seed=1, bright_classes={2})}
    finding = _finding(datasets, by="class")
    assert finding.title == "Drift (K-Neighbors) · knn by class"
    assert finding.brief.endswith("/3 classes warn")


def _judged(flags: list[bool], **limits: Any) -> Any:
    """The finding `drift` makes on a chunked Output whose chunks drifted as `flags` say."""
    output = SimpleNamespace(details=pl.DataFrame({"drifted": flags}), drifted=any(flags))
    config = DriftConfig.model_validate({"input": "knn", **limits})
    (finding,) = DriftCheck().run(config, {"input": SimpleNamespace(value=output, config=None)}, CheckContext("t", "s"))
    return finding


def test_three_scattered_chunks_of_ten_warn_by_percent_alone():
    flags = [True, False, False, True, False, False, True, False, False, False]
    finding = _judged(flags)
    assert (finding.severity, finding.brief) == ("warning", "3/10 chunks drifted")
    assert finding.description == "3/10 chunks drifted (30%) | max consecutive: 1"
    assert _judged(flags, chunk_percent=None).severity == "info"


def test_three_chunks_in_a_row_warn_with_percent_off():
    flags = [False, True, True, True, False, False, False, False, False, False]
    finding = _judged(flags, chunk_percent=None)
    assert finding.severity == "warning"
    assert finding.description == "3/10 chunks drifted (30%) | max consecutive: 3"


def test_a_zero_chunk_percent_with_no_chunk_drifted_is_ok():
    assert _judged([False] * 10, chunk_percent=0).severity == "ok"
