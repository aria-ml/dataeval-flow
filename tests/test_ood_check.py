"""The `ood` check, and the OOD evaluators' report section (ood-detection spec §5.1, §8)."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from dataeval.shift import OODOutput

from dataeval_flow import run_task
from dataeval_flow._blocks import Fields, Paragraph, Table
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators.shift import OODKNeighborsConfig
from dataeval_flow.evaluators.shift._report import derived_threshold, ood_section
from dataeval_flow.evaluators.shift._rows import OODRowsOutput
from dataeval_flow.steps import ChainResult, CheckContext
from dataeval_flow.steps.checks import OODCheck, OODConfig
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages

_ROWS = {
    "unit": "detections",
    "confidence": 0.25,
    "compared": {"reference": 50, "tests[cam1]": 30},
    "images": {"reference": 25, "tests[cam1]": 5},
    "detections": [{"image": i % 4, "score": 1.0, "is_ood": i < 3} for i in range(30)],
    "unassessed": [4],
}


def _output(flags: list[bool]) -> OODOutput:
    scores = np.asarray([2.0 if flag else 1.0 for flag in flags], dtype=np.float32)
    return OODOutput(is_ood=np.asarray(flags), instance_score=scores, feature_score=None)


def _judge(output: Any, **thresholds: Any) -> Any:
    config = OODConfig(input="knn", subject="KNN", **thresholds)
    (finding,) = OODCheck().run(config, {"input": SimpleNamespace(value=output, config=None)}, CheckContext("t", "s"))
    return finding


@pytest.mark.parametrize(
    ("flagged", "severity"), [(0, "ok"), (1, "info"), (9, "info"), (10, "warning"), (40, "warning")]
)
def test_severity_follows_the_percent_of_images_flagged(flagged: int, severity: str) -> None:
    finding = _judge(_output([True] * flagged + [False] * (100 - flagged)))
    assert (finding.severity, finding.title) == (severity, "KNN")
    assert finding.brief == f"{flagged}/100 images OOD ({flagged:.1f}%)"
    assert finding.description == f"{flagged} of 100 test images score above the detector's threshold."


def test_a_null_threshold_judges_nothing_at_its_level() -> None:
    assert _judge(_output([True] * 50 + [False] * 50), warning=None).severity == "info"
    assert _judge(_output([True] * 5 + [False] * 95), info=None).severity == "ok"
    assert _judge(_output([False] * 100), warning=None, info=None).severity == "info"


def test_info_zero_makes_nothing_flagged_info_as_legacy_did() -> None:
    assert _judge(_output([False] * 10), info=0.0).severity == "info"


def test_on_detection_rows_the_percent_is_of_assessed_images_and_detections_are_counted() -> None:
    scores = np.asarray([2, 1, 2, 1, np.nan], dtype=np.float32)
    output = OODRowsOutput(
        is_ood=np.asarray([True, False, True, False, False]), instance_score=scores, feature_score=None, rows=_ROWS
    )
    finding = _judge(output)
    assert finding.brief == "2/4 images OOD (50.0%; 3/30 detections at confidence ≥ 0.25)"
    assert finding.description == (
        "2 of 4 test images score above the detector's threshold. An image is out of distribution when any of its "
        "detections is."
    )


def test_the_derived_threshold_is_the_highest_unflagged_assessed_score() -> None:
    assert derived_threshold([1.0, 3.0, 2.0, 5.0], [False, True, False, True]) == 2.0
    assert derived_threshold([1.0, float("nan"), 2.0], [False, False, True]) == 1.0
    assert derived_threshold([None, float("nan")], [False, False]) is None


def test_every_image_flagged_takes_the_lowest_score_as_threshold() -> None:  # Review Focus 5
    assert derived_threshold([3.0, 2.0, float("nan"), 4.0], [True, True, False, True]) == 2.0


def test_the_section_shows_the_histogram_and_the_counts() -> None:
    data = {"is_ood": [False] * 8 + [True] * 2, "instance_score": [float(i) for i in range(10)], "feature_score": None}
    histogram, fields = ood_section({"data": data})
    assert isinstance(histogram, Table)
    assert sum(cast("int", row["ood"]) for row in histogram.rows) == 2
    assert fields == Fields(items=[("Flagged", 2), ("Assessed", 10), ("Threshold", 7.0)])


def test_on_detection_rows_the_section_says_what_was_compared_and_what_was_not_assessed() -> None:
    rows = {**_ROWS, "compared": {"reference": 50, "tests[cam1]": 6}, "images": {"reference": 25, "tests[cam1]": 4}}
    rows["unassessed"] = [3]
    data = {
        "is_ood": [True, False, False, False],
        "instance_score": [3.0, 1.0, 2.0, float("nan")],
        "feature_score": None,
        "rows": rows,
    }
    *_, fields, unassessed = ood_section({"data": data})
    assert isinstance(fields, Fields)
    assert ("Assessed", 3) in fields.items
    compared = "`reference` 50 in 25 images; `tests[cam1]` 6 in 4 images (confidence ≥ 0.25)"
    assert ("Compared", compared) in fields.items
    assert unassessed == Paragraph(text="Not assessed, with no detection at confidence ≥ 0.25: images 3.")


def test_a_chain_judges_a_brightened_source_and_shows_the_detector_s_section() -> None:
    steps = [
        {"name": "knn", "evaluator": "knn", "input": ["reference", "tests"]},
        {"name": "knn-check", "check": "ood", "input": "knn"},
    ]
    workflow = {"name": "w", "inputs": ["reference", {"name": "tests", "list": True}], "steps": steps}
    config = chain_pipeline(
        workflows=[workflow],
        evaluators=[OODKNeighborsConfig(name="knn", k=5, distance_metric="euclidean")],
        datasets={"reference": ToyImages(40), "cam1": ToyImages(40, seed=1, bright=True)},
        extractor=True,
    )
    result = run_task(TaskConfig(name="t", workflow="w", sources=["reference", "cam1"], extractor="flat"), config)
    assert isinstance(result, ChainResult)
    (finding,) = result.findings
    assert (finding.severity, finding.title) == ("warning", "OOD (K-Neighbors) · knn")
    assert "Threshold" in result.report()
