"""Where OOD detectors agree on a test source's images: the `ood-union` combine, its section, and the `ood-agreement`
check (ood-detection spec §5.2, §5.3, §8)."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from dataeval.shift import OODOutput
from pydantic import ValidationError

from dataeval_flow import run_task
from dataeval_flow._blocks import ItemRef, Paragraph, Section, Table
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators.shift import OODKNeighborsConfig
from dataeval_flow.evaluators.shift._rows import OODRowsOutput
from dataeval_flow.steps import ChainResult, CheckContext
from dataeval_flow.steps.checks import OODAgreementCheck, OODAgreementConfig
from dataeval_flow.steps.combines import OODUnionOutput
from dataeval_flow.steps.combines._ood import union_blocks, union_of
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages

_ON = (SimpleNamespace(address="reference"), SimpleNamespace(address="tests[cam1]"))


def _node(step: str, flags: list[bool], scores: list[float], rows: dict[str, Any] | None = None) -> Any:
    flagged, scored = np.asarray(flags), np.asarray(scores, dtype=np.float32)
    output = (
        OODRowsOutput(is_ood=flagged, instance_score=scored, feature_score=None, rows=rows)
        if rows is not None
        else OODOutput(is_ood=flagged, instance_score=scored, feature_score=None)
    )
    return SimpleNamespace(value=output, step=step, computed_on=_ON)


# Five images. Each detector's threshold is its highest unflagged score: a 1.0, b 2.0, c 1.0.
_A = _node("a", [True, True, False, True, False], [3.0, 4.0, 1.0, 2.0, 0.5])
_B = _node("b", [True, False, True, True, False], [4.0, 1.0, 6.0, 4.0, 2.0])
_C = _node("c", [True, False, False, False, False], [2.0, 1.0, 0.5, 1.0, 0.5])


def test_with_two_detectors_each_flagged_image_is_mutual_or_unique() -> None:
    union = union_of([_A, _B])
    assert (union.source, union.detectors, union.images, union.assessed) == ("tests[cam1]", ["a", "b"], 5, 5)
    assert (union.union, union.mutual, union.partial) == ([0, 1, 2, 3], [0, 3], [])
    assert union.unique == {"a": [1], "b": [2]}
    assert union.thresholds == {"a": 1.0, "b": 2.0}
    expected = [(3.0 + 2.0) / 2, (4.0 + 0.5) / 2, (1.0 + 3.0) / 2, (2.0 + 2.0) / 2, (0.5 + 2.0 / 2.0) / 2]
    assert union.scores == pytest.approx(expected)
    assert union.flagged_detections is None


def test_with_three_detectors_an_image_some_but_not_all_flagged_is_partial() -> None:
    union = union_of([_A, _B, _C])
    assert union.mutual == [0]
    assert union.partial == [3]
    assert union.unique == {"a": [1], "b": [2], "c": []}


def test_with_one_detector_its_flags_are_mutual_and_nothing_is_unique() -> None:
    union = union_of([_A])
    assert (union.union, union.mutual, union.partial, union.unique) == ([0, 1, 3], [0, 1, 3], [], {"a": []})


def test_a_detector_with_a_non_positive_threshold_is_left_out() -> None:  # Review Focus 3
    zero = _node("z", [False] * 5, [0.0] * 5)
    union = union_of([_A, zero])
    assert union.left_out == ["z"]
    assert union.thresholds["z"] == 0.0
    assert (union.mutual, union.unique) == ([0, 1, 3], {"a": []})
    blocks = union_blocks(union)
    assert blocks[0] == Paragraph(text="Left out, with a threshold that is not positive: `z`.")


def test_unassessed_images_count_only_where_every_detector_assessed_and_score_by_those_that_did() -> None:
    rows = {
        "detections": [{"image": 0, "score": 9.0, "is_ood": True}] * 2 + [{"image": 1, "score": 0.1, "is_ood": False}]
    }
    rows |= {"unassessed": [2, 3, 4], "confidence": 0.3}
    partly = _node("p", [True, False, False, False, False], [9.0, 0.1, np.nan, np.nan, np.nan], rows)
    union = union_of([_A, partly])
    assert union.assessed == 2
    assert union.scores[4] == pytest.approx(0.5)  # a's alone: 0.5 / 1.0
    assert union.flagged_detections == [2, 0, 0, 0, 0]


def test_the_section_pictures_each_flagged_image_once_and_folds_away_the_partial_and_unique_groups() -> None:
    union = union_of([_A, _B, _C])
    mutual, partial, *unique = union_blocks(union)
    assert isinstance(mutual, Table)
    assert [row["item"] for row in mutual.rows] == [0]
    image = mutual.rows[0]["image"]
    assert isinstance(image, ItemRef)
    assert image.source == "tests[cam1]"
    assert isinstance(partial, Section)
    assert partial.reference
    assert partial.title == "Some detectors agree"
    assert [(section.title, section.reference) for section in cast("list[Section]", unique)] == [
        ("a", True),
        ("b", True),
    ]
    pictured = [row["item"] for block in (mutual, partial, *unique) for row in _rows(block)]
    assert sorted(pictured) == union.union


def _rows(block: Any) -> list[dict[str, Any]]:
    tables = [block] if isinstance(block, Table) else [inner for inner in block.blocks if isinstance(inner, Table)]
    return [row for table in tables for row in table.rows]


def test_the_agreement_check_judges_the_mutual_share_and_lists_unique_images() -> None:
    config = OODAgreementConfig(input="agreement")
    node = SimpleNamespace(value=union_of([_A, _B]), config=None)
    aggregate, unique = OODAgreementCheck().run(config, {"input": node}, CheckContext("t", "s"))
    assert (aggregate.severity, aggregate.title) == ("warning", "Aggregate OOD (all detectors agree)")
    assert aggregate.brief == "2/4 OOD images agreed by all detectors (40.0%)"
    assert (unique.severity, unique.title) == ("info", "Unique OOD Samples (single-detector only)")
    assert unique.brief == "2 image(s) flagged by only one detector"


def test_nothing_unique_makes_one_finding() -> None:
    node = SimpleNamespace(value=union_of([_A]), config=None)
    (aggregate,) = OODAgreementCheck().run(OODAgreementConfig(input="u"), {"input": node}, CheckContext("t", "s"))
    assert aggregate.brief == "3/3 OOD images agreed by all detectors (60.0%)"


def _chain(*steps: dict[str, Any], inputs: list[Any] | None = None, datasets: dict[str, Any] | None = None) -> Any:
    knn = OODKNeighborsConfig(name="knn", k=5, distance_metric="euclidean")
    knn3 = OODKNeighborsConfig(name="knn3", k=3, distance_metric="euclidean")
    workflow = {"name": "w", "inputs": inputs or ["reference", {"name": "tests", "list": True}], "steps": list(steps)}
    data = datasets or {"reference": ToyImages(40), "cam1": ToyImages(40, seed=1, bright=True)}
    return chain_pipeline(workflows=[workflow], evaluators=[knn, knn3], datasets=data, extractor=True), list(data)


def test_a_chain_combines_two_detectors_and_pictures_the_agreed_images() -> None:
    config, sources = _chain(
        {"name": "knn", "evaluator": "knn", "input": ["reference", "tests"]},
        {"name": "knn3", "evaluator": "knn3", "input": ["reference", "tests"]},
        {"name": "agreement", "combine": "ood-union", "input": ["knn", "knn3"]},
        {"name": "agreement-check", "check": "ood-agreement", "input": "agreement"},
    )
    result = run_task(TaskConfig(name="t", workflow="w", sources=sources, extractor="flat"), config, report_images=True)
    assert isinstance(result, ChainResult)
    union = (result.steps["agreement"].elements or {})["cam1"].output
    assert isinstance(union, OODUnionOutput)
    assert {asset.item.source for asset in result.assets} == {"tests[cam1]"}
    assert result.findings[0].title == "Aggregate OOD (all detectors agree)"


def test_detectors_of_different_comparisons_are_refused_at_load() -> None:
    with pytest.raises(
        ValidationError, match="Step 'agreement' reads Outputs on `input` computed on different Datasets"
    ):
        _chain(
            {"name": "knn", "evaluator": "knn", "input": ["reference", "a"]},
            {"name": "knn3", "evaluator": "knn3", "input": ["reference", "b"]},
            {"name": "agreement", "combine": "ood-union", "input": ["knn", "knn3"]},
            inputs=["reference", "a", "b"],
            datasets={"reference": ToyImages(20), "a": ToyImages(20, seed=1), "b": ToyImages(20, seed=2)},
        )
