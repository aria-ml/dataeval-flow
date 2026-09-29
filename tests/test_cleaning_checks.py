"""data-cleaning's findings, made by checks: each agrees with the finding it replaces (spec §9.2, §10.3)."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from dataeval.quality import OutliersOutput
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.evaluators.quality import LabelHealthOutput
from dataeval_flow.steps import CheckContext
from dataeval_flow.steps.checks import (
    ClasswiseOutlierRateCheck,
    ClasswiseOutlierRateConfig,
    OutlierRateCheck,
    OutlierRateConfig,
    TargetOutlierRateCheck,
    TargetOutlierRateConfig,
)
from dataeval_flow.steps.combines import ClasswiseOutliers, ClasswiseRow
from dataeval_flow.workflows import Finding
from tests.chain_toys import ToyDetections, chain_pipeline
from tests.evaluator_toys import ToyImages

_CONTEXT = CheckContext(task="t", step="s")
_CLEANING = {
    "name": "cleaning",
    "type": "data-cleaning",
    "outlier_method": "zscore",
    "outlier_flags": ["pixel", "visual"],
}
_EVALUATORS = [
    {
        "name": "outliers",
        "type": "quality.outliers",
        "flags": ["pixel", "visual"],
        "outlier_threshold": "zscore",
        "per_target": True,
    },
    {"name": "labels", "type": "quality.label-health"},
]
_OUTLIER_STEPS = [
    {"name": "outliers", "evaluator": "outliers", "input": "data"},
    {"name": "labels", "evaluator": "labels", "input": "data"},
    {"name": "by_class", "combine": "classwise-outliers", "input": "data", "outliers": "outliers"},
    {"name": "image_outliers", "check": "outlier-rate", "input": "outliers"},
    {"name": "target_outliers", "check": "target-outlier-rate", "input": "outliers", "labels": "labels"},
    {"name": "classwise", "check": "classwise-outlier-rate", "input": "by_class"},
]
_OUTLIER_TITLES = {"Image Outliers", "Target Outliers", "Classwise Outliers"}

# 24 classification images, labelled and not; 20 detection images with 28 boxes, a copied image (9 of 4) and two
# bright boxes. On these, data-cleaning finds (severity, title, brief):
#   classification: warning Image Outliers 1 images (4.2%); warning Classwise Outliers worst: b (8.3%), 1/1 classes
#     over 3.0%; warning Duplicates 2 exact (8.3%), 0 near (0.0%); info Label Distribution 2 classes, 24 items,
#     imbalance 1.0:1
#   unlabelled: the same outliers and duplicates, Classwise Outliers info worst: None (0.0%), all classes within
#     3.0%, and no Label Distribution, since no item has a label
#   detection: warning Image Outliers 2 images (10.0%); warning Target Outliers 2 targets (7.1%); warning Classwise
#     Outliers worst: van (12.5%), 1/1 classes over 3.0%; warning Duplicates 2 exact (10.0%), 0 near (0.0%); info
#     Label Distribution 3 classes, 20 items, imbalance 1.3:1
_DATASETS: dict[str, Callable[[], Any]] = {
    "classification": lambda: ToyImages(count=24),
    "unlabelled": lambda: ToyImages(count=24, labeled=False),
    "detection": lambda: ToyDetections(
        [[0, 1], [1], [0], [1, 1], [0]] * 4,
        {0: "car", 1: "van", 2: "bus"},
        duplicate_of={9: 4},
        bright={(3, 0), (8, 1)},
    ),
}


@pytest.fixture(autouse=True)
def _fresh_caches():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _both(steps: list[dict[str, Any]], dataset: Any) -> tuple[list[Finding], list[Finding]]:
    """data-cleaning's findings and the chain's, each run as its own task over the same source."""
    config = chain_pipeline(
        workflows=[_CLEANING, {"name": "judged", "inputs": ["data"], "steps": steps}],
        evaluators=_EVALUATORS,
        tasks=[
            {"name": "legacy", "workflow": "cleaning", "sources": ["src"]},
            {"name": "chain", "workflow": "judged", "sources": ["src"]},
        ],
        datasets={"src": dataset},
    )
    results = run_tasks(config)
    legacy, chain = results["legacy"], results["chain"]
    assert legacy.success, legacy.errors
    assert chain.success, chain.errors
    return legacy.findings, chain.findings  # type: ignore[attr-defined]


def _verdicts(findings: list[Finding], titles: set[str] | None = None) -> list[tuple[str, str, str | None]]:
    return [(f.severity, f.title, f.brief) for f in findings if titles is None or f.title in titles]


def _node(value: Any, items: int | None = None) -> Any:
    """A node as a check reads one: the Output, and the items its Datasets hold."""
    return SimpleNamespace(value=value, items=items)


def _issues(rows: list[tuple[int, int | None]]) -> OutliersOutput[Any]:
    """An Outliers Output flagging `rows` of (item, target); a `None` target is an image-level flag."""
    frame = pl.DataFrame(
        {
            "item_index": [item for item, _ in rows],
            "target_index": [target for _, target in rows],
            "metric_name": ["brightness"] * len(rows),
            "metric_value": [1.0] * len(rows),
        },
        schema={
            "item_index": pl.Int64,
            "target_index": pl.Int64,
            "metric_name": pl.Utf8,
            "metric_value": pl.Float64,
        },
    )
    return OutliersOutput(frame)


@pytest.mark.parametrize("name", sorted(_DATASETS))
def test_the_outlier_checks_agree_with_data_cleaning(name: str) -> None:
    legacy, chain = _both(_OUTLIER_STEPS, _DATASETS[name]())
    assert _verdicts(chain) == _verdicts(legacy, _OUTLIER_TITLES)


def test_the_outlier_checks_judge_nothing_where_their_limits_are_none() -> None:
    steps = [
        *_OUTLIER_STEPS[:3],
        {"name": "image_outliers", "check": "outlier-rate", "input": "outliers", "image": None},
        {
            "name": "target_outliers",
            "check": "target-outlier-rate",
            "input": "outliers",
            "labels": "labels",
            "target": None,
        },
        {"name": "classwise", "check": "classwise-outlier-rate", "input": "by_class", "total": None},
    ]
    _, judged = _both(_OUTLIER_STEPS, _DATASETS["detection"]())
    assert "warning" in {finding.severity for finding in judged}  # the defaults do warn on these
    _, unjudged = _both(steps, _DATASETS["detection"]())
    assert [(f.title, f.severity) for f in unjudged] == [(f.title, "info") for f in judged]


def test_outlier_rate_counts_each_image_once_and_warns_past_its_limit() -> None:
    output = _issues([(1, None), (1, None), (4, None), (6, 0)])
    (finding,) = OutlierRateCheck().run(
        OutlierRateConfig(input="o", image=10.0), {"input": _node(output, 10)}, _CONTEXT
    )
    assert (finding.severity, finding.title, finding.brief) == ("warning", "Image Outliers", "2 images (20.0%)")


def test_outlier_rate_with_nothing_flagged_passes() -> None:
    (finding,) = OutlierRateCheck().run(OutlierRateConfig(input="o"), {"input": _node(_issues([]), 10)}, _CONTEXT)
    assert (finding.severity, finding.brief, finding.description) == (
        "ok",
        "0 images (0.0%)",
        "No images flagged as outliers.",
    )


def test_target_outlier_rate_is_a_share_of_the_labels() -> None:
    labels = LabelHealthOutput({"label_count": 20}, None)
    inputs = {"input": _node(_issues([(0, 0), (0, 0), (2, 1), (3, None)])), "labels": _node(labels)}
    (finding,) = TargetOutlierRateCheck().run(TargetOutlierRateConfig(input="o", labels="l"), inputs, _CONTEXT)
    assert (finding.severity, finding.title, finding.brief) == ("warning", "Target Outliers", "2 targets (10.0%)")


def test_target_outlier_rate_makes_no_finding_without_target_level_outliers() -> None:
    labels = LabelHealthOutput({"label_count": 20}, None)
    inputs = {"input": _node(_issues([(3, None)])), "labels": _node(labels)}
    assert TargetOutlierRateCheck().run(TargetOutlierRateConfig(input="o", labels="l"), inputs, _CONTEXT) == []


_PIVOT = ClasswiseOutliers(
    count_basis="image",
    rows=[ClasswiseRow(class_name="van", count=3, pct=30.0), ClasswiseRow(class_name="car", count=1, pct=5.0)],
    total=ClasswiseRow(class_name="Total", count=4, pct=20.0),
)


def test_classwise_outlier_rate_names_the_worst_class_and_counts_those_over() -> None:
    config = ClasswiseOutlierRateConfig(input="c", total=10.0)
    (finding,) = ClasswiseOutlierRateCheck().run(config, {"input": _node(_PIVOT)}, _CONTEXT)
    assert (finding.severity, finding.title, finding.brief) == (
        "warning",
        "Classwise Outliers",
        "worst: van (30.0%), 1/2 classes over 10.0%",
    )


def test_classwise_outlier_rate_without_a_limit_names_the_worst_and_judges_nothing() -> None:
    config = ClasswiseOutlierRateConfig(input="c", total=None)
    (finding,) = ClasswiseOutlierRateCheck().run(config, {"input": _node(_PIVOT)}, _CONTEXT)
    assert (finding.severity, finding.brief) == ("info", "worst: van (30.0%)")


def test_classwise_outlier_rate_with_nothing_flagged_passes() -> None:
    empty = ClasswiseOutliers(count_basis="image", rows=[], total=None)
    (finding,) = ClasswiseOutlierRateCheck().run(
        ClasswiseOutlierRateConfig(input="c"), {"input": _node(empty)}, _CONTEXT
    )
    assert (finding.severity, finding.brief) == ("ok", "no outliers detected")


def test_classwise_outliers_refuses_outliers_found_on_another_dataset() -> None:
    wanted = "reads `outliers`, which was computed on `b`, not on `a`"
    with pytest.raises(ValidationError, match=wanted):
        chain_pipeline(
            workflows=[
                {
                    "name": "judged",
                    "inputs": ["a", "b"],
                    "steps": [
                        {"name": "outliers", "evaluator": "outliers", "input": "b"},
                        {"name": "by_class", "combine": "classwise-outliers", "input": "a", "outliers": "outliers"},
                    ],
                }
            ],
            evaluators=_EVALUATORS,
            datasets={"src": ToyImages()},
        )


def test_a_threshold_is_a_percentage() -> None:
    with pytest.raises(ValidationError):
        OutlierRateConfig(input="o", image=101.0)


def test_classwise_outliers_refuses_detection_outliers_not_found_per_box() -> None:
    evaluators = [{k: v for k, v in _EVALUATORS[0].items() if k != "per_target"}, _EVALUATORS[1]]
    config = chain_pipeline(
        workflows=[{"name": "judged", "inputs": ["data"], "steps": _OUTLIER_STEPS}],
        evaluators=evaluators,
        tasks=[{"name": "chain", "workflow": "judged", "sources": ["src"]}],
        datasets={"src": _DATASETS["detection"]()},
    )
    result = run_tasks(config)["chain"]
    message = (
        "classwise-outliers counts a detection Dataset's boxes, but `outliers` was not computed per box: "
        "set `per_target: true` on its `quality.outliers` entry."
    )
    by_class = result.steps["by_class"]
    assert by_class.status == "failed"
    assert f"ValueError: {message}" in " ".join(by_class.errors)
    (finding,) = [f for f in result.findings if f.title == "Classwise Outliers"]
    assert (finding.severity, finding.title, finding.brief) == ("info", "Classwise Outliers", "not assessed")
    assert finding.description.startswith(
        "Not assessed: `by_class` failed: ValueError: classwise-outliers counts a detection Dataset's boxes"
    )
    assert result.health["status"] == "failed"
