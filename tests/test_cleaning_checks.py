"""data-cleaning's findings, made by checks: each agrees with the finding it replaces (spec §9.2, §10.3)."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from dataeval.quality import DuplicatesOutput, OutliersOutput
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.evaluators.quality import LabelHealthOutput
from dataeval_flow.steps import ChainResult, CheckContext
from dataeval_flow.steps.checks import (
    ClassImbalanceCheck,
    ClassImbalanceConfig,
    ClasswiseOutlierRateCheck,
    ClasswiseOutlierRateConfig,
    DuplicateRateCheck,
    DuplicateRateConfig,
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
    {"name": "dupes", "type": "quality.duplicates", "merge_near_duplicates": True},
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
#   no duplicates (6 images, none copied): ok Image Outliers 0 images (0.0%); ok Classwise Outliers no outliers
#     detected; info Label Distribution 2 classes, 6 items, imbalance 1.0:1; and no Duplicates finding at all
_DATASETS: dict[str, Callable[[], Any]] = {
    "classification": lambda: ToyImages(count=24),
    "no duplicates": lambda: ToyImages(count=6),
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
    assert isinstance(result, ChainResult)
    message = (
        "classwise-outliers counts a detection Dataset's boxes, but `outliers` was not computed per box: "
        "set `per_target: true` on its `quality.outliers` entry."
    )
    by_class = result.steps["by_class"]
    assert by_class.status == "failed"
    assert f"ValueError: {message}" in " ".join(by_class.errors)
    (finding,) = [f for f in result.findings if f.title == "Classwise Outliers"]
    assert (finding.severity, finding.title, finding.brief) == ("info", "Classwise Outliers", "not assessed")
    assert finding.description is not None
    assert finding.description.startswith(
        "Not assessed: `by_class` failed: ValueError: classwise-outliers counts a detection Dataset's boxes"
    )
    assert result.health["status"] == "failed"


_ALL_STEPS = [
    *_OUTLIER_STEPS[:3],
    {"name": "dupes", "evaluator": "dupes", "input": "data"},
    *_OUTLIER_STEPS[3:],
    {"name": "duplicates", "check": "duplicate-rate", "input": "dupes"},
    {"name": "imbalance", "check": "class-imbalance", "input": "labels"},
]


@pytest.mark.parametrize("name", sorted(_DATASETS))
def test_data_cleaning_s_whole_report_agrees_as_a_chain(name: str) -> None:
    legacy, chain = _both(_ALL_STEPS, _DATASETS[name]())
    assert _verdicts(chain) == _verdicts(legacy)
    if name == "no duplicates":
        assert "Duplicates" not in {f.title for f in [*legacy, *chain]}
    assert {finding.step for finding in chain} <= {step["name"] for step in _ALL_STEPS if "check" in step}


def test_the_duplicate_and_label_checks_judge_nothing_where_their_limits_are_none() -> None:
    steps = [
        *_ALL_STEPS[:-2],
        {"name": "duplicates", "check": "duplicate-rate", "input": "dupes", "exact": None, "near": None},
        {"name": "imbalance", "check": "class-imbalance", "input": "labels", "ratio": None},
    ]
    _, chain = _both(steps, _DATASETS["detection"]())
    assert {f.title: f.severity for f in chain if f.title in {"Duplicates", "Label Distribution"}} == {
        "Duplicates": "info",
        "Label Distribution": "info",
    }


def _groups(rows: list[tuple[str, str, list[int]]]) -> DuplicatesOutput[Any, Any]:
    """A Duplicates Output of `rows`: each (level, dup_type, item indices)."""
    frame = pl.DataFrame(
        {
            "group_id": list(range(len(rows))),
            "level": [level for level, _, _ in rows],
            "dup_type": [kind for _, kind, _ in rows],
            "item_indices": [items for _, _, items in rows],
            "methods": [["xxhash"]] * len(rows),
        },
        schema={
            "group_id": pl.Int64,
            "level": pl.Utf8,
            "dup_type": pl.Utf8,
            "item_indices": pl.List(pl.Int64),
            "methods": pl.List(pl.Utf8),
        },
    )
    return DuplicatesOutput(frame)


def test_duplicate_rate_counts_each_group_s_members_as_a_share_of_the_dataset() -> None:
    output = _groups([("item", "exact", [0, 5]), ("item", "near", [1, 2, 3]), ("target", "exact", [4, 4])])
    (finding,) = DuplicateRateCheck().run(DuplicateRateConfig(input="d"), {"input": _node(output, 20)}, _CONTEXT)
    assert (finding.severity, finding.title, finding.brief) == (
        "warning",
        "Duplicates",
        "2 exact (10.0%), 3 near (15.0%)",
    )
    assert finding.description == "1 exact duplicate groups, 1 near-duplicate groups found."


def test_duplicate_rate_makes_no_finding_without_duplicate_items() -> None:
    output = _groups([("target", "exact", [4, 4])])
    assert DuplicateRateCheck().run(DuplicateRateConfig(input="d"), {"input": _node(output, 20)}, _CONTEXT) == []


def test_duplicate_rate_within_both_limits_is_information() -> None:
    output = _groups([("item", "near", [1, 2])])
    config = DuplicateRateConfig(input="d", exact=0.0, near=50.0)
    (finding,) = DuplicateRateCheck().run(config, {"input": _node(output, 20)}, _CONTEXT)
    assert (finding.severity, finding.brief) == ("info", "0 exact (0.0%), 2 near (10.0%)")


def _labels(counts: dict[str, int], *, classes: int, items: int, source: str | None = None) -> LabelHealthOutput:
    data = {
        "item_count": items,
        "class_count": classes,
        "label_count": sum(counts.values()),
        "label_counts_per_class": counts,
        "image_counts_per_class": counts,
        "empty_image_count": 0,
        "label_source": source,
    }
    return LabelHealthOutput(data, None)


def test_class_imbalance_is_the_largest_class_over_the_smallest() -> None:
    labels = _labels({"car": 12, "van": 2}, classes=3, items=14)
    (finding,) = ClassImbalanceCheck().run(ClassImbalanceConfig(input="l"), {"input": _node(labels)}, _CONTEXT)
    assert (finding.severity, finding.title, finding.brief) == (
        "warning",
        "Label Distribution",
        "3 classes, 14 items, imbalance 6.0:1",
    )


def test_class_imbalance_names_labels_read_from_file_paths() -> None:
    labels = _labels({"car": 4, "van": 4}, classes=2, items=8, source="filepath")
    (finding,) = ClassImbalanceCheck().run(ClassImbalanceConfig(input="l"), {"input": _node(labels)}, _CONTEXT)
    assert (finding.severity, finding.title, finding.brief) == (
        "info",
        "Label/Directory_Name Distribution",
        "2 classes, 8 items, imbalance 1.0:1",
    )


def test_class_imbalance_makes_no_finding_where_no_item_has_a_label() -> None:
    labels = _labels({}, classes=2, items=6)
    assert ClassImbalanceCheck().run(ClassImbalanceConfig(input="l"), {"input": _node(labels)}, _CONTEXT) == []


def test_class_imbalance_warns_on_a_class_with_no_labels_even_without_a_ratio_limit() -> None:
    labels = _labels({"car": 4, "van": 0}, classes=2, items=4)
    config = ClassImbalanceConfig(input="l", ratio=None)
    (finding,) = ClassImbalanceCheck().run(config, {"input": _node(labels)}, _CONTEXT)
    assert (finding.severity, finding.brief) == ("warning", "2 classes, 4 items, imbalance 0.0:1")
