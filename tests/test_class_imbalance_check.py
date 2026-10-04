"""`class-imbalance`: every Dataset with classes gets its Label Distribution, judged over the classes with labels,
with an optional info band (coverage spec §5.3)."""

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow._blocks import Paragraph, Section, Table
from dataeval_flow.steps.checks import ClassImbalanceCheck, ClassImbalanceConfig


def _judge(
    counts: dict[str, int],
    *,
    classes: int | None = None,
    items: int = 10,
    unlabelled: Sequence[int] = (),
    on: bool = True,
    **settings: Any,
) -> Any:
    data = {
        "item_count": items,
        "class_count": len(counts) if classes is None else classes,
        "label_count": sum(counts.values()),
        "label_counts_per_class": counts,
        "image_counts_per_class": counts,
        "empty_image_count": len(unlabelled),
        "empty_image_indices": list(unlabelled),
        "label_source": None,
    }
    node = SimpleNamespace(value=SimpleNamespace(data=lambda: data))
    if on:
        node.computed_on = (SimpleNamespace(address="data"),)
    return ClassImbalanceCheck().run(ClassImbalanceConfig(input="labels", **settings), {"input": node}, None)  # type: ignore[arg-type]


def test_an_unlabelled_dataset_that_declares_classes_warns() -> None:
    (finding,) = _judge({"a": 0, "b": 0}, items=6, unlabelled=[0, 1, 2, 3, 4, 5])
    assert (finding.severity, finding.title) == ("warning", "Label Distribution")
    assert any("Classes with no labels: a, b" in getattr(block, "text", "") for block in finding.blocks)


def test_a_dataset_with_neither_classes_nor_labels_makes_no_finding() -> None:
    assert _judge({}, classes=0) == []


def test_the_ratio_is_over_the_classes_with_labels() -> None:
    (finding,) = _judge({"a": 12, "b": 16, "c": 0}, ratio=None)
    assert finding.brief == "3 classes, 10 items, imbalance 1.3:1"
    assert finding.severity == "warning"  # the empty class


@pytest.mark.parametrize(
    ("counts", "severity"),
    [({"a": 10, "b": 10}, "ok"), ({"a": 30, "b": 10}, "info"), ({"a": 60, "b": 10}, "warning")],
)
def test_the_info_band(counts: dict[str, int], severity: str) -> None:
    (finding,) = _judge(counts, ratio=5.0, info=2.0)
    assert finding.severity == severity


def test_without_an_info_band_a_ratio_under_the_limit_informs() -> None:
    (finding,) = _judge({"a": 10, "b": 10})
    assert finding.severity == "info"


def test_info_above_ratio_is_refused() -> None:
    with pytest.raises(ValidationError, match="info"):
        ClassImbalanceConfig(input="labels", ratio=2.0, info=3.0)


def test_its_evidence_has_shares_and_the_empty_images() -> None:
    (finding,) = _judge({"a": 3, "b": 1}, items=6, unlabelled=[4, 5])
    (table,) = [block for block in finding.blocks if isinstance(block, Table)]
    assert [column.header for column in table.columns][:3] == ["Class", "Count", "Share"]
    assert any(isinstance(block, Section) and block.title == "Images with no labels" for block in finding.blocks)
    assert any(
        isinstance(block, Paragraph) and block.text == "Percentages are shares of 4 labels across 6 images."
        for block in finding.blocks
    )


def test_a_node_without_computed_on_still_judges() -> None:
    (finding,) = _judge({"a": 3, "b": 1}, unlabelled=[1], on=False)
    assert finding.severity == "info"


def _texts(finding: Any) -> list[str]:
    return [getattr(block, "text", "") for block in finding.blocks]


def test_with_empty_off_a_class_with_no_labels_does_not_warn() -> None:
    (finding,) = _judge({"a": 4, "b": 0}, empty=False)
    assert finding.severity == "info"
    assert any("Classes with no labels: b" in text for text in _texts(finding))


def test_with_empty_off_the_ratio_still_judges() -> None:
    assert _judge({"a": 40, "b": 4, "c": 0}, empty=False)[0].severity == "warning"
    assert _judge({"a": 4, "b": 0}, empty=False, info=2.0)[0].severity == "ok"
    assert _judge({"a": 0, "b": 0}, empty=False, info=2.0)[0].severity == "info"
