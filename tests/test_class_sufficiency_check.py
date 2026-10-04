"""The `class-sufficiency` check: enough labels per class to learn, and to evaluate (audit spec §10.2, §5.2)."""

from types import SimpleNamespace
from typing import Any

from dataeval_flow.evaluators.quality import LabelHealthOutput
from dataeval_flow.steps import CheckContext
from dataeval_flow.steps.checks import ClassSufficiencyCheck, ClassSufficiencyConfig


def _health(address: str, counts: dict[str, int]) -> SimpleNamespace:
    output = LabelHealthOutput(
        {
            "item_count": sum(counts.values()),
            "class_count": len(counts),
            "label_count": sum(counts.values()),
            "label_counts_per_class": counts,
            "image_counts_per_class": counts,
            "empty_image_count": 0,
            "empty_image_indices": [],
            "label_source": None,
        },
        None,
    )
    return SimpleNamespace(value=output, computed_on=(SimpleNamespace(address=address),), address=f"labels[{address}]")


def _evals(*nodes: SimpleNamespace) -> SimpleNamespace:
    present = {node.computed_on[0].address: node for node in nodes}
    return SimpleNamespace(present=present, elements=present, reason=None)


_TRAIN = _health("train", {"a": 100, "b": 10, "c": 0, "d": 0})
_VAL = _health("evals[val]", {"a": 40, "b": 31, "c": 5})
_TEST = _health("evals[test]", {"a": 40, "b": 5})


def _judge(evals: Any, **limits: Any) -> Any:
    config = ClassSufficiencyConfig(input="labels", evals="evals-labels", **limits)
    (finding,) = ClassSufficiencyCheck().run(config, {"input": _TRAIN, "evals": evals}, CheckContext("t", "s"))
    return finding


def test_a_thin_class_in_train_or_an_evaluation_split_warns() -> None:
    finding = _judge(_evals(_VAL, _TEST))
    assert finding.severity == "warning"
    assert finding.title == "Class Sufficiency"
    assert finding.brief == "1 under 20 in train, 1 under 30 in evals[test]"


def test_with_no_evaluation_split_train_is_still_judged() -> None:
    finding = _judge(_evals())
    assert finding.severity == "warning"
    assert finding.brief == "1 under 20 in train"


def test_with_both_limits_off_it_judges_nothing() -> None:
    finding = _judge(_evals(_VAL, _TEST), train=None, eval=None)
    assert finding.severity == "info"
    assert finding.brief == "2 classes sufficient"
    assert "None" not in finding.description


def test_a_train_with_no_labelled_class_is_not_assessed() -> None:
    config = ClassSufficiencyConfig(input="labels", evals="evals-labels")
    empty = _health("train", {"a": 0})
    (finding,) = ClassSufficiencyCheck().run(config, {"input": empty, "evals": _evals()}, None)
    assert finding.severity == "info"
    assert finding.brief == "not assessed"


def test_a_class_train_holds_that_a_split_lacks_is_thin_there() -> None:
    finding = _judge(_evals(_health("evals[x]", {"a": 40})), train=None)
    assert finding.severity == "warning"
    assert finding.brief == "1 under 30 in evals[x]"
    rows = {row["name"]: row for row in finding.blocks[0].rows}
    assert rows["b"]["s0"] == 0
    assert rows["a"]["s0"] == 40
