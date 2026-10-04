"""The `untrained-classes` check: evaluation classes train lacks (audit spec §10.2, §5.2)."""

from types import SimpleNamespace
from typing import Any

from dataeval_flow.evaluators.quality import LabelHealthOutput
from dataeval_flow.steps import CheckContext
from dataeval_flow.steps.checks import UntrainedClassesCheck, UntrainedClassesConfig


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


def _judge(evals: Any, **settings: Any) -> Any:
    config = UntrainedClassesConfig(input="labels", evals="evals-labels", **settings)
    (finding,) = UntrainedClassesCheck().run(config, {"input": _TRAIN, "evals": evals}, CheckContext("t", "s"))
    return finding


def test_an_evaluation_class_train_lacks_warns() -> None:
    finding = _judge(_evals(_VAL, _TEST))
    assert finding.severity == "warning"
    assert finding.title == "Untrained Classes"
    assert finding.brief == "1 class in evaluation but not in train"


def test_a_declared_class_in_no_split_does_not_warn_unless_asked() -> None:
    evals = _evals(_health("evals[val]", {"a": 4, "b": 4}))
    assert _judge(evals).severity == "ok"
    asked = _judge(evals, declared=True)
    assert asked.severity == "warning"
    assert "2 declared classes in no split" in asked.brief


def test_with_no_evaluation_split_it_compares_nothing_unless_declared() -> None:
    assert _judge(_evals()).severity == "info"
    assert _judge(_evals()).brief == "no evaluation split to compare"
    declared = _judge(_evals(), declared=True)
    assert declared.severity == "warning"
    assert declared.brief == "no evaluation split, 2 declared classes in no split"
