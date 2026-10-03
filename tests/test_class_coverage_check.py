"""`class-coverage`: legacy data-coverage's Embedding Coverage finding, judged per class (coverage spec §6.2, §17)."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

from dataeval_flow.steps.checks import ClassCoverageCheck, ClassCoverageConfig


def _row(
    name: str,
    *,
    dispersion: float | None = 1.0,
    isotropy: float | None = None,
    dup: float | None = 0.0,
    assessable: bool = True,
) -> dict[str, Any]:
    return {
        "class": name,
        "count": 30,
        "uncovered": 0,
        "uncovered_fraction": 0.0,
        "dispersion": dispersion,
        "isotropy": isotropy,
        "near_duplicate_fraction": dup,
        "assessable": assessable,
    }


def _judge(rows: list[dict[str, Any]], *, uncovered: int = 1, items: int = 90, on: Any = None, **settings: Any) -> Any:
    value = SimpleNamespace(data=lambda: pl.DataFrame(rows), uncovered_indices=np.arange(uncovered))
    node = SimpleNamespace(value=value, items=items, computed_on=(on,) if on is not None else ())
    config = ClassCoverageConfig(input="coverage", **settings)
    (finding,) = ClassCoverageCheck().run(config, {"input": node}, None)  # type: ignore[arg-type]
    return finding


def test_an_uncovered_item_informs() -> None:
    finding = _judge([_row("cat")])
    assert (finding.severity, finding.title, finding.brief) == ("info", "Embedding Coverage", "1 uncovered (1.1%)")
    assert finding.description == "1 of 90 images uncovered in embedding space."


def test_nothing_uncovered_is_ok() -> None:
    assert _judge([_row("cat")], uncovered=0).severity == "ok"


@pytest.mark.parametrize(
    ("row", "flag"),
    [
        (_row("cat", dispersion=0.2), "1 clustered"),
        (_row("cat", isotropy=0.1), "1 one-dimensional"),
        (_row("cat", dup=0.5), "1 duplicate-padded"),
    ],
)
def test_a_clustered_flat_or_padded_class_warns(row: dict[str, Any], flag: str) -> None:
    finding = _judge([row])
    assert finding.severity == "warning"
    assert finding.brief == f"1 uncovered (1.1%) · {flag}"


def test_a_criterion_set_to_null_is_off() -> None:
    assert _judge([_row("cat", dispersion=0.2)], dispersion=None).severity == "info"


def test_an_unassessable_class_is_not_judged() -> None:
    assert _judge([_row("bird", dispersion=0.0, assessable=False)]).severity == "info"


def test_on_crops_it_counts_detection_crops_and_notes_the_dropped() -> None:
    from dataeval.data import DetectionCrops

    from tests.golden.coverage import CoverageDetections

    crops = DetectionCrops(CoverageDetections(), min_size=4)  # type: ignore[arg-type]
    finding = _judge([_row("car")], items=len(crops), on=SimpleNamespace(value=crops))
    assert finding.description == f"1 of {len(crops)} detection crops uncovered in embedding space."
    texts = [getattr(block, "text", "") for block in finding.blocks]
    assert any("one per ground-truth box" in text for text in texts)
    assert any(f"{crops.n_dropped} detection(s) were too small" in text for text in texts)


def _texts(finding: Any) -> list[str]:
    return [block.text for block in finding.blocks]


@pytest.mark.parametrize(
    ("row", "note"),
    [
        (_row("cat", dispersion=0.2), "Clustered (low dispersion): cat."),
        (_row("cat", isotropy=0.1), "One-dimensional (low isotropy): cat."),
        (_row("cat", dup=0.5), "Duplicate-padded: cat."),
    ],
)
def test_a_flagged_class_is_named_in_its_note(row: dict[str, Any], note: str) -> None:
    assert _texts(_judge([row])) == [note]


@pytest.mark.parametrize(
    "row",
    [_row("cat", dispersion=0.5), _row("cat", dispersion=1.0, isotropy=0.5), _row("cat", dup=0.1)],
)
def test_a_class_exactly_at_a_limit_is_not_flagged(row: dict[str, Any]) -> None:
    finding = _judge([row])
    assert (finding.severity, finding.brief) == ("info", "1 uncovered (1.1%)")


def test_two_flagged_classes_are_counted_and_listed() -> None:
    finding = _judge([_row("cat", dispersion=0.2), _row("dog", dispersion=0.3)])
    assert finding.brief == "1 uncovered (1.1%) · 2 clustered"
    assert _texts(finding) == ["Clustered (low dispersion): cat, dog."]
