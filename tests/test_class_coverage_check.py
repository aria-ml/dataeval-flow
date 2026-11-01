"""`class-coverage`: legacy data-coverage's Embedding Coverage finding, judged per class (coverage spec §6.2, §17)."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

from dataeval_flow.steps.checks import ClassCoverageCheck, ClassCoverageConfig

_FLAGS = ("is_clustered", "is_one_dimensional", "has_near_duplicates")


def _row(name: str, *, flag: str | None = None, assessable: bool = True) -> dict[str, Any]:
    """A class row with DataEval's verdicts: `flag` true, the others false, and all null where it is unassessable."""
    verdicts = {key: (key == flag) if assessable else None for key in _FLAGS}
    return {"class": name, "count": 30, "uncovered": 0, "assessable": assessable, **verdicts}


def _judge(rows: list[dict[str, Any]], *, uncovered: int = 1, items: int = 90, on: Any = None) -> Any:
    value = SimpleNamespace(data=lambda: pl.DataFrame(rows), uncovered_indices=np.arange(uncovered))
    node = SimpleNamespace(value=value, items=items, computed_on=(on,) if on is not None else ())
    (finding,) = ClassCoverageCheck().run(ClassCoverageConfig(input="coverage"), {"input": node}, None)  # type: ignore[arg-type]
    return finding


def test_an_uncovered_item_informs() -> None:
    finding = _judge([_row("cat")])
    assert (finding.severity, finding.title, finding.brief) == ("info", "Class Coverage", "1 uncovered (1.1%)")
    assert finding.description == "1 of 90 images uncovered in embedding space."


def test_nothing_uncovered_is_ok() -> None:
    assert _judge([_row("cat")], uncovered=0).severity == "ok"


@pytest.mark.parametrize(
    ("flag", "brief"),
    [
        ("is_clustered", "1 clustered"),
        ("is_one_dimensional", "1 one-dimensional"),
        ("has_near_duplicates", "1 duplicate-padded"),
    ],
)
def test_a_class_dataeval_flags_warns(flag: str, brief: str) -> None:
    finding = _judge([_row("cat", flag=flag)])
    assert finding.severity == "warning"
    assert finding.brief == f"1 uncovered (1.1%) · {brief}"


def test_a_false_or_null_flag_is_no_flag() -> None:
    finding = _judge([_row("cat"), _row("bird", assessable=False)])
    assert (finding.severity, finding.brief) == ("info", "1 uncovered (1.1%)")


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
    ("flag", "note"),
    [
        ("is_clustered", "Clustered (low dispersion): cat."),
        ("is_one_dimensional", "One-dimensional (low isotropy): cat."),
        ("has_near_duplicates", "Duplicate-padded: cat."),
    ],
)
def test_a_flagged_class_is_named_in_its_note(flag: str, note: str) -> None:
    assert _texts(_judge([_row("cat", flag=flag)])) == [note]


def test_two_flagged_classes_are_counted_and_listed() -> None:
    finding = _judge([_row("cat", flag="is_clustered"), _row("dog", flag="is_clustered")])
    assert finding.brief == "1 uncovered (1.1%) · 2 clustered"
    assert _texts(finding) == ["Clustered (low dispersion): cat, dog."]


def test_it_flags_the_classes_dataeval_flags_under_the_factors_coverage_was_given() -> None:
    from dataeval.scope import Coverage

    rng = np.random.default_rng(0)
    spreads = {0: 1.0, 1: 1.0, 2: 0.6, 3: 0.2}  # class 3 collapsed, class 2 a little tight
    embeddings = np.concatenate([rng.normal(cls * 10, spread, (40, 4)) for cls, spread in spreads.items()])
    names = {0: "a", 1: "b", 2: "c", 3: "d"}
    labels: Any = SimpleNamespace(class_labels=np.repeat(list(spreads), 40), index2label=names)
    notes: dict[str, str] = dict(
        zip(_FLAGS, ("Clustered (low dispersion)", "One-dimensional (low isotropy)", "Duplicate-padded"), strict=True)
    )
    clustered = []
    for factor in (0.5, 0.8):
        output = Coverage(dispersion_factor=factor).from_embeddings(embeddings, labels)
        rows = output.data().to_dicts()
        flagged = {flag: [row["class"] for row in rows if row[flag]] for flag in _FLAGS}
        node = SimpleNamespace(value=output, items=len(embeddings), computed_on=())
        (finding,) = ClassCoverageCheck().run(ClassCoverageConfig(input="coverage"), {"input": node}, None)  # type: ignore[arg-type]
        assert _texts(finding) == [f"{notes[flag]}: {', '.join(names)}." for flag, names in flagged.items() if names]
        clustered.append(flagged["is_clustered"])
    assert clustered == [["d"], ["d", "c"]]  # the factor is DataEval's, and the check follows it
