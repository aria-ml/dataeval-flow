"""The checks that judge labels against an ontology: legacy data-coverage's findings, as steps (coverage spec §3.4)."""

from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from dataeval_flow.steps.checks import (
    ClassShortfallCheck,
    ClassShortfallConfig,
    LabelConformanceCheck,
    LabelConformanceConfig,
    LeafCoverageCheck,
    LeafCoverageConfig,
    OntologyStructureCheck,
    OntologyStructureConfig,
)

_WORK = ["concept", "label", "parent", "action", "count", "target", "deficit"]


def _representation(
    *,
    leaf: float = 1.0,
    deficit: int = 0,
    worklist: list[dict[str, Any]] = (),  # type: ignore[assignment]
    violations: list[dict[str, Any]] = (),  # type: ignore[assignment]
    dark: list[dict[str, Any]] = (),  # type: ignore[assignment]
    ignored: list[str] = (),  # type: ignore[assignment]
) -> Any:
    frame = pl.DataFrame(list(worklist), schema=_WORK) if worklist else pl.DataFrame(schema=_WORK)
    return SimpleNamespace(
        value=SimpleNamespace(
            data=lambda: frame,
            leaf_coverage=leaf,
            total_deficit=deficit,
            violations=pl.DataFrame(list(violations)) if violations else pl.DataFrame(),
            dark_branches=pl.DataFrame(list(dark)) if dark else pl.DataFrame(),
            ignored_expected=list(ignored),
            ontology_source="inline",
        )
    )


def _leaf(**kwargs: Any) -> Any:
    """The finding on a representation built from `kwargs`; `coverage` and `empty_branches` are the check's fields,
    `leaf` the representation's leaf coverage."""
    settings = {key: kwargs.pop(key) for key in ("coverage", "empty_branches") if key in kwargs}
    (finding,) = LeafCoverageCheck().run(
        LeafCoverageConfig(input="rep", **settings),
        {"input": _representation(**kwargs)},
        None,  # type: ignore[arg-type]
    )
    return finding


_ROW = {"concept": "c", "label": "bus", "parent": "v", "action": "acquire", "count": 0, "target": 4, "deficit": 4}


def test_a_full_spread_is_ok() -> None:
    finding = _leaf()
    assert (finding.severity, finding.title) == ("ok", "Leaf Coverage")
    assert finding.brief == "leaf coverage 100.0% · 0 to acquire · deficit 0"


def test_a_worklist_informs() -> None:
    assert _leaf(worklist=[_ROW], deficit=4, leaf=0.95).severity == "info"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"leaf": 0.5, "worklist": [_ROW]},
        {"dark": [{"concept": "air", "label": "air", "leaves": 2}]},
        {"violations": [{"concept": "c", "label": "bus", "floor": 0.5, "actual": 0.1, "shortfall": 4}]},
    ],
)
def test_low_coverage_an_empty_branch_or_an_unmet_share_warns(kwargs: dict[str, Any]) -> None:
    assert _leaf(**kwargs).severity == "warning"


def test_null_thresholds_turn_off_their_criteria() -> None:
    dark = [{"concept": "a", "label": "air", "leaves": 2}]
    finding = _leaf(leaf=0.5, empty_branches=None, dark=dark, worklist=[_ROW])
    assert finding.severity == "warning"  # coverage is still judged, at its default 0.9
    finding = _leaf(leaf=0.5, coverage=None, empty_branches=None, dark=dark, worklist=[_ROW])
    assert finding.severity == "info"


def test_an_ignored_expected_entry_is_noted_under_its_new_name() -> None:
    texts = [getattr(block, "text", "") for block in _leaf(ignored=["plane"]).blocks]
    assert any(text.startswith("Ignored `expected` entries") and "plane" in text for text in texts)


def _conformance(data: dict[str, Any], **settings: Any) -> Any:
    node = SimpleNamespace(value=SimpleNamespace(data=lambda: data))
    (finding,) = LabelConformanceCheck().run(LabelConformanceConfig(input="rec", **settings), {"input": node}, None)  # type: ignore[arg-type]
    return finding


def test_conforming_names_are_ok() -> None:
    finding = _conformance({"conforms": True, "matched": {"a": "a"}, "unmatched": [], "ambiguous": {}})
    assert (finding.severity, finding.title, finding.brief) == ("ok", "Label Conformance", "conforms")


def test_an_unmatched_or_ambiguous_name_warns() -> None:
    unmatched = _conformance({"conforms": False, "matched": {}, "unmatched": ["truk"], "ambiguous": {}})
    ambiguous = _conformance({"conforms": False, "matched": {}, "unmatched": [], "ambiguous": {"car": ["c1", "c2"]}})
    assert (unmatched.severity, unmatched.brief) == ("warning", "1 unmatched, 0 ambiguous")
    assert ambiguous.severity == "warning"


def test_unmatched_names_within_the_limit_are_ok() -> None:
    data = {"conforms": False, "matched": {}, "unmatched": ["truk"], "ambiguous": {}}
    assert _conformance(data, unmatched=1).severity == "ok"


def test_unmatched_with_no_threshold_is_still_judged_for_ambiguity() -> None:
    """A criterion without a threshold keeps judging: `unmatched=None` lifts only the unmatched limit."""
    unmatched = _conformance({"conforms": False, "matched": {}, "unmatched": ["truk"], "ambiguous": {}}, unmatched=None)
    ambiguous = _conformance(
        {"conforms": False, "matched": {}, "unmatched": ["truk"], "ambiguous": {"car": ["c1", "c2"]}}, unmatched=None
    )
    assert unmatched.severity == "ok"
    assert ambiguous.severity == "warning"


def _structure(**data: Any) -> Any:
    base = {
        "concept_count": 4,
        "leaf_count": 3,
        "max_depth": 1,
        "roots": ["v"],
        "isolated": [],
        "external_ancestors": {},
        "redundant_edges": [],
        "ancestor_siblings": [],
        "unary_parents": [],
        "label_collisions": {},
        "nonconforming_labels": {},
    }
    node = SimpleNamespace(value=SimpleNamespace(data=lambda: base | data))
    (finding,) = OntologyStructureCheck().run(OntologyStructureConfig(input="lint"), {"input": node}, None)  # type: ignore[arg-type]
    return finding


def test_a_clean_ontology_informs() -> None:
    finding = _structure()
    assert (finding.severity, finding.title, finding.brief) == (
        "info",
        "Ontology Structure",
        "4 concepts, 3 leaves, depth 1",
    )


def test_a_label_collision_warns_and_smells_join_the_brief() -> None:
    finding = _structure(label_collisions={"car": ["c1", "c2"]}, unary_parents=["x"])
    assert finding.severity == "warning"
    assert finding.brief == "4 concepts, 3 leaves, depth 1 · 1 single-child links"


def _shortfall(**kwargs: Any) -> Any:
    (finding,) = ClassShortfallCheck().run(
        ClassShortfallConfig(input="rep"),
        {"input": _representation(**kwargs)},
        None,  # type: ignore[arg-type]
    )
    return finding


def test_a_shortfall_worklist_informs() -> None:
    finding = _shortfall(worklist=[_ROW], deficit=4)
    assert (finding.severity, finding.title) == ("info", "Class Shortfall")
    assert finding.brief == "1 classes short · deficit 4"
    assert "`label-space`" in finding.description
    assert "configure an `ontology`" not in finding.description


def test_an_unmet_share_warns_in_a_shortfall() -> None:
    violation = {"concept": "c", "label": "bus", "floor": 0.5, "actual": 0.1, "shortfall": 4}
    finding = _shortfall(worklist=[_ROW], deficit=4, violations=[violation], ignored=["x"])
    assert finding.severity == "warning"
    assert finding.blocks[0].text == "Asserted minimum shares not met: bus (10.0% < 50.0%)."
    assert finding.blocks[1].text == "Ignored `expected` entries (no class has that name): x."
    assert any(type(block).__name__ == "Table" for block in finding.blocks)


def test_nothing_short_is_ok() -> None:
    finding = _shortfall()
    assert (finding.severity, finding.brief) == ("ok", "0 classes short · deficit 0")
