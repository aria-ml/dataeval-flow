"""data-coverage agrees with what it produced before its port: each finding's severity, title, brief and description
where the step reproduces legacy's, and what they were computed from (coverage spec §8.2).

Deliberate differences from its legacy run (step-chaining spec §10.3 item 3), each with its reason:

- **The embedding findings report "not assessed" without an extractor,** where legacy said nothing, and Class
  Coverage does where legacy skipped it for too few items: a check reading a skipped step says why (step-chaining
  §9.1). So does Dimensional Completeness when every box is dropped, where legacy skipped it silently.
- **A source of fewer than two embeddings fails `completeness`,** and Dimensional Completeness reports "not assessed"
  with the step's ValueError, where legacy scored it nan and `ok`. No golden case has one.
- **"Factor Coverage Gaps" reports "not assessed" on a dataset with no metadata factors,** where legacy made no
  finding: `balance` is skipped there, and the gaps read it.
- **Gaps of equal deficit come in value order,** where legacy's order among them was Polars' chance, so the gaps are
  compared with those ties in one order.
- **Legacy's one "Class Coverage" finding is two under `naive`:** `class-coverage`'s, with legacy's brief and
  description, and `uncovered-items`'s. Legacy's severity is the worse of the two. Under `adaptive` the uncovered rate
  is not judged, so legacy's note saying so goes.
- **Naive coverage that overflows is skipped** with "failed: OverflowError", where legacy re-ran it as adaptive; the
  golden's 192 dimensions do not overflow.
- **Skip reasons are the steps' own.**
- **"Metadata Distribution", always info, is the `factor-summary` section,** and balance and diversity are
  sections, as in data-splitting.
- **Class Imbalance's brief, notes and description are the `class-imbalance` check's;** an ImageFolder source's
  finding is titled "Class Imbalance". Its severity agrees.
- **The unlabelled case's Class Shortfall reports "not assessed",** where legacy ran Representation on zero
  counts: `representation` raises on a Dataset with no labels, and the step is optional.
- **Class Shortfall's description points to `label-space`,** where legacy's said "configure an `ontology`",
  which data-coverage now refuses; its ignored-entries note says `expected`.
- **Ontology findings are `label-space`'s;** the summary line and `metadata.has_extractor` go.
- **Names follow the naming pass** (naming spec §3.2): recorded titles are read through `tests/golden/_renames.py`.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.steps import ChainResult
from tests.golden._renames import title as renamed
from tests.golden.coverage import CASES, pipeline
from tests.golden.rerouting import approximately

_GOLDEN: dict[str, Any] = json.loads((Path(__file__).parent / "golden" / "coverage.json").read_text("utf-8"))
_RANK = {"ok": 0, "info": 1, "warning": 2}


def test_every_case_is_recorded() -> None:
    assert sorted(_GOLDEN) == sorted(CASES)


def _run(name: str) -> ChainResult:
    result = run_tasks(pipeline(name, legacy=False))["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


def _left_out(name: str) -> set[str]:
    """The findings whose steps the case's settings leave out of the chain."""
    settings = CASES[name].preset
    left = {"Factor Coverage Gaps"} if "gaps" in settings and settings["gaps"] is None else set()
    return left | ({"Dimensional Completeness"} if settings.get("completeness") is False else set())


def _titles(name: str) -> list[str]:
    """Legacy's finding titles in its order, as the preset makes them: "Metadata Distribution" is a section, "Uncovered
    Items" follows "Class Coverage" under `naive`, and a finding legacy did not make, where the docstring lists
    it, is made "not assessed" in its place unless the settings leave its step out."""
    case, left_out = CASES[name], _left_out(name)
    titles = [renamed(f[1]) for f in _GOLDEN[name]["findings"] if f[1] != "Metadata Distribution"]
    if not case.extractor:  # legacy said nothing of the embeddings
        embedding = ["Class Coverage", "Dimensional Completeness"]
        titles = [title for title in embedding if title not in left_out] + titles
    if case.preset.get("coverage", {}).get("method") == "naive":
        titles.insert(titles.index("Class Coverage") + 1, "Uncovered Items")
    if "Factor Coverage Gaps" not in titles and "Factor Coverage Gaps" not in left_out:  # no metadata factors
        titles.insert(titles.index("Class Shortfall"), "Factor Coverage Gaps")
    return titles


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_findings_agree(name: str) -> None:
    result = _run(name)
    legacy = {renamed(f[1]): [f[0], renamed(f[1]), *f[2:]] for f in _GOLDEN[name]["findings"]}
    preset = {f.title: [f.severity, f.title, f.brief, f.description] for f in result.findings}
    assert [finding.title for finding in result.findings] == _titles(name)
    assert preset["Class Imbalance"][0] == legacy["Class Imbalance"][0]
    coverage = legacy.get("Class Coverage")
    if coverage is None or coverage[2] == "skipped":
        assert preset["Class Coverage"][2] == "not assessed"
    else:
        uncovered = [preset["Uncovered Items"][0]] if "Uncovered Items" in preset else []
        assert max([preset["Class Coverage"][0], *uncovered], key=_RANK.__getitem__) == coverage[0]
        assert preset["Class Coverage"][2:] == coverage[2:]
    for title in ("Dimensional Completeness", "Factor Coverage Gaps"):
        if title in legacy:
            assert preset[title] == legacy[title]
        elif title in _left_out(name):
            assert title not in preset
        else:
            assert preset[title][2] == "not assessed"
    worklist = legacy["Class Shortfall"]
    if name == "unlabelled":
        assert preset["Class Shortfall"][2] == "not assessed"
    else:
        assert preset["Class Shortfall"][:3] == worklist[:3]


@pytest.mark.parametrize("name", sorted(CASES))
def test_what_they_were_computed_from_agrees(name: str) -> None:
    result = _run(name)
    golden = _GOLDEN[name]
    labels = result.steps["labels"].output.data()
    assert labels["label_counts_per_class"] == golden["labels"]["counts"]
    assert labels["empty_image_indices"] == golden["labels"]["empty_images"]
    # Through JSON, as the golden was written: a discrete factor's top values are keyed by number.
    summary = json.loads(json.dumps(result.steps["summary"].output.data()["summary"]))
    assert summary == approximately(golden["summary"])
    if golden["coverage"] is not None:
        _coverage_agrees(result, golden["coverage"])
    if golden["completeness"] is not None:
        data = result.steps["completeness"].output.data()
        # judged to three places; its fourth moves with the lowest declared dependencies' arithmetic
        assert data["completeness"] == pytest.approx(golden["completeness"]["score"], rel=1e-3)
        assert len(data["nearest_neighbor_pairs"]) == golden["completeness"]["pairs"]
    if golden["gaps"] is not None:
        gaps = result.steps["gaps"].output
        assert gaps.mutual_information == approximately(golden["gaps"]["mutual_information"])
        assert [(gap.class_name, gap.factor_name) for gap in gaps.gaps] == [
            (gap["class_name"], gap["factor_name"]) for gap in golden["gaps"]["gaps"]
        ]
        assert _canonical([gap.model_dump() for gap in gaps.gaps]) == approximately(_canonical(golden["gaps"]["gaps"]))
    if golden["worklist"] is not None and name != "unlabelled":
        assert result.steps["worklist"].output.data().to_dicts() == approximately(golden["worklist"])


def _canonical(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The gaps with those of equal deficit in one order, the order among them being a deliberate difference."""
    return sorted(gaps, key=lambda gap: (-gap["deficit"], gap["class_name"], gap["factor_name"], gap["factor_value"]))


def _coverage_agrees(result: ChainResult, golden: dict[str, Any]) -> None:
    output = result.steps["coverage"].output
    rows = [{"class_name": row.pop("class"), **row} for row in output.data().to_dicts()]
    assert rows == approximately(golden["per_class"])
    assert output.coverage_radius == pytest.approx(golden["radius"], rel=1e-4)
    crops = result.steps["crops"]
    assert len(output.critical_value_radii) == golden["observations"]
    if hasattr(crops.output, "item_indices"):
        assert (crops.details or {})["dropped"] == golden["dropped"]
        assert (crops.details or {})["items"] == golden["observations"]
    else:
        assert golden["dropped"] == 0
    crops = crops.output
    mapped = []
    for index, name in zip(output.uncovered_indices, output.uncovered_classes, strict=True):
        index = int(index)
        if hasattr(crops, "item_indices"):
            item, target = int(crops.item_indices[index]), int(crops.target_indices[index])
        else:
            item, target = index, None
        mapped.append([item, target, name, float(output.critical_value_radii[index])])
    assert mapped == approximately(golden["uncovered"])
