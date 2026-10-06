"""data-splitting agrees with what it produced before its port: each part's indices and label counts, the whole's
balance and diversity rows, its class-imbalance verdict, and the largest stratification deviation and its verdict
(data-splitting spec §9).

Deliberate differences from its legacy run (step-chaining spec §10.3 item 3), each with its reason:

- **Titles and briefs are the checks' own:** "Class Imbalance", "Stratification".
- **The split sizes are the split step's section, not findings.**
- **The whole set's balance, diversity and class imbalance are data-bias's:** each case runs a data-bias entry on the
  same source, at legacy's class-imbalance limit, and they are compared there. Balance and diversity are sections,
  not findings.
- **The cross-split class distribution is the stratification finding's evidence,** not a finding of its own.
- **Stratification is one finding per fold,** not one for the worst fold.
- **Stratification judges the split before rebalancing.** Legacy judged the rebalanced train, so its rebalanced cases
  are compared on indices and counts only.
- **Coverage is not computed,** where legacy judged each part's uncovered items: coverage before splitting is
  data-coverage's question, and after it the audit's (audit-as-a-step spec D4). The "coverage" case's recorded
  counts go unchecked.
- **The class-imbalance ratio is judged rounded to one place,** as the check rounds it; legacy judged it unrounded.
- **`folds: 1` with `val_frac: 0` runs,** holding out a test only; legacy raised.
- **The task's `metadata:` policy governs** the split and the label, balance and diversity metadata; legacy used
  DataEval's defaults.
- **On detection data each part's label counts are its boxes' labels;** legacy indexed box labels by image index, so
  its per-part counts were wrong there. The golden is classification data.
- **A declared class with no labels in a split is listed at 0.** `label-health` lists every declared class, at 0 where
  unseen (coverage spec §5.3), so a part's label table, and stratification's, name it. `_same_counts` checks them.
- **Indices are into the dataset at the bottom of the views;** the golden's source has none, so they agree.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.steps import ChainResult
from tests.golden.rerouting import approximately
from tests.golden.splitting import CASES, pipeline

_GOLDEN = json.loads((Path(__file__).parent / "golden" / "splitting.json").read_text())
_RANK = {"ok": 0, "info": 1, "warning": 2}


def _deviation(brief: str | None) -> float:
    return float(re.search(r"max deviation ([\d.]+)pp", brief or "").group(1))  # type: ignore[union-attr]


def _counts(record: Any) -> dict[str, int]:
    return dict(record.output.data()["label_counts_per_class"])


_DECLARED = {"cat", "dog", "bird"}  # the golden dataset's `index2label`


def _same_counts(produced: dict[str, int], golden: dict[str, int]) -> None:
    """The golden's classes agree exactly; the classes it left out are the declared ones, each at 0."""
    if not golden:  # no such part
        assert produced == {}
        return
    assert {name: produced[name] for name in golden} == golden
    extra = {name: count for name, count in produced.items() if name not in golden}
    assert extra == dict.fromkeys(_DECLARED - golden.keys(), 0)


def _elements(record: Any, keys: list[str] | None) -> list[Any]:
    return [record] if keys is None else [(record.elements or {})[key] for key in keys]


def _produced(result: ChainResult, bias: ChainResult) -> dict[str, Any]:
    steps = result.steps
    indices = (steps["split"].details or {})["indices"]
    keys = list(indices["train"]) if isinstance(indices["train"], dict) else None
    trains = [indices["train"]] if keys is None else [indices["train"][key] for key in keys]
    vals = [indices["val"]] if keys is None else [indices["val"][key] for key in keys]
    train_labels = steps["label-health-train"]
    if "rebalanced" in steps:
        trains = [(record.details or {}).get("indices") for record in _elements(steps["rebalanced"], keys)]
        train_labels = steps["label-health-rebalanced"]
    parts = "label-health-test" in steps
    return {
        "test": indices["test"] if parts else [],
        "test_counts": _counts(steps["label-health-test"]) if parts else {},
        "folds": [
            {"train": train, "val": val, "train_counts": _counts(t), "val_counts": _counts(v)}
            for train, val, t, v in zip(
                trains, vals, _elements(train_labels, keys), _elements(steps["label-health-val"], keys), strict=True
            )
        ],
        "full_counts": _counts(steps["label-health"]),
        "balance": bias.steps["balance"].output.balance.to_dicts(),
        "diversity": bias.steps["diversity"].output.factors.to_dicts(),
    }


def test_every_case_is_recorded() -> None:
    assert sorted(_GOLDEN) == sorted(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_data_splitting_gives_the_splits_it_gave_before_its_port(name: str) -> None:
    results = run_tasks(pipeline(name, legacy=False))
    result, bias = results["t"], results["b"]
    assert isinstance(result, ChainResult)
    assert isinstance(bias, ChainResult)
    assert result.success, result.errors
    golden = _GOLDEN[name]
    produced = _produced(result, bias)
    assert produced["test"] == golden["test"]
    _same_counts(produced["test_counts"], golden["test_counts"])
    _same_counts(produced["full_counts"], golden["full_counts"])
    assert len(produced["folds"]) == len(golden["folds"])
    for fold, recorded in zip(produced["folds"], golden["folds"], strict=True):
        assert (fold["train"], fold["val"]) == (recorded["train"], recorded["val"])
        _same_counts(fold["train_counts"], recorded["train_counts"])
        _same_counts(fold["val_counts"], recorded["val_counts"])
    assert produced["balance"] == approximately(golden["balance"])
    assert produced["diversity"] == approximately(golden["diversity"])
    imbalance = next(finding for finding in bias.findings if finding.step == "class-imbalance")
    assert imbalance.severity == golden["class_distribution"]


@pytest.mark.parametrize(
    "name", sorted(name for name, (legacy, _, _) in CASES.items() if "rebalance_method" not in legacy)
)
def test_the_largest_stratification_deviation_and_its_verdict_agree(name: str) -> None:
    result = run_tasks(pipeline(name, legacy=False))["t"]
    assert isinstance(result, ChainResult)
    judged = [finding for finding in result.findings if finding.title == "Stratification"]
    worst = max(judged, key=lambda finding: (_deviation(finding.brief), _RANK[finding.severity]))
    golden = _GOLDEN[name]["stratification"]
    assert (_deviation(worst.brief), worst.severity) == (_deviation(golden["brief"]), golden["severity"])
