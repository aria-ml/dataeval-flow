"""data-splitting agrees with what it produced before its port: each part's indices and label counts, the whole's
balance and diversity rows, its class-imbalance verdict, the largest stratification deviation and its verdict, and
each part's uncovered count (data-splitting spec §9).

Deliberate differences from its legacy run (step-chaining spec §10.3 item 3), each with its reason:

- **Titles and briefs are the checks' own:** "Label Distribution", "Stratification", "Uncovered Rate".
- **The split sizes are the split step's section, not findings.**
- **Balance and diversity are sections, not findings,** and are skipped, not fatal, on a dataset with no factors.
- **The cross-split class distribution is the stratification finding's evidence,** not a finding of its own.
- **Stratification is one finding per fold,** not one for the worst fold.
- **Stratification judges the split before rebalancing.** Legacy judged the rebalanced train, so its rebalanced cases
  are compared on indices and counts only.
- **The whole set's coverage is reported,** which legacy did not do; it embeds the whole set once for every part.
- **Adaptive coverage's uncovered count is not judged.** Adaptive coverage marks `int(max(n * percent, 1))` items by
  construction; legacy's findings were `info`, or `warning` on a part under 20 items.
- **Each part's uncovered items are not compared.** Legacy rescaled each embedding dimension over the whole set;
  DataEval's `Coverage` rescales a part's array only when it falls outside the unit interval, so which items are the
  sparsest can differ. The count does not.
- **A coverage step that raises is skipped,** where legacy's whole run failed.
- **The class-imbalance ratio is judged rounded to one place,** as the check rounds it; legacy judged it unrounded.
- **`folds: 1` with `val_frac: 0` runs,** holding out a test only; legacy raised.
- **The task's `metadata:` policy governs** the split and the label, balance and diversity metadata; legacy used
  DataEval's defaults.
- **On detection data each part's label counts are its boxes' labels;** legacy indexed box labels by image index, so
  its per-part counts were wrong there. The golden is classification data.
- **Indices are into the dataset at the bottom of the views;** the golden's source has none, so they agree.
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.steps import ChainResult
from tests.golden.splitting import CASES, pipeline

_GOLDEN = json.loads((Path(__file__).parent / "golden" / "splitting.json").read_text())
_RANK = {"ok": 0, "info": 1, "warning": 2}


def _deviation(brief: str | None) -> float:
    return float(re.search(r"max deviation ([\d.]+)pp", brief or "").group(1))  # type: ignore[union-attr]


def _counts(record: Any) -> dict[str, int]:
    return dict(record.output.data()["label_counts_per_class"])


def _elements(record: Any, keys: list[str] | None) -> list[Any]:
    return [record] if keys is None else [(record.elements or {})[key] for key in keys]


def _produced(result: ChainResult) -> dict[str, Any]:
    steps = result.steps
    indices = (steps["split"].details or {})["indices"]
    keys = list(indices["train"]) if isinstance(indices["train"], dict) else None
    trains = [indices["train"]] if keys is None else [indices["train"][key] for key in keys]
    vals = [indices["val"]] if keys is None else [indices["val"][key] for key in keys]
    train_labels = steps["labels-train"]
    if "rebalance" in steps:
        trains = [(record.details or {}).get("indices") for record in _elements(steps["rebalance"], keys)]
        train_labels = steps["labels-rebalanced"]
    parts = "labels-test" in steps
    return {
        "test": indices["test"] if parts else [],
        "test_counts": _counts(steps["labels-test"]) if parts else {},
        "folds": [
            {"train": train, "val": val, "train_counts": _counts(t), "val_counts": _counts(v)}
            for train, val, t, v in zip(
                trains, vals, _elements(train_labels, keys), _elements(steps["labels-val"], keys), strict=True
            )
        ],
        "full_counts": _counts(steps["labels"]),
        "balance": steps["balance"].output.balance.to_dicts(),
        "diversity": steps["diversity"].output.factors.to_dicts(),
    }


def test_every_case_is_recorded() -> None:
    assert sorted(_GOLDEN) == sorted(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_data_splitting_gives_the_splits_it_gave_before_its_port(name: str) -> None:
    result = run_tasks(pipeline(name, legacy=False))["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    golden = _GOLDEN[name]
    produced = _produced(result)
    assert produced["test"] == golden["test"]
    assert produced["test_counts"] == golden["test_counts"]
    assert produced["full_counts"] == golden["full_counts"]
    assert [{key: fold[key] for key in produced["folds"][0]} for fold in golden["folds"]] == produced["folds"]
    assert produced["balance"] == golden["balance"]
    assert produced["diversity"] == golden["diversity"]
    imbalance = next(finding for finding in result.findings if finding.step == "labels-check")
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


def test_each_part_s_uncovered_count_agrees() -> None:
    result = run_tasks(pipeline("coverage", legacy=False))["t"]
    assert isinstance(result, ChainResult)
    golden = _GOLDEN["coverage"]
    (fold,) = golden["folds"]
    for step, uncovered in [
        ("coverage-train", fold["train_uncovered"]),
        ("coverage-val", fold["val_uncovered"]),
        ("coverage-test", golden["test_uncovered"]),
    ]:
        assert len(result.steps[step].output.uncovered_indices) == len(uncovered), step
