"""audit agrees with what data-analysis produced: each split's outlier, duplicate, class and empty-image counts, the
cross-split duplicate groups, divergence from train, and train's factors (audit spec §12.2).

Deliberate differences from data-analysis (audit spec §12.3), each with its reason. The agreement test sees only the
values below, so the rest are listed to say they are not compared:

- **Chi-square label parity is replaced by `class-stratification`, and label overlap is folded into
  `untrained-classes` and `class-sufficiency`:** the golden records neither.
- **The Bias finding is split into `shortcut-risk` on train plus diversity evidence.** Low diversity no longer warns,
  and legacy's warning on every `balance: true` run, which counted Balance's `class_label` row, is gone.
- **Label Balance's warning on unlabelled images is gone,** and `class-imbalance` does not warn on a declared class with
  no labels.
- **The imbalance ratio is not compared directly.** It is `round(max / min, 1)` over the labelled classes' counts,
  which are compared exactly above; `class-imbalance` exposes the ratio only in its brief and notes, and the test does
  not parse prose.
- **The outlier rate is derived,** not an independent unrounded rate: the compared outlier count over `label-health`'s
  item count, which is checked against each split's size.
- **Rates are not compared as rounded.** data-analysis compared rates rounded to one place, where audit's checks judge
  the unrounded share; the test compares the golden's unrounded rates.
- **audit lists declared classes with no labels** (at 0), which data-analysis omitted; the class counts are compared
  over the classes that have labels.
- **audit computes divergence only for train against each evaluation split,** where data-analysis also did it for
  evaluation pairs; the golden's pairs are not compared.
- **Evaluation splits' per-factor values are not compared:** audit summarises train's factors only. Train's are
  compared in every case that has factors, `detection` included (`ToyDetections` has a `site` factor).
- **Outliers count items, not targets:** audit pins `per_target: false`, as data-analysis counted each flagged
  image once.
- **Duplicate and outlier findings are per split, with quality's wording; `factor-issues`, the verdict, the
  record, grouping by question, next steps and the coverage, completeness, gaps, sufficiency, untrained-classes,
  leakage-by-group, evaluation-coverage and digest checks are new.**
"""

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest

from dataeval_flow import run_tasks
from dataeval_flow.steps import ChainResult
from tests.golden.audit import CASES, pipeline
from tests.golden.rerouting import approximately

_GOLDEN: dict[str, Any] = json.loads((Path(__file__).parent / "golden" / "audit.json").read_text("utf-8"))


def test_every_case_is_recorded() -> None:
    assert sorted(_GOLDEN) == sorted(CASES)


def _run(name: str) -> ChainResult:
    result = run_tasks(pipeline(name, legacy=False))["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


def _output(result: ChainResult, step: str, key: str | None = None) -> Any:
    record = result.steps[step]
    return (record.elements[key] if key else record).output  # type: ignore[index]


def _split(result: ChainResult, kind: str, source: str, train: str) -> Any:
    """The `<kind>` step's output for `source`: `<kind>-train`, or its `<kind>-evals` element."""
    return _output(result, f"{kind}-train") if source == train else _output(result, f"{kind}-evals", source)


def _groups(output: Any, names: list[str], dup_type: str) -> list[dict[str, list[int]]]:
    """`dup_type` groups as legacy stored them (source name to sorted indices), keeping those that span two sources."""
    found = []
    for row in output.data().filter(output.data()["dup_type"] == dup_type).iter_rows(named=True):
        members: dict[str, list[int]] = {}
        for item, dataset in zip(row["item_indices"], row["dataset_indices"], strict=True):
            members.setdefault(names[dataset], []).append(item)
        if len(members) > 1:
            found.append({name: sorted(items) for name, items in members.items()})
    return found


def _canonical(groups: list[dict[str, list[int]]]) -> list[Any]:
    return sorted(sorted((name, sorted(items)) for name, items in group.items()) for group in groups)


@pytest.mark.parametrize("name", sorted(CASES))
def test_audit_agrees_with_data_analysis(name: str) -> None:
    golden, result = _GOLDEN[name], _run(name)
    sources = list(CASES[name].datasets())
    train = sources[0]

    for source, expected in golden["splits"].items():
        outliers = _split(result, "outliers", source, train).data()["item_index"].n_unique()
        duplicates = _split(result, "duplicates", source, train)
        health = _split(result, "label-health", source, train).data()
        assert outliers == expected["outlier_count"], source
        assert health["item_count"] == len(CASES[name].datasets()[source]), source
        assert outliers / health["item_count"] == approximately(expected["outlier_rate"]), source
        assert sum(len(g) for g in duplicates.exact) == expected["exact_duplicates_count"], source
        assert sum(len(items) for items, _ in duplicates.near) == expected["near_duplicates_count"], source
        counts = {c: n for c, n in health["label_counts_per_class"].items() if n}
        assert counts == expected["class_distribution"], source
        assert health["empty_image_count"] == expected["empty_images"], source

    assert sorted(golden["cross"]) == sorted(f"{a}_vs_{b}" for a, b in combinations(sources, 2))
    for pair, expected in golden["cross"].items():
        a, b = pair.split("_vs_")
        other = b if a == train else a
        if train in (a, b):
            output, names = _output(result, "duplicates-cross", other), [train, other]
        else:
            output, names = _output(result, "duplicates-pairs", pair), [a, b]
        assert _canonical(_groups(output, names, "exact")) == _canonical(expected["exact_groups"]), pair
        assert _canonical(_groups(output, names, "near")) == _canonical(expected["near_groups"]), pair
        if train in (a, b):
            divergence = _output(result, "divergence", other)
            value = None if divergence is None else divergence.data()["divergence"]
            assert value == approximately(expected["divergence"]), pair

    if "train_factors" in golden:
        summary = _output(result, "factor-summary").data()["summary"]
        assert json.loads(json.dumps(summary)) == approximately(golden["train_factors"])
