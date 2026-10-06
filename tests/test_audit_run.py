"""audit, run end to end on toy data: what it refuses at preflight, and its verdict on one, two and three splits (audit
spec §4.1, §6.3, §13, §14).

`ToyImages` plants an exact duplicate and a white outlier, and has no metadata factors. Two `ToyImages` of more than
seven items share that white image, so they leak; `_leaky()` builds a test split sharing exactly one train image.
`ToyFactors` has the factors `site` and `angle`, and every instance draws the same images.
"""

from typing import Any

import pytest

from dataeval_flow import dataset_digest, run_tasks
from dataeval_flow._blocks import Fields, Table
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._graph import GraphError
from dataeval_flow._chain._verdict import Acceptance, Verdict, moot_checks, next_step_lines
from dataeval_flow.config import PipelineConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.steps._registry import get_check
from dataeval_flow.workflows.audit import AuditConfig, AuditWorkflow
from dataeval_flow.workflows.audit._workflow import NO_EVALUATION_SPLIT, PER_EVALUATION_SPLIT
from tests.chain_toys import ToyDetections, chain_pipeline
from tests.evaluator_toys import Items, ToyFactors, ToyImages
from tests.test_audit_preset import _HEADINGS, _OUTLIERS
from tests.test_chain_verdict_report import _outline, _section, _top


@pytest.fixture(autouse=True)
def _fresh_caches():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _pipeline(datasets: dict[str, Any], entry: dict[str, Any] | None = None, *, extractor: bool) -> PipelineConfig:
    task: dict[str, Any] = {"name": "t", "workflow": "w", "sources": list(datasets)}
    if extractor:
        task["extractor"] = "flat"
    workflow = {"name": "w", "type": "audit", **_OUTLIERS, **(entry or {})}
    return chain_pipeline(workflows=[workflow], tasks=[task], datasets=datasets, extractor=extractor)


def _audit(datasets: dict[str, Any], entry: dict[str, Any] | None = None, *, extractor: bool = False) -> ChainResult:
    result = run_tasks(_pipeline(datasets, entry, extractor=extractor))["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


def _verdict(result: ChainResult) -> Verdict:
    assert result.verdict is not None
    return result.verdict


def _next_steps(result: ChainResult) -> list[str]:
    assert result.preset_chain is not None
    return next_step_lines(_verdict(result), result.preset_chain.next_steps)


def _leaky() -> dict[str, Any]:
    """Train, and a test split whose first item is train's item 3, then seven fresh items planting nothing."""
    train, fresh = ToyImages(), ToyImages(7, seed=1)
    return {"train": train, "test": Items([train[3], *(fresh[i] for i in range(len(fresh)))])}


def _three() -> dict[str, Any]:
    return {"train": ToyImages(), "val": ToyImages(seed=1), "test": ToyImages(seed=2)}


def _detections(count: int, dataset_id: str) -> ToyDetections:
    return ToyDetections([[0] if i % 2 == 0 else [1, 0] for i in range(count)], {0: "a", 1: "b"}, dataset_id=dataset_id)


def test_a_one_split_audit_is_ready_with_caveats() -> None:
    result = _audit({"train": ToyImages()}, extractor=True)
    verdict = _verdict(result)
    groups = AuditWorkflow.chain(AuditConfig.model_validate({"name": "w", **_OUTLIERS})).groups
    clean, labels, splits = groups[0], groups[1], groups[-1]
    assert splits.heading == "Are the splits fit to evaluate on?"
    assert sorted((item.check, item.reason) for item in verdict.not_assessed if item.check in splits.checks) == sorted(
        (check, NO_EVALUATION_SPLIT) for check in splits.checks
    )
    # A per-split check over the evaluation splits judges nothing, but the same check over train judges train.
    assert not [item for item in verdict.not_assessed if item.check in (*clean.checks, *labels.checks)]
    assert _section(_top(result), clean.heading).brief == "2 warnings"
    over_evals = [record for name, record in result.steps.items() if record.kind == "check" and name.endswith("-evals")]
    assert over_evals
    assert {record.not_assessed for record in over_evals} == {NO_EVALUATION_SPLIT}
    assert moot_checks(result.steps) == {record.name for record in over_evals}
    assert verdict.blocking == []
    assert verdict.level == "ready-with-caveats"
    assert [line for line in _next_steps(result) if line.startswith("Give an evaluation split")]
    sufficiency = result.steps["class-sufficiency"]
    assert sufficiency.not_assessed is None
    assert [(finding.severity, finding.brief) for finding in sufficiency.output] == [("warning", "2 under 20 in train")]


def test_an_audit_with_no_extractor_names_the_unassessed_checks() -> None:
    result = _audit(_leaky(), {"coverage": {"method": "naive"}})
    embedded = ["eval-coverage", "distribution-shift", "class-coverage", "uncovered-items", "dimensional-completeness"]
    reasons = {item.check: item.reason for item in _verdict(result).not_assessed if item.check in embedded}
    assert sorted(reasons) == sorted(embedded)
    assert all(reason.endswith("was skipped: requires an extractor") for reason in reasons.values())
    (line,) = [line for line in _next_steps(result) if line.startswith("Name an extractor")]
    steps = {
        "eval-coverage": "eval-coverage[test]",
        "distribution-shift": "distribution-shift[test]",
        "class-coverage": "class-coverage",
        "uncovered-items": "uncovered-items",
        "dimensional-completeness": "dimensional-completeness",
    }
    titles = ", ".join(f"{get_check(check).title} ({step})" for check, step in steps.items())
    assert line == f"Name an extractor to assess these checks. Not assessed: {titles}."


def test_an_audit_over_data_with_no_factors_still_gives_a_verdict() -> None:
    result = _audit({"train": ToyImages(), "test": ToyImages(seed=1)})
    for step in ("balance", "diversity", "factor-gaps"):
        assert result.steps[step].status == "skipped"
    unassessed = {item.check for item in _verdict(result).not_assessed}
    assert {"factor-coverage-gaps", "shortcut-risk"} <= unassessed
    shortcut = _section(_top(result), "Could the model learn a shortcut?")
    assert shortcut.brief == "not assessed: no factors found in provided metadata"
    advice = "Name a metadata policy, or add metadata factors, to assess these checks."
    gaps = "Factor Coverage Gaps (factor-coverage-gaps)"
    assert f"{advice} Not assessed: Shortcut Risk (shortcut-risk), {gaps}." in _next_steps(result)


def test_two_splits_sharing_an_image_are_not_ready() -> None:
    result = _audit(_leaky())
    assert [finding.severity for finding in result.steps["leakage"].output] == ["warning"]
    verdict = _verdict(result)
    assert verdict.level == "not-ready"
    assert verdict.blocking[0].check == "leakage"


def test_an_accepted_blocking_warning_is_ready_with_caveats() -> None:
    result = _audit(_leaky(), {"accepted": {"leakage": "Shared calibration frames."}})
    verdict = _verdict(result)
    assert verdict.blocking == []
    assert verdict.accepted == [Acceptance(check="leakage", reason="Shared calibration frames.", state="warned")]
    assert verdict.level == "ready-with-caveats"


@pytest.mark.parametrize("empty", ["train", "test"])
def test_a_split_with_no_items_is_refused_naming_it(empty: str) -> None:
    datasets = {"train": ToyImages(), "test": ToyImages(seed=1)} | {empty: Items([])}
    message = f"Split `{empty}` holds no items; an audit judges only splits with data."
    with pytest.raises(GraphError, match=message):
        run_tasks(_pipeline(datasets, extractor=False))


@pytest.mark.parametrize(
    ("datasets", "message"),
    [
        (
            {"train": ToyImages(), "test": _detections(12, "det")},
            "`train` is classification, `evals` is object_detection",
        ),
        (
            {"train": ToyImages(), "val": ToyImages(seed=1), "test": _detections(12, "det")},
            "An audit's splits must be one kind: train: classification, val: classification, test: object_detection.",
        ),
    ],
    ids=["train-against-evals", "among-evals"],
)
def test_splits_of_different_kinds_are_refused(datasets: dict[str, Any], message: str) -> None:
    with pytest.raises(GraphError, match=message):
        run_tasks(_pipeline(datasets, extractor=False))


def test_detection_splits_are_audited_with_crops_before_coverage() -> None:
    result = _audit({"train": _detections(40, "det-train"), "test": _detections(12, "det-test")}, extractor=True)
    order = list(result.steps)
    assert order.index("ood-kneighbors") < order.index("crops") < order.index("coverage")
    crops, coverage = result.steps["crops"], result.steps["coverage"]
    assert (crops.status, len(crops.output)) == ("ok", 60)  # one crop per box
    assert (coverage.status, coverage.inputs) == ("ok", ["crops"])


def test_three_splits_pair_their_evaluation_splits() -> None:
    result = _audit(_three())
    pairs, stratification = result.steps["duplicates-pairs"], result.steps["stratification"]
    assert pairs.elements is not None
    assert list(pairs.elements) == ["val_vs_test"]
    assert stratification.elements is not None
    assert {key: len(run.output) for key, run in stratification.elements.items()} == {"val": 1, "test": 1}


def test_the_record_gives_every_split_its_digests() -> None:
    datasets = _three()
    record = _section(_top(_audit(datasets)), "What was audited")
    table, digests = record.blocks[:2]
    assert isinstance(table, Table)
    assert isinstance(digests, Fields)
    assert [column.header for column in table.columns] == ["", "train", "val", "test"]
    (row,) = [row for row in table.rows if row[""] == "Content digest"]
    full = dict(digests.items)
    for name, dataset in datasets.items():
        content = dataset_digest(dataset).content
        assert full[f"Content digest ({name})"] == content
        assert row[name] == f"{content[:12]}…"


def test_the_report_reads_verdict_record_five_questions_next_steps() -> None:
    outline = _outline(_audit(_three()))
    assert outline[: outline.index("Next steps") + 1] == [
        "verdict",
        "Verdict",
        "What was audited",
        *_HEADINGS,
        "Next steps",
    ]


def test_every_factor_step_reads_one_encoding() -> None:
    # Encoded on its own, ToyFactors(5)'s angles (0..4) cut differently from ToyFactors(60)'s.
    result = _audit({"train": ToyFactors(60), "val": ToyFactors(30), "test": ToyFactors(5)})
    digest = result.metadata.encoding_digest
    assert digest is not None
    binning = result.metadata.metadata_binning
    assert binning is not None
    assert {split: record["encoding_digest"] for split, record in binning["per_split"].items()} == dict.fromkeys(
        ["train", "evals[val]", "evals[test]"], digest
    )


def test_an_acceptance_keyed_by_one_split_s_step_leaves_the_other_split_s_warning() -> None:
    result = _audit(_three(), {"accepted": {"image-outliers-evals[test]": "Night shots, by design."}})
    steps = {item.step for item in _verdict(result).warnings if item.check == "image-outliers"}
    assert "image-outliers-evals[val]" in steps
    assert "image-outliers-evals[test]" not in steps


def test_the_steps_that_take_a_split_are_the_checks_run_once_per_evaluation_split() -> None:
    # An extractor and an ontology, so every check over the evaluation splits runs.
    result = _audit(_three(), {"ontology": {"a": {}, "b": {}}}, extractor=True)
    per_split = {name for name, record in result.steps.items() if record.kind == "check" and record.elements}
    assert per_split == PER_EVALUATION_SPLIT


def test_an_acceptance_naming_no_split_of_the_task_is_refused_before_the_run() -> None:
    config = _pipeline(_three(), {"accepted": {"image-outliers-evals[tset]": "x"}}, extractor=False)
    with pytest.raises(GraphError, match=r"`image-outliers-evals\[tset\]`, but this task has no evaluation split"):
        run_tasks(config)


@pytest.mark.parametrize("element", ["train", "val_vs_test"])
def test_an_acceptance_naming_train_or_a_pair_is_refused_before_the_run(element: str) -> None:
    config = _pipeline(_three(), {"accepted": {f"image-outliers-evals[{element}]": "x"}}, extractor=False)
    with pytest.raises(GraphError, match=rf"has no evaluation split `{element}`"):
        run_tasks(config)
