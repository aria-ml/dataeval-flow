"""audit run as a step of a custom workflow: its preflight, its reference encoding, and (Task 4) its verdict, record and
questions on the task's result (audit-as-a-step spec §4)."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow._blocks import Fields, Paragraph, Table
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._presets import SpliceRun
from dataeval_flow._runner import run
from dataeval_flow.config import PipelineConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import ChainResult
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import Items, ToyFactors, ToyImages
from tests.golden.splitting import SplitImages
from tests.test_audit_preset import _HEADINGS, _OUTLIERS
from tests.test_audit_run import _audit, _leaky, _three
from tests.test_chain_verdict_report import _outline, _section, _top


@pytest.fixture(autouse=True)
def _fresh_caches():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


_AUDIT = {"name": "release", "type": "audit", **_OUTLIERS}
_COLLECT_AUDIT = [
    {"name": "evals", "transform": "collect", "input": ["splits.val", "splits.test"]},
    {"name": "audit", "workflow": "release", "input": ["splits.train", "evals"]},
]


def _split_pipeline(
    steps: list[dict[str, Any]] | None = None,
    *,
    splitting: dict[str, Any] | None = None,
    audit: dict[str, Any] | None = None,
    extractor: bool = False,
    extra: dict[str, Any] | None = None,
) -> PipelineConfig:
    """One source `data`, split by a data-splitting step `splits`, then `steps` (by default collect, then audit)."""
    task: dict[str, Any] = {"name": "t", "workflow": "outer", "sources": ["data"]}
    if extractor:
        task["extractor"] = "flat"
    return chain_pipeline(
        workflows=[
            {"name": "splitting", "type": "data-splitting", **(splitting or {})},
            {**_AUDIT, **(audit or {})},
            {
                "name": "outer",
                "inputs": ["data"],
                "steps": [{"name": "splits", "workflow": "splitting", "input": "data"}, *(steps or _COLLECT_AUDIT)],
            },
        ],
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[task],
        datasets={"data": SplitImages(90)},
        extractor=extractor,
        extra={"seed": 0, **(extra or {})},
    )


def _run(config: PipelineConfig) -> ChainResult:
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    return result


def _audited_digests(result: ChainResult, addresses: tuple[str, ...]) -> set[str]:
    """The encoding digests the binning record holds for the Datasets at `addresses`."""
    binning = result.metadata.metadata_binning
    assert binning is not None
    per_split = binning["per_split"]
    digests = {key.partition(" (")[0]: record["encoding_digest"] for key, record in per_split.items()}
    missing = [address for address in addresses if address not in digests]
    assert not missing, f"no binning record for {missing}; recorded: {sorted(digests)}"
    return {digests[address] for address in addresses}


def test_a_spliced_audit_encodes_every_part_like_train() -> None:
    result = _run(_split_pipeline())
    assert result.success, result.errors
    assert len(_audited_digests(result, ("splits/split.train", "evals[val]", "evals[test]"))) == 1


def test_an_empty_part_fails_the_splice_through_the_preset_s_preflight() -> None:
    config = chain_pipeline(
        workflows=[
            _AUDIT,
            {
                "name": "outer",
                "inputs": ["train", {"name": "evals", "list": True}],
                "steps": [
                    {"name": "first", "evaluator": "dupes", "input": "train"},
                    {"name": "audit", "workflow": "release", "input": ["train", "evals"]},
                ],
            },
        ],
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["train", "val", "test"]}],
        datasets={"train": ToyImages(), "val": Items([]), "test": ToyImages(seed=2)},
    )
    result = _run(config)
    assert not result.success
    assert result.steps["first"].status == "ok"
    first_audit = next(name for name in result.steps if name.startswith("audit/"))
    assert result.steps[first_audit].status == "failed"
    assert "Step 'audit': Split `val` holds no items" in result.steps[first_audit].errors[0]
    later = [record for name, record in result.steps.items() if name.startswith("audit/") and name != first_audit]
    assert later
    assert all(record.status == "skipped" for record in later)
    assert all("`audit` failed as it started" in (record.reason or "") for record in later)
    assert sum("holds no items" in error for error in result.errors) == 1


def test_a_chain_made_train_with_no_items_is_named_by_its_slot() -> None:
    config = chain_pipeline(
        workflows=[
            _AUDIT,
            {
                "name": "outer",
                "inputs": ["train", "test"],
                "steps": [
                    {"name": "kept", "transform": "collect", "input": ["train"]},
                    {"name": "evals", "transform": "collect", "input": ["test"]},
                    {"name": "audit", "workflow": "release", "input": ["kept[train]", "evals"]},
                ],
            },
        ],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["train", "test"]}],
        datasets={"train": Items([]), "test": ToyImages(seed=2)},
    )
    result = _run(config)
    first_audit = next(name for name in result.steps if name.startswith("audit/"))
    assert "Split `train` holds no items" in result.steps[first_audit].errors[0]


def test_a_fold_list_bound_to_a_single_slot_is_refused_at_load() -> None:
    with pytest.raises(ValidationError, match=r"name one element, such as `splits\.train\[0\]`"):
        _split_pipeline(
            [
                {"name": "evals", "transform": "collect", "input": ["splits.test"]},
                {"name": "audit", "workflow": "release", "input": ["splits.train", "evals"]},
            ],
            splitting={"folds": 3},
        )


def test_reference_split_under_a_spliced_audit_is_refused_at_load() -> None:
    with pytest.raises(ValidationError, match="reference_split"):
        _split_pipeline(
            audit={"metadata": "pinned"},
            extra={"metadata": [{"name": "pinned", "reference_split": "data"}]},
        )


def test_a_dataset_made_from_the_reference_alone_reads_under_the_step_s_own_policy() -> None:
    # D7: what a splice's step makes from its reference alone, as `crops` from train, reads as the reference does;
    # what it makes from the reference and another Dataset reads under the derived policy.
    from dataeval_flow._chain._nodes import Node
    from dataeval_flow._chain._run import StepContext, _grow
    from dataeval_flow.steps._port import DataType

    crops = Node("audit/crops", DataType.DATASET, inputs=("splits/split.train",))
    mixed = Node("mixed", DataType.DATASET, inputs=("splits/split.train", "evals[val]"))
    alone = {"splits/split.train"}
    _grow(alone, {crops.address: crops, mixed.address: mixed})
    assert alone == {"splits/split.train", "audit/crops"}
    own: Any = object()
    derived: Any = object()
    context = StepContext(metadata_policy=own, derived_policy=derived, reference_addresses=frozenset(alone))
    assert context.policy_for(crops) is own
    assert context.policy_for(mixed) is derived


def test_collected_evals_beside_a_source_bound_train_are_not_refused_for_kind() -> None:
    # Spec §4.2: the one-kind check passes on chain-made Datasets, which carry no kind; check_kinds has judged them.
    config = chain_pipeline(
        workflows=[
            _AUDIT,
            {
                "name": "outer",
                "inputs": ["train", "val", "test"],
                "steps": [
                    {"name": "evals", "transform": "collect", "input": ["val", "test"]},
                    {"name": "audit", "workflow": "release", "input": ["train", "evals"]},
                ],
            },
        ],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["train", "val", "test"]}],
        datasets={"train": ToyImages(), "val": ToyImages(seed=1), "test": ToyImages(seed=2)},
    )
    result = _run(config)
    assert result.success, result.errors


def test_a_splice_whose_input_was_skipped_skips_every_step() -> None:
    config = chain_pipeline(
        workflows=[
            {"name": "splitting", "type": "data-splitting", "split_on": ["nope"]},
            _AUDIT,
            {
                "name": "outer",
                "inputs": ["data"],
                "steps": [
                    {"name": "splits", "workflow": "splitting", "input": "data", "optional": True},
                    *_COLLECT_AUDIT,
                ],
            },
        ],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["data"]}],
        datasets={"data": SplitImages(90)},
        extra={"seed": 0},
    )
    result = _run(config)
    assert result.success, result.errors
    audit = {name: record for name, record in result.steps.items() if name.startswith("audit/")}
    assert audit
    assert {record.status for record in audit.values()} == {"skipped"}
    assert all("splits.train" in (record.reason or "") for record in audit.values())


def _spliced_three(
    datasets: dict[str, Any], audit: dict[str, Any] | None = None, *, extra: dict[str, Any] | None = None
) -> PipelineConfig:
    """Sources train, val and test bound to slots of those names, collect over val and test, then the audit step."""
    return chain_pipeline(
        workflows=[
            {**_AUDIT, **(audit or {})},
            {
                "name": "outer",
                "inputs": list(datasets),
                "steps": [
                    {"name": "evals", "transform": "collect", "input": list(datasets)[1:]},
                    {"name": "audit", "workflow": "release", "input": [list(datasets)[0], "evals"]},
                ],
            },
        ],
        tasks=[{"name": "t", "workflow": "outer", "sources": list(datasets)}],
        datasets=datasets,
        extra=extra,
    )


def _bare(value: Any) -> Any:
    """`value` with the splice's prefix dropped from every name and address in it."""
    return json.loads(json.dumps(value).replace("audit/", ""))


def _record_row(result: ChainResult, label: str) -> dict[str, Any]:
    """The record table's row `label`, by column, less its label."""
    record = _section(_top(result), "What was audited")
    table = next(block for block in record.blocks if isinstance(block, Table))
    row = next(row for row in table.rows if row[""] == label)
    return {column: value for column, value in row.items() if column}


def _exit(config: PipelineConfig, tmp_path: Path, **kwargs: Any) -> int:
    with patch("dataeval_flow._runner._resolve_config", return_value=config):
        return run("pipeline.yaml", None, data_dir=tmp_path, **kwargs)


# --- 7.1 equivalence ---


def test_a_spliced_audit_gives_the_task_s_verdict_and_findings() -> None:
    task = _audit(_three())
    DatasetCache.clear_instances()
    spliced = _run(_spliced_three(_three()))
    assert spliced.verdict is not None
    assert task.verdict is not None
    assert _bare(spliced.verdict.model_dump(mode="json")) == task.verdict.model_dump(mode="json")
    # Lists, not tuples: `_bare` round-trips through JSON, which reads a tuple back as a list.
    found = [[f.step, f.title, f.severity, f.brief] for f in spliced.findings]
    assert _bare(found) == [[f.step, f.title, f.severity, f.brief] for f in task.findings]


def test_a_spliced_audit_encodes_each_split_as_the_task_does() -> None:
    datasets = {"train": ToyFactors(60), "val": ToyFactors(30), "test": ToyFactors(5)}
    task = _audit(datasets)
    DatasetCache.clear_instances()
    spliced = _run(_spliced_three({"train": ToyFactors(60), "val": ToyFactors(30), "test": ToyFactors(5)}))
    addresses = ("train", "evals[val]", "evals[test]")
    assert _audited_digests(spliced, addresses) == _audited_digests(task, addresses)


# --- 7.2 behaviour ---


def test_split_collect_audit_gives_a_verdict_and_a_record_column_per_part() -> None:
    result = _run(_split_pipeline())
    assert result.success, result.errors
    assert result.verdict is not None
    record = _section(_top(result), "What was audited")
    table = next(block for block in record.blocks if isinstance(block, Table))
    assert [column.header for column in table.columns][1:] == ["train", "val", "test"]
    encoding = _record_row(result, "Encoding")
    assert encoding["train"]
    assert len(set(encoding.values())) == 1, encoding


def test_a_train_test_split_audits_with_test_alone() -> None:
    steps = [
        {"name": "evals", "transform": "collect", "input": ["splits.test"]},
        {"name": "audit", "workflow": "release", "input": ["splits.train", "evals"]},
    ]
    result = _run(_split_pipeline(steps, splitting={"val_frac": 0.0}))
    assert result.verdict is not None


def test_split_collect_audit_with_an_extractor_assesses_the_embedding_checks() -> None:
    result = _run(_split_pipeline(extractor=True))
    assert result.verdict is not None
    unassessed = {item.check for item in result.verdict.not_assessed if "extractor" in item.reason}
    assert not unassessed & {"eval-coverage", "embedding-divergence", "class-coverage"}


def test_the_verdict_names_spliced_steps_and_an_inner_acceptance_covers_one() -> None:
    plain = _run(_spliced_three(_three()))
    assert plain.verdict is not None
    warned = {item.step for item in plain.verdict.warnings if item.check == "image-outliers"}
    assert "audit/image-outliers-evals[test]" in warned
    DatasetCache.clear_instances()
    accepted = _run(_spliced_three(_three(), {"accepted": {"image-outliers-evals[test]": "Night shots."}}))
    assert accepted.verdict is not None
    steps = {item.step for item in accepted.verdict.warnings if item.check == "image-outliers"}
    assert "audit/image-outliers-evals[test]" not in steps
    assert "audit/image-outliers-evals[val]" in steps
    assert [(a.check, a.state) for a in accepted.verdict.accepted] == [("image-outliers-evals[test]", "warned")]


def test_every_verdict_step_names_a_step_of_the_result() -> None:
    result = _run(_spliced_three(_three()))
    payload = result.to_dict()
    assert "verdict" in payload
    verdict: Any = payload["verdict"]
    for item in [*verdict["warnings"], *verdict["blocking"]]:
        assert item["step"].split("[", 1)[0] in result.steps


def test_an_acceptance_naming_no_collected_element_is_refused_at_load() -> None:
    with pytest.raises(ValidationError, match=r"`image-outliers-evals\[nope\]`.*Its elements: val, test"):
        _spliced_three(_three(), {"accepted": {"image-outliers-evals[nope]": "x"}})


def test_two_steps_that_give_a_verdict_are_refused_at_load() -> None:
    steps = [
        {"name": "evals", "transform": "collect", "input": ["splits.val", "splits.test"]},
        {"name": "audit-a", "workflow": "release", "input": ["splits.train", "evals"]},
        {"name": "audit-b", "workflow": "release", "input": ["splits.train", "evals"]},
    ]
    with pytest.raises(ValidationError, match="runs two steps that give a verdict, `audit-a` and `audit-b`"):
        _split_pipeline(steps)


def test_an_optional_step_that_gives_a_verdict_is_refused_at_load() -> None:
    steps = [
        {"name": "evals", "transform": "collect", "input": ["splits.val", "splits.test"]},
        {"name": "audit", "workflow": "release", "input": ["splits.train", "evals"], "optional": True},
    ]
    with pytest.raises(ValidationError, match="gives a verdict, so it may not be optional"):
        _split_pipeline(steps)


def test_an_element_keyed_like_a_single_slot_is_refused_at_load() -> None:
    # The record's columns and the preflight's names hold the slot and the element alike: they may not share a name.
    steps = [
        {"name": "evals", "transform": "collect", "input": ["val", "test"], "keys": ["train", "test"]},
        {"name": "audit", "workflow": "release", "input": ["train", "evals"]},
    ]
    with pytest.raises(ValidationError, match="has the name of its slot `train`"):
        chain_pipeline(
            workflows=[_AUDIT, {"name": "outer", "inputs": ["train", "val", "test"], "steps": steps}],
            tasks=[{"name": "t", "workflow": "outer", "sources": ["train", "val", "test"]}],
            datasets=_three(),
        )


def test_a_spliced_preset_that_records_nothing_takes_an_element_keyed_like_its_slot() -> None:
    # A collision confuses only a record's columns and a preflight's names; drift-monitoring keeps neither.
    drift = {"name": "drift", "type": "drift-monitoring", "detectors": [{"type": "drift-kneighbors", "k": 3}]}
    steps = [
        {"name": "evals", "transform": "collect", "input": ["val", "test"], "keys": ["reference", "test"]},
        {"name": "drift", "workflow": "drift", "input": ["train", "evals"]},
    ]
    config = chain_pipeline(
        workflows=[drift, {"name": "outer", "inputs": ["train", "val", "test"], "steps": steps}],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["train", "val", "test"], "extractor": "flat"}],
        datasets=_three(),
        extractor=True,
    )
    assert [task.name for task in config.tasks or ()] == ["t"]


def _never_started() -> PipelineConfig:
    return chain_pipeline(
        workflows=[
            {"name": "splitting", "type": "data-splitting", "split_on": ["nope"]},
            _AUDIT,
            {
                "name": "outer",
                "inputs": ["data"],
                "steps": [
                    {"name": "splits", "workflow": "splitting", "input": "data", "optional": True},
                    *_COLLECT_AUDIT,
                ],
            },
        ],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["data"]}],
        datasets={"data": SplitImages(90)},
        extra={"seed": 0, "result": {"require": "ready-with-caveats"}},
    )


def test_a_splice_that_never_started_gives_no_verdict_and_says_why(tmp_path: Path) -> None:
    config = _never_started()
    result = _run(config)
    assert result.success, result.errors
    assert result.verdict is None
    assert result.no_verdict is not None
    assert result.no_verdict.startswith("step `audit` did not run:")
    DatasetCache.clear_instances()
    with patch("dataeval_flow._runner._logger") as logger:
        assert _exit(config, tmp_path) == 4
    logged = " ".join(str(call) for call in logger.error.call_args_list)
    assert "no verdict: step `audit` did not run" in logged


def test_a_missing_verdict_says_why_in_the_json_and_the_report_head() -> None:
    result = _run(_never_started())
    assert result.no_verdict is not None
    assert result.to_dict()["no_verdict"] == result.no_verdict
    head = _top(result, detailed=False)[0]
    assert isinstance(head, Paragraph)
    assert head.text == f"No verdict: {result.no_verdict}"
    assert "no_verdict" not in _run(_spliced_three(_leaky())).to_dict()


def test_a_splice_that_failed_as_it_started_gives_no_verdict_even_when_its_first_step_is_optional() -> None:
    result = _run(_never_started())
    result.splice_runs["audit"] = SpliceRun(failed="could not be encoded")
    result.verdict = result.no_verdict = None
    assert result.preset_chain is not None
    result.attach_preset(result.preset_chain, splice="audit")
    assert result.verdict is None
    assert result.no_verdict == "step `audit` failed as it started: could not be encoded"


def test_a_spliced_audit_that_is_not_ready_exits_4_under_require(tmp_path: Path) -> None:
    config = _spliced_three(_leaky(), extra={"result": {"require": "ready-with-caveats"}})
    assert _exit(config, tmp_path) == 4


def test_each_matrix_run_of_a_spliced_audit_is_judged(tmp_path: Path) -> None:
    datasets = {**_three(), "empty": Items([])}
    config = chain_pipeline(
        workflows=[
            _AUDIT,
            {
                "name": "outer",
                "inputs": ["train", {"name": "evals", "list": True}],
                "steps": [{"name": "audit", "workflow": "release", "input": ["train", "evals"]}],
            },
        ],
        tasks=[
            {
                "name": "t",
                "workflow": "outer",
                "sources": ["train", "val"],
                "matrix": {"sources": [["train", "val"], ["train", "empty"]]},
            }
        ],
        datasets=datasets,
        # The failed run alone would exit 1; `fail_on: never` lets the verdict gate speak for it.
        extra={"result": {"require": "ready-with-caveats", "fail_on": "never"}},
    )
    with patch("dataeval_flow._runner._logger") as logger:
        assert _exit(config, tmp_path) == 4
    logged = " ".join(str(call) for call in logger.error.call_args_list)
    assert "t run 2 (no verdict: it failed)" in logged


def test_data_splitting_s_steps_are_not_audit_evidence() -> None:
    result = _run(_split_pipeline())
    assert result.preset_chain is not None
    heading = next(group.heading for group in result.preset_chain.groups if "diversity" in group.evidence)
    question = _section(_top(result), heading)
    dumped = json.dumps([block.model_dump() for block in question.blocks], default=str)
    assert "splits/diversity" not in dumped
    assert "audit/diversity" in dumped


def test_a_check_outside_the_splice_is_neither_judged_nor_under_an_audit_question() -> None:
    config = chain_pipeline(
        workflows=[
            _AUDIT,
            {
                "name": "outer",
                "inputs": ["train", "val", "test"],
                "steps": [
                    {"name": "dupes-all", "evaluator": "dupes", "input": "train"},
                    {"name": "dupes-check", "check": "image-duplicates", "input": "dupes-all"},
                    {"name": "evals", "transform": "collect", "input": ["val", "test"]},
                    {"name": "audit", "workflow": "release", "input": ["train", "evals"]},
                ],
            },
        ],
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["train", "val", "test"]}],
        datasets=_three(),
    )
    result = _run(config)
    assert result.verdict is not None
    judged = {item.step for item in [*result.verdict.warnings, *result.verdict.blocking]}
    assert all(step.startswith("audit/") for step in judged)
    clean = _section(_top(result), _HEADINGS[0])
    assert "dupes-check" not in json.dumps([block.model_dump() for block in clean.blocks], default=str)


def test_the_record_shows_the_audit_s_criteria_and_the_envelope_records_its_entry() -> None:
    result = _run(_spliced_three(_three()))
    criteria = _section(_section(_top(result), "What was audited").blocks, "Criteria")
    assert "image-outliers" in json.dumps([block.model_dump() for block in criteria.blocks], default=str)
    assert result.metadata.resolved_config["presets"]["audit"]["type"] == "audit"


def test_the_detailed_report_reads_verdict_record_questions_next_steps_then_custom_groups() -> None:
    # A custom group needs a check outside the splice of a type it names.
    steps = [
        {"name": "dupes-all", "evaluator": "dupes", "input": "train"},
        {"name": "dupes-check", "check": "image-duplicates", "input": "dupes-all"},
        {"name": "evals", "transform": "collect", "input": ["val", "test"]},
        {"name": "audit", "workflow": "release", "input": ["train", "evals"]},
    ]
    config = chain_pipeline(
        workflows=[
            _AUDIT,
            {
                "name": "outer",
                "inputs": ["train", "val", "test"],
                "steps": steps,
                "groups": [{"heading": "Our own", "checks": ["image-duplicates"]}],
            },
        ],
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["train", "val", "test"]}],
        datasets=_three(),
    )
    outline = _outline(_run(config))
    assert outline[: outline.index("Our own") + 1] == [
        "verdict",
        "Verdict",
        "What was audited",
        *_HEADINGS,
        "Next steps",
        "Our own",
    ]


def test_the_short_report_lists_the_audit_questions_then_the_custom_groups() -> None:
    steps = [
        {"name": "dupes-all", "evaluator": "dupes", "input": "train"},
        {"name": "dupes-check", "check": "image-duplicates", "input": "dupes-all"},
        {"name": "evals", "transform": "collect", "input": ["val", "test"]},
        {"name": "audit", "workflow": "release", "input": ["train", "evals"]},
    ]
    config = chain_pipeline(
        workflows=[
            _AUDIT,
            {
                "name": "outer",
                "inputs": ["train", "val", "test"],
                "steps": steps,
                "groups": [{"heading": "Our own", "checks": ["image-duplicates"]}],
            },
        ],
        evaluators=[DuplicatesConfig(name="dupes")],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["train", "val", "test"]}],
        datasets=_three(),
    )
    result = _run(config)
    questions = _section(_top(result, detailed=False), "Questions")
    (fields,) = questions.blocks
    assert isinstance(fields, Fields)
    assert [label for label, _ in fields.items] == [*_HEADINGS, "Our own"]


def test_clean_then_audit_encodes_every_split_like_the_cleaned_train() -> None:
    datasets = {"train": ToyFactors(60), "val": ToyFactors(30), "test": ToyFactors(5)}
    config = chain_pipeline(
        workflows=[
            {"name": "cleaning", "type": "data-cleaning", **_OUTLIERS},
            _AUDIT,
            {
                "name": "outer",
                "inputs": ["train", {"name": "evals", "list": True}],
                "steps": [
                    {"name": "clean-train", "workflow": "cleaning", "input": "train"},
                    {"name": "clean-evals", "workflow": "cleaning", "input": "evals"},
                    {"name": "audit", "workflow": "release", "input": ["clean-train.clean", "clean-evals.clean"]},
                ],
            },
        ],
        tasks=[{"name": "t", "workflow": "outer", "sources": list(datasets)}],
        datasets=datasets,
    )
    result = _run(config)
    assert result.verdict is not None
    binning = result.metadata.metadata_binning
    assert binning is not None
    audit_reads = {
        key: record["encoding_digest"]
        for key, record in binning["per_split"].items()
        if key.partition(" (")[0] in {"clean-train/clean", "clean-evals/clean[val]", "clean-evals/clean[test]"}
    }
    assert len(set(audit_reads.values())) == 1, audit_reads


def test_the_record_encodes_by_the_splice_s_own_reads_not_an_outer_step_s() -> None:
    # An outer step reads train first under another binning; the audit's record shows the audit's encoding.
    datasets = {"train": ToyFactors(60), "val": ToyFactors(30), "test": ToyFactors(5)}
    config = chain_pipeline(
        workflows=[
            _AUDIT,
            {
                "name": "outer",
                "inputs": list(datasets),
                "steps": [
                    {"name": "pre", "evaluator": "balance-binned", "input": "train"},
                    {"name": "evals", "transform": "collect", "input": ["val", "test"]},
                    {"name": "audit", "workflow": "release", "input": ["train", "evals"]},
                ],
            },
        ],
        evaluators=[{"name": "balance-binned", "type": "balance", "metadata": "binned"}],
        tasks=[{"name": "t", "workflow": "outer", "sources": list(datasets)}],
        datasets=datasets,
        extra={"metadata": [{"name": "binned", "continuous_factor_bins": {"angle": 3}}]},
    )
    result = _run(config)
    assert result.verdict is not None
    binning = result.metadata.metadata_binning
    assert binning is not None
    outer, audit = (
        binning["per_split"]["train"]["encoding_digest"],
        binning["per_split"]["evals[val]"]["encoding_digest"],
    )
    assert outer != audit
    assert _record_row(result, "Encoding") == {"train": audit, "val": audit, "test": audit}
