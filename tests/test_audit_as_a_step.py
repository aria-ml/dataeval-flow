"""audit run as a step of a custom workflow: its preflight, its reference encoding, and (Task 4) its verdict, record and
questions on the task's result (audit-as-a-step spec §4)."""

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.config import PipelineConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.steps import ChainResult
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import Items, ToyImages
from tests.golden.splitting import SplitImages
from tests.test_audit_preset import _OUTLIERS


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
