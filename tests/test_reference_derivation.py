"""Reference derivation: in a preset naming a reference, every split's metadata is encoded like it (audit spec §9.3)."""

from typing import Any

import pytest

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._chain._graph import GraphError
from dataeval_flow._chain._nodes import Node, Root
from dataeval_flow._chain._run import StepContext
from dataeval_flow._policy import ResolvedPolicy
from dataeval_flow.steps._port import DataType
from dataeval_flow.workflows import WorkflowContext
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyFactors
from tests.preset_toys import register_presets


@pytest.fixture(autouse=True)
def presets(plugins):
    register_presets(plugins)
    DatasetCache.clear_instances()
    yield plugins
    DatasetCache.clear_instances()


def _run(datasets: dict[str, Any], policies: tuple[dict[str, Any], ...] = (), **settings: Any) -> Any:
    config = chain_pipeline(
        workflows=[{"name": "ref", "type": "toy-reference-preset", **settings}],
        tasks=[{"name": "t", "workflow": "ref", "sources": list(datasets)}],
        datasets=datasets,
        extra={"metadata": list(policies)},
    )
    return run_tasks(config)["t"]


def test_every_split_shares_train_s_encoding() -> None:
    # ToyFactors(5)'s angles are 0..4, so encoded on its own it differs from ToyFactors(60); derived, it matches.
    result = _run({"train": ToyFactors(60), "test": ToyFactors(5)})
    assert result.success, result.errors
    per_split = result.metadata.metadata_binning["per_split"]
    assert list(per_split) == ["train", "evals[test]", "parts.train[test]"]
    assert len({record["encoding_digest"] for record in per_split.values()}) == 1
    assert result.metadata.encoding_digest == per_split["train"]["encoding_digest"]
    train = result.steps["train-triage"].output.data()["findings"]
    evals = result.steps["evals-triage"].elements["test"].output.data()["findings"]
    assert sorted({finding.factor for finding in train}) == ["angle", "site"]
    assert evals == []  # an evaluation split takes train's binning advice


def test_reference_split_names_the_reference() -> None:
    result = _run(
        {"train": ToyFactors(5), "test": ToyFactors(60)},
        policies=({"name": "p", "reference_split": "test"},),
        metadata="p",
    )
    assert result.success, result.errors
    per_split = result.metadata.metadata_binning["per_split"]
    assert per_split["train"]["encoding_digest"] == per_split["evals[test]"]["encoding_digest"]


def test_an_unbound_reference_split_is_refused() -> None:
    with pytest.raises(GraphError, match="reference_split='holdout'"):
        _run(
            {"train": ToyFactors(60), "test": ToyFactors(5)},
            policies=({"name": "p", "reference_split": "holdout"},),
            metadata="p",
        )


def test_a_reference_that_cannot_be_encoded_fails_the_task() -> None:
    result = _run(
        {"train": ToyFactors(60), "test": ToyFactors(5)},
        policies=({"name": "p", "factor_levels": {"site": ["north"]}, "strict": True},),
        metadata="p",
    )
    assert not result.success
    assert "reference split `train`" in result.errors[0]
    assert result.steps == {}


def test_a_step_picks_its_policy_by_lineage() -> None:
    own, derived = ResolvedPolicy(), ResolvedPolicy(exclude=("x",))
    step = StepContext(metadata_policy=own, derived_policy=derived, reference="train")

    def node(*sources: str) -> Node:
        return Node("n", DataType.DATASET, roots=tuple(Root(source, "c") for source in sources))

    assert step.policy_for(node("train")) is own
    assert step.policy_for(node("test")) is derived
    assert step.policy_for(node("train", "test")) is derived
    assert StepContext(metadata_policy=own).policy_for(node("test")) is own


def test_a_context_reads_each_source_under_its_own_policy() -> None:
    own, derived = ResolvedPolicy(), ResolvedPolicy(exclude=("x",))
    context = WorkflowContext(metadata_policy=own, metadata_policies={"s": derived})
    assert context.metadata_policy_for("s") is derived
    assert context.metadata_policy_for("t") is own
