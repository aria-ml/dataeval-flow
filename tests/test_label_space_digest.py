"""A chain's `label-alignment` steps stamp its label space where nothing conformed one (coverage spec §3.5)."""

from typing import Any

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._orchestrator import _stamp_alignment_digest
from dataeval_flow._result import LabelSpaceRecord, ResultMetadata
from dataeval_flow.steps import ChainResult
from dataeval_flow.steps._result import StepResult
from tests.chain_toys import chain_pipeline


def _run(steps: list[dict[str, Any]], evaluators: list[dict[str, Any]], **kwargs: Any) -> ChainResult:
    DatasetCache.clear_instances()
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": kwargs.pop("inputs", ["data"]), "steps": steps}],
        evaluators=evaluators,
        tasks=[{"name": "t", "workflow": "w", "sources": kwargs.pop("sources", ["src"])}],
        **kwargs,
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


_ALIGN = {"name": "align", "type": "label-alignment", "ontology": {"a": None, "b": None}}


def test_one_alignment_step_stamps_its_digest() -> None:
    result = _run([{"name": "align", "evaluator": "align", "input": "data"}], [_ALIGN])
    digest = result.steps["align"].output.alignment.label_space_digest
    assert result.metadata.label_space_digest == digest


def test_a_broadcast_alignment_whose_elements_agree_stamps_their_digest() -> None:
    from tests.evaluator_toys import ToyImages

    result = _run(
        [{"name": "align", "evaluator": "align", "input": "splits"}],
        [_ALIGN],
        inputs=[{"name": "splits", "list": True}],
        sources=["a", "b"],
        datasets={"a": ToyImages(count=10), "b": ToyImages(count=12, seed=1)},
    )
    assert result.metadata.label_space_digest is not None


def test_two_alignments_that_disagree_stamp_nothing() -> None:
    other = {"name": "other", "type": "label-alignment", "ontology": {"x": {"a": None}, "b": None}}
    result = _run(
        [
            {"name": "align", "evaluator": "align", "input": "data"},
            {"name": "other", "evaluator": "other", "input": "data"},
        ],
        [_ALIGN, other],
    )
    assert (
        result.steps["align"].output.alignment.label_space_digest
        != result.steps["other"].output.alignment.label_space_digest
    )
    assert result.metadata.label_space_digest is None


def _aligned(digest: str) -> StepResult:
    from types import SimpleNamespace

    output = SimpleNamespace(alignment=SimpleNamespace(label_space_digest=digest))
    return StepResult(
        name="align", kind="evaluator", type="label-alignment", inputs=["data"], status="ok", output=output
    )


def test_a_source_s_record_wins_over_an_alignment() -> None:
    metadata = ResultMetadata(
        label_space=[LabelSpaceRecord(source="src", class_remap={"a": "A"}, target=["A"], digest="from-source")],
        label_space_digest="from-source",
    )
    _stamp_alignment_digest(metadata, {"align": _aligned("from-alignment")})
    assert metadata.label_space_digest == "from-source"


def test_with_no_record_the_alignment_stamps() -> None:
    metadata = ResultMetadata()
    _stamp_alignment_digest(metadata, {"align": _aligned("from-alignment")})
    assert metadata.label_space_digest == "from-alignment"


def test_a_failed_alignment_stamps_nothing() -> None:
    metadata = ResultMetadata()
    failed = _aligned("from-alignment")
    failed.status = "failed"
    _stamp_alignment_digest(metadata, {"align": failed})
    assert metadata.label_space_digest is None
