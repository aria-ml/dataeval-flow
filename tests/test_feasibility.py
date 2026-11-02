"""The `feasibility` evaluator: DataEval's Feasibility over one classification source, its embeddings and its labels."""

import json
import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from dataeval import Metadata
from dataeval.performance import Feasibility
from pydantic import ValidationError

from dataeval_flow import run, run_task, run_tasks
from dataeval_flow._chain._graph import GraphError
from dataeval_flow._view import root_indices
from dataeval_flow.config import (
    DatasetProtocolConfig,
    PipelineConfig,
    SourceConfig,
    TaskConfig,
    ViewConfig,
    ViewOperation,
)
from dataeval_flow.config.extractors import UncertaintyExtractorConfig
from dataeval_flow.evaluators import get_evaluator
from dataeval_flow.evaluators.performance import (
    AllowancesConfig,
    FeasibilityConfig,
    FeasibilityEvaluator,
    FeasibilityResult,
)
from dataeval_flow.evaluators.performance._feasibility_result import PopulatedFeasibilityOutput
from dataeval_flow.steps import ChainResult
from tests.chain_toys import ToyDetections, chain_pipeline
from tests.evaluator_toys import FLAT, ToyClasses, ToyImages, output_json

# Four classes a flatten extractor separates, so the built-in witnesses demonstrate an easy target; and the same with
# 60% of the labels redrawn at random, so no model does better than 1 - 0.6 * 3/4 = 0.55 on them.
_SEPARABLE: dict[str, Any] = {"count": 400, "classes": 4}
_NOISY: dict[str, Any] = {"count": 1000, "classes": 4, "flip": 0.6}
_BOTH = {"beta": 0, "gamma": 0}


def _run(data: Any = None, **settings: Any) -> FeasibilityResult:
    settings.setdefault("target", {"accuracy": 0.9})
    settings.setdefault("seed", 0)
    result = run(FeasibilityConfig(**settings), ToyClasses(**_SEPARABLE) if data is None else data, extractor=FLAT)
    assert isinstance(result, FeasibilityResult)
    return result


def _determinations(result: FeasibilityResult) -> list[dict[str, Any]]:
    assert result.success, result.errors
    return result.output.determinations.to_dicts()


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """What the evaluator was handed and what it handed DataEval, recorded as a run passes through."""
    record = SimpleNamespace()
    run_step, evaluate = FeasibilityEvaluator.run, Feasibility.evaluate

    def spy_run(self: Any, config: Any, inputs: Any) -> Any:
        record.inputs = inputs[0]
        return run_step(self, config, inputs)

    def spy_evaluate(self: Any, dataset: Any, **kwargs: Any) -> Any:
        record.dataset, record.kwargs = dataset, kwargs
        return evaluate(self, dataset, **kwargs)

    monkeypatch.setattr(FeasibilityEvaluator, "run", spy_run)
    monkeypatch.setattr(Feasibility, "evaluate", spy_evaluate)
    return record


# --- A determination, and what the result keeps of it ---


def test_separable_data_demonstrates_an_easy_target_and_the_json_keeps_its_proof() -> None:
    result = _run()
    (row,) = _determinations(result)
    assert row["determination"] == "demonstrated_feasible"
    assert row["judged_scope"] == "full_input"  # the default scope
    assert row["witness_source"].startswith("builtin_witness/v1:")
    assert row["ceiling"] is None  # nulls kept, not dropped
    assert row["limiting_factor"] is None

    output = output_json(result)
    assert output["shape"] == "mapping"
    assert set(output["data"]) == {"determinations", "evidence"}  # DataEval's `data()`, not its `plot_type` echo
    table = output["data"]["determinations"]
    assert table["columns"] == result.output.determinations.columns
    assert len(table["columns"]) == 15
    assert table["rows"][0]["determination"] == "demonstrated_feasible"  # an enum, as its string
    assert table["rows"][0]["evidence_ids"] == row["evidence_ids"]
    assert set(output["data"]["evidence"]) == set(result.output.evidence)
    witness = output["data"]["evidence"][row["witness_source"]]
    assert witness["information_scope"] == "fixed_representation"
    assert witness["details"]["call_alpha"] == 0.05
    assert witness["details"]["seed"] == 0


def test_the_json_round_trips_and_export_writes_it(tmp_path: Any) -> None:
    result = _run()
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()
    written = json.loads(result.export(tmp_path).read_text())
    assert written["output"] == result.to_dict()["output"]


def test_the_result_states_alpha_and_the_population() -> None:
    result = _run(ToyClasses(**_SEPARABLE, declared=6, unlabeled=range(10)))
    assert result.metadata.resolved_config["evaluator"]["alpha"] == 0.05  # stated, not null
    assert result.metadata.dataeval.name.endswith("Feasibility.evaluate")
    assert output_json(result)["extras"] == {"items": 400, "labeled": 390, "absent_classes": ["c4", "c5"]}
    assert isinstance(result.output, PopulatedFeasibilityOutput)
    assert result.output.population() == (
        "labeled items only, 390 of 400; classes with no labeled item are in no claim: c4, c5"
    )


def test_the_evaluators_section_is_empty_because_the_finding_draws_the_table() -> None:
    result = _run()
    report = result.report()
    assert "Proof" not in report
    assert "witness_source" not in report  # the raw table's columns
    assert "All bounds hold together" not in report  # the printout is one call away, and the finding draws the table
    assert "All bounds hold together at 95% confidence" in str(result.output)


def test_a_capped_target_is_unresolved_until_the_user_asserts_allowances() -> None:
    target = {"accuracy": 0.95}
    (default,) = _determinations(_run(ToyClasses(**_NOISY), target=target))
    assert (default["determination"], default["limiting_factor"]) == ("unresolved", "scope_restricted")
    (beta,) = _determinations(_run(ToyClasses(**_NOISY), target=target, allowances={"beta": 0}))
    assert beta["determination"] == "unresolved"  # beta alone widens the ceiling to `fixed_representation`
    result = _run(ToyClasses(**_NOISY), target=target, allowances=_BOTH)
    (ruled_out,) = _determinations(result)
    assert ruled_out["determination"] == "infeasible_under_recorded_assumptions"
    assert ruled_out["ceiling_scope"] == "full_input"
    assert ruled_out["ceiling"] < 0.95
    recorded = output_json(result)["data"]["evidence"][ruled_out["evidence_ids"][0]]["details"]
    assert recorded["allowances"] == {"beta": 0.0, "gamma": 0.0}
    assert "user-supplied and assumed" in " ".join(recorded["assumptions"])
    assert result.metadata.resolved_config["evaluator"]["allowances"] == {"beta": 0.0, "gamma": 0.0}


def test_a_scope_narrower_than_the_default_can_rule_a_target_out_without_allowances() -> None:
    (row,) = _determinations(_run(ToyClasses(**_NOISY), target={"accuracy/partition_restricted": 0.95}))
    assert (row["determination"], row["judged_scope"]) == (
        "infeasible_under_recorded_assumptions",
        "partition_restricted",
    )


def test_more_targets_widen_every_bound() -> None:
    """`alpha` is one budget shared by every target, so a second target moves the first one's bound."""
    one = _determinations(_run(target={"accuracy": 0.5}))[0]["witness"]
    two = _determinations(_run(target={"accuracy": 0.5, "ap": 0.5}))[0]["witness"]
    assert two < one


def test_the_seed_and_alpha_reach_dataeval() -> None:
    result = _run(alpha=0.2, seed=5)
    evidence = next(iter(result.output.evidence.values()))
    assert (evidence.details["call_alpha"], evidence.details["seed"]) == (0.2, 5)
    assert result.metadata.resolved_config["evaluator"]["seed"] == 5


# --- What the evaluator refuses ---


def _refused(result: FeasibilityResult, *fragments: str) -> None:
    assert not result.success
    (message,) = result.errors
    for fragment in fragments:
        assert fragment in message, message


def test_an_empty_source_is_refused() -> None:
    _refused(_run(ToyClasses(count=0)), "Source 'dataset' has no items")


def test_a_source_with_no_labeled_item_is_refused() -> None:
    result = _run(ToyClasses(count=30, unlabeled=range(30)))
    _refused(result, "Source 'dataset' has no labeled item among its 30", "only labeled items count")


def test_a_source_with_one_observed_class_is_refused_though_it_declares_more() -> None:
    result = _run(ToyClasses(count=60, classes=1, declared=3))
    _refused(result, "Source 'dataset' has labeled items in only one class, 'c0'", "insufficient data")


def test_a_source_with_two_observed_classes_among_more_declared_is_judged() -> None:
    result = _run(ToyClasses(count=200, classes=2, declared=4))
    assert result.success, result.errors
    assert output_json(result)["extras"]["absent_classes"] == ["c2", "c3"]


def _crops_workflow(**step: Any) -> PipelineConfig:
    """A custom workflow that wraps a detection dataset into crops and judges a target on them."""
    steps = [
        {"name": "crops", "transform": "wrap", "input": "data", "wrapper": "DetectionCrops"},
        {"name": "feas", "evaluator": "feas", "input": "crops", **step},
        {"name": "gate", "check": "performance-feasibility", "input": "feas"},
    ]
    return chain_pipeline(
        workflows=[{"name": "w", "inputs": ["data"], "steps": steps}],
        evaluators=[FeasibilityConfig(name="feas", target={"accuracy": 0.5})],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"], "extractor": "flat"}],
        datasets={"src": ToyDetections([[0, 1], [1, 0], [0, 0], [1, 1]] * 4, {0: "a", 1: "b"})},
        extractor=True,
    )


def test_crops_made_by_wrap_are_refused_naming_the_wrapper_and_never_judged() -> None:
    result = run_tasks(_crops_workflow())["t"]
    assert isinstance(result, ChainResult)
    assert not result.success
    message = " ".join(result.errors)
    assert "Source " in message
    assert "is a derived dataset, a DetectionCrops over another dataset" in message
    assert "items share a source and are not independent" in message
    assert "judges no target on it" in message
    assert "determination" not in message


def test_a_derived_dataset_is_refused_through_a_view_too(seen: SimpleNamespace) -> None:
    from dataeval.data import DetectionCrops, Limit, View

    crops = DetectionCrops(cast("Any", ToyDetections([[0, 1], [1, 0]] * 4, {0: "a", 1: "b"})))
    config = FeasibilityConfig(target={"accuracy": 0.5})
    inputs = SimpleNamespace(
        source="s",
        dataset=View(crops, Limit(6)),
        embeddings=np.zeros((6, 3)),
        labels=np.array([0, 1] * 3),
        index2label={0: "a", 1: "b"},
    )
    with pytest.raises(ValueError, match="a DetectionCrops over another dataset"):
        FeasibilityEvaluator().run(config, [cast("Any", inputs)])
    assert not hasattr(seen, "kwargs")  # DataEval was never reached


# --- What the engine refuses, tested here and not reimplemented ---


_UNCERTAINTY = UncertaintyExtractorConfig(
    name="unc", model_path="model.onnx", metadata_path="model.json", preds_type="logits"
)


def test_a_model_running_extractor_is_refused_when_the_config_loads_and_again_when_the_run_starts() -> None:
    task = TaskConfig(name="t", workflow="feas", kind="evaluator", sources=["src"], extractor="unc")
    config = chain_pipeline(
        evaluators=[FeasibilityConfig(name="feas", target={"accuracy": 0.9})],
        datasets={"src": ToyClasses(count=40)},
        extra={"extractors": [_UNCERTAINTY]},
        graphs=False,
    )
    result = run_task(config, task)
    assert not result.success
    assert "needs one embeddings row per item, and extractor `unc` runs a model" in result.errors[0]
    assert "only drift and OOD evaluators read it" in result.errors[0]
    with pytest.raises(ValidationError, match="extractor `unc` runs a model.*only drift and OOD evaluators read it"):
        chain_pipeline(
            evaluators=[FeasibilityConfig(name="feas", target={"accuracy": 0.9})],
            tasks=[{"name": "t", "evaluator": "feas", "sources": ["src"], "extractor": "unc"}],
            datasets={"src": ToyClasses(count=40)},
            extra={"extractors": [_UNCERTAINTY]},
        )


def test_a_task_with_no_extractor_is_refused_when_the_config_loads() -> None:
    with pytest.raises(ValidationError, match="needs an extractor to produce embeddings"):
        chain_pipeline(
            evaluators=[FeasibilityConfig(name="feas", target={"accuracy": 0.9})],
            tasks=[{"name": "t", "evaluator": "feas", "sources": ["src"]}],
            datasets={"src": ToyClasses(count=40)},
        )


def test_by_is_refused_on_a_feasibility_step_when_the_config_loads() -> None:
    steps = [{"name": "feas", "evaluator": "feas", "input": "data", "by": "class"}]
    with pytest.raises(
        ValidationError, match="has `by:`, which slices embeddings and labels, and `feasibility` reads dataset"
    ):
        chain_pipeline(
            workflows=[{"name": "w", "inputs": ["data"], "steps": steps}],
            evaluators=[FeasibilityConfig(name="feas", target={"accuracy": 0.9})],
            tasks=[{"name": "t", "workflow": "w", "sources": ["src"], "extractor": "flat"}],
            datasets={"src": ToyClasses(count=40)},
            extractor=True,
        )


def test_detection_data_is_a_failed_result_in_an_evaluator_task_and_a_graph_error_in_a_workflow() -> None:
    boxes = ToyDetections([[0, 1], [1, 0], [0, 0], [1, 1]] * 4, {0: "a", 1: "b"})
    failed = run(FeasibilityConfig(target={"accuracy": 0.5}), boxes, extractor=FLAT)
    assert not failed.success
    assert "NotImplementedError" in failed.errors[0]  # DataEval's: object detection is planned

    config = chain_pipeline(
        workflows=[
            {"name": "w", "inputs": ["data"], "steps": [{"name": "feas", "evaluator": "feas", "input": "data"}]}
        ],
        evaluators=[FeasibilityConfig(name="feas", target={"accuracy": 0.5})],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"], "extractor": "flat"}],
        datasets={"src": boxes},
        extractor=True,
    )
    with pytest.raises(GraphError, match="`input` takes classification"):
        run_tasks(config)


# --- The data the evaluator is handed ---


def _viewed(*operations: ViewOperation) -> PipelineConfig:
    return PipelineConfig(
        datasets=[DatasetProtocolConfig(name="d", format="maite", dataset=ToyClasses(count=300, classes=4))],
        views=[ViewConfig(name="v", operations=list(operations))],
        sources=[SourceConfig(name="src", dataset="d", view="v")],
        extractors=[FLAT],
        evaluators=[FeasibilityConfig(name="e", target={"accuracy": 0.5}, seed=0)],
        tasks=[TaskConfig(name="t", workflow="e", sources=["src"], kind="evaluator", extractor="flat")],
    )


def test_labels_agree_with_metadata_through_a_relabel_then_a_limit(seen: SimpleNamespace) -> None:
    config = _viewed(
        ViewOperation(
            type="Relabel",
            params={"class_remap": {"c0": "c0", "c1": "c1", "c3": "c3"}, "target": ["c0", "c1", "c3"]},
        ),
        ViewOperation(type="Limit", params={"size": 120}),
    )
    result = run_tasks(config)["t"]
    assert result.success, result.errors
    metadata = Metadata(seen.dataset)
    assert len(seen.dataset) == 120
    assert seen.inputs.labels.tolist() == np.asarray(metadata.class_labels).tolist()
    assert dict(seen.inputs.index2label) == dict(metadata.index2label)
    assert dict(seen.inputs.index2label) == {0: "c0", 1: "c1", 2: "c3"}  # c2 dropped, c3 moved to index 2
    assert set(seen.inputs.labels.tolist()) == {0, 1, 2}


def test_the_dataset_and_its_embeddings_stay_aligned_under_a_seeded_shuffle(seen: SimpleNamespace) -> None:
    result = run_tasks(_viewed(ViewOperation(type="Shuffle", params={"seed": 7})))["t"]
    assert result.success, result.errors
    items = np.stack([np.ravel(seen.dataset[i][0]) for i in range(len(seen.dataset))])
    assert np.allclose(seen.kwargs["embeddings"], items)
    assert seen.kwargs["groups"] == root_indices(seen.dataset) != list(range(300))  # the shuffle reordered the units


def test_the_units_are_the_items_at_the_bottom_of_a_chain_of_views(seen: SimpleNamespace) -> None:
    """`view` then `select` stack two `View`s; the units are the original items, a repeat among them one unit."""
    ranking = {"name": "ranking", "evaluator": "rank", "input": "kept"}
    steps = [
        {
            "name": "kept",
            "transform": "view",
            "input": "data",
            "operations": [{"type": "Indices", "params": {"indices": [5, 9, 9, 2, 40, 7, 5, 11]}}],
        },
        ranking,
        {"name": "top", "transform": "select", "input": "kept", "ranking": "ranking", "n": 5},
        {"name": "feas", "evaluator": "feas", "input": "top"},
    ]
    config = chain_pipeline(
        workflows=[{"name": "w", "inputs": ["data"], "steps": steps}],
        evaluators=[
            {"name": "rank", "type": "prioritization"},
            FeasibilityConfig(name="feas", target={"accuracy|sequence_mean": 0.5}),
        ],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"], "extractor": "flat"}],
        datasets={"src": ToyClasses(count=60, classes=3)},
        extractor=True,
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    chosen = result.steps["top"].output.resolve_indices()
    expected = [[5, 9, 9, 2, 40, 7, 5, 11][i] for i in chosen]
    assert seen.kwargs["groups"] == expected == root_indices(seen.dataset)
    assert len(set(expected)) < len(expected)  # a repeat: two items, one unit


# --- The config ---


@pytest.mark.parametrize("key", ["precision", "recall", "ap/everything", "ap|frame_pooled", "ap@0.5", ""])
def test_a_malformed_target_key_is_refused_when_the_config_loads(key: str) -> None:
    with pytest.raises(ValidationError, match="is not a target key"):
        FeasibilityConfig(target={key: 0.9})


@pytest.mark.parametrize("key", ["accuracy", "ap", "ap:cat", "accuracy/fixed_representation", "accuracy|sequence_mean"])
def test_a_key_in_dataevals_grammar_loads(key: str) -> None:
    assert FeasibilityConfig(target={key: 0.9}).target == {key: 0.9}


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01, math.nan, math.inf])
def test_a_threshold_outside_zero_to_one_is_refused(threshold: float) -> None:
    with pytest.raises(ValidationError, match=r"must be a number in \(0, 1\]"):
        FeasibilityConfig(target={"accuracy": threshold})


def test_the_threshold_one_is_allowed() -> None:
    assert FeasibilityConfig(target={"accuracy": 1.0}).target == {"accuracy": 1.0}


def test_a_target_is_required_and_cannot_be_empty() -> None:
    with pytest.raises(ValidationError, match="Field required"):
        FeasibilityConfig.model_validate({})
    with pytest.raises(ValidationError, match="at least 1 item"):
        FeasibilityConfig(target={})


def test_an_unknown_field_and_dataevals_unoffered_ones_are_refused() -> None:
    for field in ("replicas", "predictions", "extractor", "batch_size", "max_sequence_length", "beta", "bogus"):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            FeasibilityConfig.model_validate({"target": {"accuracy": 0.9}, field: 1})


@pytest.mark.parametrize(
    "settings", [{"alpha": 0}, {"alpha": 1}, {"alpha": math.nan}, {"panel_size": 1}, {"n_cells": 0}]
)
def test_a_setting_outside_its_range_is_refused(settings: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        FeasibilityConfig(target={"accuracy": 0.9}, **settings)


@pytest.mark.parametrize(
    ("allowances", "why"),
    [
        ({"beta": -0.1}, "greater than or equal to 0"),
        ({"gamma": 0}, "only together with `beta`"),
        ({"delta": 0}, "Extra"),
    ],
)
def test_an_allowance_is_non_negative_and_gamma_needs_beta(allowances: dict[str, Any], why: str) -> None:
    with pytest.raises(ValidationError, match=why):
        FeasibilityConfig.model_validate({"target": {"accuracy": 0.9}, "allowances": allowances})


def test_the_config_round_trips_with_its_fields() -> None:
    config = FeasibilityConfig(
        target={"accuracy": 0.9, "ap:c1/fixed_representation": 0.5},
        alpha=0.01,
        panel_size=4,
        n_cells=8,
        builtin_witnesses=False,
        seed=3,
        allowances=AllowancesConfig(beta=0, gamma=0.1),
    )
    again = FeasibilityConfig.model_validate(config.model_dump(mode="json"))
    assert again == config
    assert again.alpha == 0.01
    assert FeasibilityConfig(target={"accuracy": 0.9}).alpha == 0.05


def test_the_evaluator_is_the_registered_one_and_experimental() -> None:
    evaluator = get_evaluator("feasibility")
    assert evaluator is FeasibilityEvaluator
    assert evaluator.experimental
    assert evaluator.config_type is FeasibilityConfig
    assert evaluator.config_type.result_type is FeasibilityResult
    assert ". " not in evaluator.description


# --- A matrix, which would spend the error budget once per run ---


def _matrix(matrix: Any, **task: Any) -> PipelineConfig:
    return chain_pipeline(
        evaluators=[FeasibilityConfig(name="feas", target={"accuracy": 0.9})],
        tasks=[{"name": "t", "evaluator": "feas", "sources": ["src"], "extractor": "flat", "matrix": matrix, **task}],
        datasets={"src": ToyImages(), "other": ToyImages(seed=1)},
        extractor=True,
    )


@pytest.mark.parametrize(
    "matrix",
    [
        {"alpha": [0.05, 0.01]},
        {"target": [{"accuracy": 0.9}, {"accuracy": 0.8}]},
        {"extractor": ["flat"]},
        {"seed": [0, 1]},
    ],
)
def test_a_matrix_over_anything_but_sources_is_refused_with_the_reason(matrix: dict[str, Any]) -> None:
    with pytest.raises(ValidationError) as refused:
        _matrix(matrix)
    text = str(refused.value)
    assert "evaluator `feas` allows only `sources` to vary" in text
    assert "choosing among runs needs `alpha` divided by their number" in text


def test_a_matrix_over_sources_alone_runs() -> None:
    from dataeval_flow._matrix._build import expand_matrix

    config = _matrix({"sources": ["src", "other"]})
    assert config.tasks
    assert [run.task.source_names for run in expand_matrix(config.tasks[0], config)] == [["src"], ["other"]]
