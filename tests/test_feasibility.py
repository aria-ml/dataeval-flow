"""The `feasibility` evaluator: DataEval's Feasibility over one classification source, its embeddings and its labels."""

import json
import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from dataeval import Metadata
from dataeval.performance import Allowances, Feasibility, TargetSpec
from pydantic import ValidationError

from dataeval_flow import InputKind, run, run_task, run_tasks
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
    TargetConfig,
)
from dataeval_flow.evaluators.performance._feasibility_result import PopulatedFeasibilityOutput
from dataeval_flow.steps import ChainResult
from tests.chain_toys import ToyDetections, chain_pipeline
from tests.evaluator_toys import FLAT, ToyClasses, ToyCopies, ToyImages, output_json

# Four classes a flatten extractor separates, so the built-in witnesses demonstrate an easy target; and the same with
# 60% of the labels redrawn at random, so no model does better than 1 - 0.6 * 3/4 = 0.55 on them.
_SEPARABLE: dict[str, Any] = {"count": 400, "classes": 4}
_NOISY: dict[str, Any] = {"count": 1000, "classes": 4, "flip": 0.6}
_BOTH = {"beta": 0, "gamma": 0}


def _accuracy(threshold: float, **fields: Any) -> TargetConfig:
    """A `target` entry for accuracy."""
    return TargetConfig(metric="accuracy", threshold=threshold, **fields)


def _run(data: Any = None, **settings: Any) -> FeasibilityResult:
    settings.setdefault("target", [_accuracy(0.9)])
    settings.setdefault("seed", 0)
    result = run(FeasibilityConfig(**settings), ToyClasses(**_SEPARABLE) if data is None else data, extractor=FLAT)
    assert isinstance(result, FeasibilityResult)
    return result


def _partition(groups: Any) -> list[int]:
    """`groups` renumbered from 0 by first appearance, so two labelings of one partition compare equal."""
    ids: dict[Any, int] = {}
    return [ids.setdefault(group, len(ids)) for group in groups]


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
    assert row["witness_source"] == "knn"
    assert row["envelope"] is None  # nulls kept, not dropped
    assert row["limitation"] is None

    output = output_json(result)
    assert output["shape"] == "mapping"
    # DataEval's `data()`, not its `plot_type` echo: what was asked, what was assumed, and the two tables.
    assert set(output["data"]) == {"requirements", "allowances", "determinations", "bounds"}
    assert output["data"]["requirements"][0]["metric"] == "accuracy"
    assert output["data"]["allowances"] is None
    table = output["data"]["determinations"]
    assert table["columns"] == result.output.determinations.columns
    assert table["rows"][0]["determination"] == "demonstrated_feasible"  # an enum, as its string
    bounds = output["data"]["bounds"]
    assert bounds["columns"] == result.output.bounds.columns
    deciding = [bound for bound in bounds["rows"] if bound["is_determining"]]
    assert [(bound["source"], bound["role"], bound["information_scope"]) for bound in deciding] == [
        ("knn", "witness", "fixed_representation")
    ]
    assert deciding[0]["alpha"] > 0  # each bound states its share of the error budget
    assert deciding[0]["assumptions"]  # and what it rests on


def test_the_json_round_trips_and_export_writes_it(tmp_path: Any) -> None:
    result = _run()
    assert json.loads(json.dumps(result.to_dict())) == result.to_dict()
    written = json.loads(result.export(tmp_path).read_text())
    assert written["output"] == result.to_dict()["output"]


def test_the_result_states_alpha_the_seed_and_the_population() -> None:
    result = _run(ToyClasses(**_SEPARABLE, declared=6, unlabeled=range(10)), seed=None)
    assert result.metadata.resolved_config["evaluator"]["alpha"] == 0.05  # stated, not null
    assert result.metadata.resolved_config["evaluator"]["seed"] is None  # unset: DataEval resolves it
    assert result.metadata.dataeval.name.endswith("Feasibility.evaluate")
    assert output_json(result)["extras"] == {
        "seed": 0,  # DataEval's resolved seed, which the config does not state
        "items": 400,
        "labeled": 390,
        "absent_classes": ["c4", "c5"],
        "groups": 400,  # no two items repeat, share a factor or duplicate each other
        "largest_group": 1,
        "duplicate_merges": 0,
    }
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
    assert "joint confidence: 95%" in str(result.output)


def test_a_capped_target_is_unresolved_until_the_user_asserts_allowances() -> None:
    target = [_accuracy(0.95)]
    (default,) = _determinations(_run(ToyClasses(**_NOISY), target=target))
    assert (default["determination"], default["limitation"]) == ("unresolved", "scope_restricted")
    (beta,) = _determinations(_run(ToyClasses(**_NOISY), target=target, allowances={"beta": 0}))
    assert beta["determination"] == "unresolved"  # beta alone widens the envelope to `fixed_representation`
    result = _run(ToyClasses(**_NOISY), target=target, allowances=_BOTH)
    (ruled_out,) = _determinations(result)
    assert ruled_out["determination"] == "infeasible_under_recorded_assumptions"
    assert ruled_out["envelope_scope"] == "full_input"
    assert ruled_out["envelope"] < 0.95
    assert output_json(result)["data"]["allowances"] == {"beta": 0.0, "gamma": 0.0}
    assert result.output.allowances == Allowances(beta=0.0, gamma=0.0)
    assert "allowance_beta" in {a for row in result.output.bounds.to_dicts() for a in row["assumptions"]}
    assert result.metadata.resolved_config["evaluator"]["allowances"] == {"beta": 0.0, "gamma": 0.0}


def test_a_scope_narrower_than_the_default_can_rule_a_target_out_without_allowances() -> None:
    target = [_accuracy(0.95, scope="partition_restricted")]
    (row,) = _determinations(_run(ToyClasses(**_NOISY), target=target))
    assert (row["determination"], row["judged_scope"]) == (
        "infeasible_under_recorded_assumptions",
        "partition_restricted",
    )


def test_more_targets_widen_every_bound() -> None:
    """`alpha` is one budget shared by every target, so a second target moves the first one's bound."""
    one = _determinations(_run(target=[_accuracy(0.5)]))[0]["witness"]
    two = _determinations(_run(target=[_accuracy(0.5), TargetConfig(metric="ap", threshold=0.5)]))[0]["witness"]
    assert two < one


def test_the_seed_and_alpha_reach_dataeval() -> None:
    result = _run(alpha=0.2, seed=5)
    state = result.output.meta().state
    assert (state["alpha"], state["resolved_seed"]) == (0.2, 5)
    assert output_json(result)["extras"]["seed"] == 5
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
        evaluators=[FeasibilityConfig(name="feas", target=[_accuracy(0.5)])],
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
    config = FeasibilityConfig(target=[_accuracy(0.5)])
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
        evaluators=[FeasibilityConfig(name="feas", target=[_accuracy(0.9)])],
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
            evaluators=[FeasibilityConfig(name="feas", target=[_accuracy(0.9)])],
            tasks=[{"name": "t", "evaluator": "feas", "sources": ["src"], "extractor": "unc"}],
            datasets={"src": ToyClasses(count=40)},
            extra={"extractors": [_UNCERTAINTY]},
        )


def test_a_task_with_no_extractor_is_refused_when_the_config_loads() -> None:
    with pytest.raises(ValidationError, match="needs an extractor to produce embeddings"):
        chain_pipeline(
            evaluators=[FeasibilityConfig(name="feas", target=[_accuracy(0.9)])],
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
            evaluators=[FeasibilityConfig(name="feas", target=[_accuracy(0.9)])],
            tasks=[{"name": "t", "workflow": "w", "sources": ["src"], "extractor": "flat"}],
            datasets={"src": ToyClasses(count=40)},
            extractor=True,
        )


def test_detection_data_is_a_failed_result_in_an_evaluator_task_and_a_graph_error_in_a_workflow() -> None:
    boxes = ToyDetections([[0, 1], [1, 0], [0, 0], [1, 1]] * 4, {0: "a", 1: "b"})
    failed = run(FeasibilityConfig(target=[_accuracy(0.5)]), boxes, extractor=FLAT)
    assert not failed.success
    assert "NotImplementedError" in failed.errors[0]  # DataEval's: object detection is planned

    config = chain_pipeline(
        workflows=[
            {"name": "w", "inputs": ["data"], "steps": [{"name": "feas", "evaluator": "feas", "input": "data"}]}
        ],
        evaluators=[FeasibilityConfig(name="feas", target=[_accuracy(0.5)])],
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
        evaluators=[FeasibilityConfig(name="e", target=[_accuracy(0.5)], seed=0)],
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
    roots = root_indices(seen.dataset)
    assert _partition(seen.kwargs["groups"]) == _partition(roots)
    assert roots != list(range(300))  # the shuffle reordered the units


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
            FeasibilityConfig(name="feas", target=[_accuracy(0.5, aggregation="sequence_mean")]),
        ],
        tasks=[{"name": "t", "workflow": "w", "sources": ["src"], "extractor": "flat"}],
        datasets={"src": ToyClasses(count=60, classes=3)},
        extractor=True,
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    chosen = result.steps["top"].output.resolve_indices()
    expected = [[5, 9, 9, 2, 40, 7, 5, 11][i] for i in chosen]
    assert expected == root_indices(seen.dataset)
    assert _partition(seen.kwargs["groups"]) == _partition(expected)
    assert len(set(expected)) < len(expected)  # a repeat: two items, one unit


# --- The independent units: a factor, exact and near duplicates ---


# A target on units of more than one item must say how to average over a unit.
_AVERAGED = [_accuracy(0.9, aggregation="sequence_mean")]


def _site_by_pair(base: int, copy: int) -> str:
    """Bases 2k and 2k+1 share a site, and each of their copies has a site of its own."""
    return f"pair{base // 2}" if copy == 0 else f"own{base}-{copy}"


def _group_sizes(groups: Any) -> list[int]:
    return sorted(np.bincount(_partition(groups)).tolist())


def test_exact_and_near_copies_are_one_group_by_default(seen: SimpleNamespace) -> None:
    result = _run(ToyCopies(bases=40, exact=1, near=1), target=_AVERAGED)
    assert result.success, result.errors
    assert len(seen.kwargs["groups"]) == 120
    assert _partition(seen.kwargs["groups"]) == [base for base in range(40) for _ in range(3)]
    assert output_json(result)["extras"] == {
        "seed": 0,
        "items": 120,
        "labeled": 120,
        "absent_classes": [],
        "groups": 40,
        "largest_group": 3,
        "duplicate_merges": 80,
    }
    assert seen.inputs.stats is not None
    assert seen.inputs.metadata is None  # no factor was named, so none was read


def test_exact_copies_alone_are_found_too(seen: SimpleNamespace) -> None:
    assert _run(ToyCopies(bases=40, exact=2), target=_AVERAGED).success
    assert _group_sizes(seen.kwargs["groups"]) == [3] * 40


def test_near_copies_alone_are_found_too(seen: SimpleNamespace) -> None:
    assert _run(ToyCopies(bases=40, near=2), target=_AVERAGED).success
    assert _group_sizes(seen.kwargs["groups"]) == [3] * 40


def test_without_copies_every_item_is_a_group_and_nothing_is_merged() -> None:
    result = _run(ToyCopies(bases=40))
    assert result.success, result.errors
    extras = output_json(result)["extras"]
    assert (extras["groups"], extras["largest_group"], extras["duplicate_merges"]) == (40, 1, 0)


def test_turning_duplicate_grouping_off_reads_no_hash_and_leaves_each_item_a_group(seen: SimpleNamespace) -> None:
    result = _run(ToyCopies(bases=40, exact=1, near=1), group_near_duplicates=False)
    assert result.success, result.errors
    assert seen.inputs.stats is None
    assert seen.inputs.metadata is None
    assert _partition(seen.kwargs["groups"]) == list(range(120))
    extras = output_json(result)["extras"]
    assert (extras["groups"], extras["largest_group"], extras["duplicate_merges"]) == (120, 1, 0)


def test_a_factor_alone_groups_the_items_that_share_its_value(seen: SimpleNamespace) -> None:
    data = ToyCopies(bases=40, exact=1, near=1, site=lambda base, copy: f"s{base % 8}")  # noqa: ARG005
    result = _run(data, groups="site", group_near_duplicates=False, target=_AVERAGED)
    assert result.success, result.errors
    assert seen.inputs.stats is None
    assert seen.inputs.metadata is not None
    assert _group_sizes(seen.kwargs["groups"]) == [15] * 8
    extras = output_json(result)["extras"]
    assert (extras["groups"], extras["largest_group"], extras["duplicate_merges"]) == (8, 15, 0)


def test_the_factor_and_the_duplicates_make_one_grouping(seen: SimpleNamespace) -> None:
    """Here 100 of the 120 items hold a value of their own, so DataEval's metadata drops `site` as a factor, as it does
    a scene or recording ID. It still groups, because the evaluator reads the value as recorded."""
    result = _run(ToyCopies(bases=40, exact=1, near=1, site=_site_by_pair), groups="site", target=_AVERAGED)
    assert result.success, result.errors
    assert _group_sizes(seen.kwargs["groups"]) == [6] * 20  # a base pair by its site, each base's copies by hash
    extras = output_json(result)["extras"]
    # By the factor alone: 20 pairs of base images and 80 copies on their own. Duplicates join each copy to its base.
    assert (extras["groups"], extras["largest_group"], extras["duplicate_merges"]) == (20, 6, 80)


def test_the_groups_a_view_repeats_still_join_the_factor_and_duplicates(seen: SimpleNamespace) -> None:
    indices = [0, 1, 2, 3, 0, 1]  # items 0 and 1 are copies of one image, and the last two repeat them
    config = PipelineConfig(
        datasets=[DatasetProtocolConfig(name="d", format="maite", dataset=ToyCopies(bases=40, exact=1))],
        views=[ViewConfig(name="v", operations=[ViewOperation(type="Indices", params={"indices": indices})])],
        sources=[SourceConfig(name="src", dataset="d", view="v")],
        extractors=[FLAT],
        evaluators=[FeasibilityConfig(name="e", target=[_accuracy(0.5, aggregation="sequence_mean")], seed=0)],
        tasks=[TaskConfig(name="t", workflow="e", sources=["src"], kind="evaluator", extractor="flat")],
    )
    result = run_tasks(config)["t"]
    assert result.success, result.errors
    assert _partition(seen.kwargs["groups"]) == [0, 0, 1, 1, 0, 0]


def test_the_grouping_is_in_the_resolved_config() -> None:
    result = _run(ToyCopies(bases=40, site=lambda base, copy: f"s{base % 8}"), groups="site", target=_AVERAGED)  # noqa: ARG005
    assert result.success, result.errors
    assert result.metadata.resolved_config["evaluator"]["groups"] == "site"
    assert result.metadata.resolved_config["evaluator"]["group_near_duplicates"] is True


def test_a_factor_the_source_lacks_is_refused_naming_the_factors_it_has() -> None:
    result = _run(ToyCopies(bases=40, site=lambda base, copy: "s"), groups="place", group_near_duplicates=False)  # noqa: ARG005
    _refused(result, "Source 'dataset' has no factor 'place'", "its metadata holds site")


def test_a_numeric_factor_with_no_value_for_a_labeled_item_is_refused() -> None:
    """Counting such an item alone would claim it independent of every other, which nothing shows."""
    data = ToyCopies(bases=40, site=lambda base, copy: math.nan if base < 3 else float(base % 5))  # noqa: ARG005
    result = _run(data, groups="site", group_near_duplicates=False, target=_AVERAGED)
    _refused(result, "Source 'dataset'", "`site`", "3 of its 40 labeled items", "no value")


def test_an_accuracy_target_without_an_average_is_refused_where_items_share_a_unit_and_says_what_to_add(
    seen: SimpleNamespace,
) -> None:
    result = _run(ToyCopies(bases=40, exact=1), target=[_accuracy(0.9)])
    _refused(
        result,
        "The 80 items of source 'dataset' fall into 40 independent units, the largest of 2 items",
        "add `aggregation: sequence_mean` to `accuracy ≥0.9`",
    )
    assert not hasattr(seen, "kwargs")  # DataEval was never reached


def test_an_ap_target_is_refused_where_items_share_a_unit_because_no_average_is_defined_for_it(
    seen: SimpleNamespace,
) -> None:
    target = [_accuracy(0.9, aggregation="sequence_mean"), TargetConfig(metric="ap", threshold=0.8, class_name="a")]
    result = _run(ToyCopies(bases=40, exact=1), target=target)
    _refused(
        result,
        "fall into 40 independent units",
        "`ap:a ≥0.8` cannot be judged on such units",
        "DataEval defines `sequence_mean` for `accuracy` only",
        "`group_near_duplicates: false`",
    )
    assert "add `aggregation" not in result.errors[0]  # the accuracy target already says how to average
    assert not hasattr(seen, "kwargs")


def test_an_average_is_refused_where_no_two_items_share_a_unit() -> None:
    result = _run(ToyCopies(bases=40), target=_AVERAGED)
    _refused(result, "are each a unit of their own", "remove `aggregation` from `accuracy ≥0.9`")


def test_a_target_without_an_average_is_fine_where_no_two_items_share_a_unit() -> None:
    assert _run(ToyCopies(bases=40), target=[_accuracy(0.9)]).success


def test_the_inputs_the_run_reads_follow_its_settings() -> None:
    always = {InputKind.DATASET, InputKind.EMBEDDINGS, InputKind.LABELS}
    target = [_accuracy(0.9)]
    assert FeasibilityConfig(target=target).wanted_kinds() == always | {InputKind.STATS}
    assert FeasibilityConfig(target=target, group_near_duplicates=False).wanted_kinds() == always
    assert FeasibilityConfig(target=target, groups="site").wanted_kinds() == always | {
        InputKind.STATS,
        InputKind.METADATA,
    }
    assert FeasibilityConfig(target=target, groups="site", group_near_duplicates=False).wanted_kinds() == always | {
        InputKind.METADATA
    }


def test_an_empty_factor_name_is_refused() -> None:
    with pytest.raises(ValidationError, match="at least 1 character"):
        FeasibilityConfig(target=[_accuracy(0.9)], groups="")


# --- The config ---


def _loads(*targets: Any, **settings: Any) -> FeasibilityConfig:
    return FeasibilityConfig.model_validate({"target": list(targets), **settings})


@pytest.mark.parametrize(
    "entry",
    [
        {"metric": "accuracy", "threshold": 0.9},
        {"metric": "ap", "threshold": 0.5},
        {"metric": "ap", "threshold": 0.5, "class_name": "cat"},
        {"metric": "accuracy", "threshold": 0.5, "scope": "fixed_representation"},
        {"metric": "accuracy", "threshold": 0.5, "aggregation": "sequence_mean"},
    ],
)
def test_a_structured_target_loads_and_becomes_the_spec_dataeval_is_given(entry: dict[str, Any]) -> None:
    (target,) = _loads(entry).target
    assert target.spec() == TargetSpec(**entry)


@pytest.mark.parametrize(
    ("entry", "why"),
    [
        ({"metric": "precision", "threshold": 0.9}, "Input should be 'accuracy' or 'ap'"),
        ({"metric": "recall", "threshold": 0.9}, "Input should be 'accuracy' or 'ap'"),
        ({"metric": "accuracy"}, "threshold"),
        ({"metric": "accuracy", "threshold": 0.9, "iou": 0.5}, "Extra inputs are not permitted"),
        ({"metric": "accuracy", "threshold": 0.9, "scope": "everything"}, "Input should be 'full_input'"),
        ({"metric": "accuracy", "threshold": 0.9, "aggregation": "frame_pooled"}, "Input should be 'sequence_mean'"),
        ({"metric": "accuracy", "threshold": 0.9, "class_name": "cat"}, "class_name applies to 'ap' and 'recall' only"),
        ({"metric": "ap", "threshold": 0.9, "aggregation": "sequence_mean"}, "defined for 'accuracy' only"),
        ({"metric": "accuracy", "threshold": 0.9, "class_name": ""}, "at least 1 character"),
    ],
)
def test_a_target_dataeval_would_refuse_is_refused_when_the_config_loads(entry: dict[str, Any], why: str) -> None:
    with pytest.raises(ValidationError, match=why):
        _loads(entry)


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01, math.nan, math.inf])
def test_a_threshold_outside_zero_to_one_is_refused(threshold: float) -> None:
    with pytest.raises(ValidationError):
        _loads(_accuracy(threshold))


def test_the_threshold_one_is_allowed() -> None:
    assert _loads(_accuracy(1.0)).target[0].threshold == 1.0


def test_the_old_mapping_form_is_refused_when_the_config_loads() -> None:
    with pytest.raises(ValidationError, match="Input should be a valid list"):
        FeasibilityConfig.model_validate({"target": {"accuracy": 0.9}})


def test_a_target_is_required_and_cannot_be_empty() -> None:
    with pytest.raises(ValidationError, match="Field required"):
        FeasibilityConfig.model_validate({})
    with pytest.raises(ValidationError, match="at least 1 item"):
        FeasibilityConfig(target=[])


def test_an_unknown_field_and_dataevals_unoffered_ones_are_refused() -> None:
    for field in (
        "replicas",
        "predictions",
        "extractor",
        "batch_size",
        "max_sequence_length",
        "beta",
        "panel_size",
        "n_cells",
        "builtin_witnesses",
        "bogus",
    ):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            _loads(_accuracy(0.9), **{field: 1})


@pytest.mark.parametrize("settings", [{"alpha": 0}, {"alpha": 1}, {"alpha": math.nan}])
def test_a_setting_outside_its_range_is_refused(settings: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        FeasibilityConfig(target=[_accuracy(0.9)], **settings)


@pytest.mark.parametrize(
    ("allowances", "why"),
    [
        ({"beta": -0.1}, "greater than or equal to 0"),
        ({"gamma": 0}, "beta\n  Field required"),
        ({}, "beta\n  Field required"),
        ({"beta": 0, "delta": 0}, "Extra"),
    ],
)
def test_beta_is_required_and_the_allowances_are_non_negative(allowances: dict[str, Any], why: str) -> None:
    with pytest.raises(ValidationError, match=why):
        _loads(_accuracy(0.9), allowances=allowances)


def test_the_allowances_become_the_ones_dataeval_is_given() -> None:
    assert AllowancesConfig(beta=0, gamma=0.1).allowances() == Allowances(beta=0, gamma=0.1)
    assert AllowancesConfig(beta=0.2).allowances() == Allowances(beta=0.2)


def test_the_config_round_trips_with_its_fields() -> None:
    config = FeasibilityConfig(
        target=[
            _accuracy(0.9),
            TargetConfig(metric="ap", threshold=0.5, class_name="c1", scope="fixed_representation"),
        ],
        alpha=0.01,
        seed=3,
        allowances=AllowancesConfig(beta=0, gamma=0.1),
    )
    again = FeasibilityConfig.model_validate(config.model_dump(mode="json"))
    assert again == config
    assert again.alpha == 0.01
    assert FeasibilityConfig(target=[_accuracy(0.9)]).alpha == 0.05


def test_targets_keep_the_order_they_were_written_in() -> None:
    result = _run(
        target=[TargetConfig(metric="ap", threshold=0.5), _accuracy(0.9), _accuracy(0.5, scope="fixed_representation")]
    )
    assert [(row["metric"], row["judged_scope"]) for row in _determinations(result)] == [
        ("ap", "full_input"),
        ("accuracy", "full_input"),
        ("accuracy", "fixed_representation"),
    ]
    assert [row["requirement"] for row in _determinations(result)] == [0, 1, 2]


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
        evaluators=[FeasibilityConfig(name="feas", target=[_accuracy(0.9)])],
        tasks=[{"name": "t", "evaluator": "feas", "sources": ["src"], "extractor": "flat", "matrix": matrix, **task}],
        datasets={"src": ToyImages(), "other": ToyImages(seed=1)},
        extractor=True,
    )


@pytest.mark.parametrize(
    "matrix",
    [
        {"alpha": [0.05, 0.01]},
        {"target": [[_accuracy(0.9).model_dump()], [_accuracy(0.8).model_dump()]]},
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
