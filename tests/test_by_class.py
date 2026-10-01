"""`by: class` on evaluate steps: one run per class or group, inside one PerClassOutput (spec §5.9)."""

from collections.abc import Sequence
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dataeval_flow import run_task
from dataeval_flow._blocks import Paragraph, Table
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators import PerClassOutput
from dataeval_flow.evaluators.shift import DriftKNeighborsConfig
from dataeval_flow.steps import ChainResult, StepEntry, StepResult
from tests.chain_toys import chain_pipeline, register_toys
from tests.drift_toys import BoxImages, ClassImages


def _run(
    by: Any, reference: Any, test: Any, *, optional: bool = False, steps: Sequence[dict[str, Any]] = ()
) -> ChainResult:
    """A custom workflow over `reference` and one test source `cam1`: `drift-kneighbors` with `by`, then `steps`."""
    knn: dict[str, Any] = {"name": "knn", "evaluator": "knn", "input": ["reference", "tests"]}
    if by is not None:
        knn["by"] = by
    if optional:
        knn["optional"] = True
    workflow = {
        "name": "per_class",
        "inputs": ["reference", {"name": "tests", "list": True}],
        "steps": [knn, *steps],
    }
    config = chain_pipeline(
        workflows=[workflow],
        evaluators=[DriftKNeighborsConfig(name="knn", k=3)],
        datasets={"reference": reference, "cam1": test},
        extractor=True,
    )
    result = run_task(
        TaskConfig(name="t", workflow="per_class", sources=["reference", "cam1"], extractor="flat"), config
    )
    assert isinstance(result, ChainResult)
    return result


def _knn(result: ChainResult) -> StepResult:
    """The `knn` step's run on `cam1`, the one element of `tests`."""
    elements = result.steps["knn"].elements
    assert elements is not None
    return elements["cam1"]


def test_each_class_runs_on_its_own_items_in_class_order():
    result = _run("class", ClassImages({0: 12, 1: 12, 2: 12}), ClassImages({0: 12, 1: 12, 2: 12}, seed=1))
    output = _knn(result).output
    assert isinstance(output, PerClassOutput)
    assert list(output.outputs) == ["cat", "dog", "bird"]
    assert output.skipped == {}


def test_each_key_selects_its_own_items_on_each_input():
    import numpy as np

    from dataeval_flow.evaluators import EvaluatorInputs
    from dataeval_flow.evaluators._per_class import split_keys
    from dataeval_flow.steps._by import ByConfig

    names = {0: "cat", 1: "dog"}
    inputs = [
        EvaluatorInputs(source="a", labels=np.array([0, 1, 0, 1]), index2label=names),
        EvaluatorInputs(source="b", labels=np.array([1, 1, 0, 0, 0]), index2label=names),
    ]
    masks, skipped = split_keys(inputs, ByConfig.model_validate("class"))
    selected = {key: [mask.nonzero()[0].tolist() for mask in per_input] for key, per_input in masks.items()}
    assert selected == {"cat": [[0, 2], [2, 3, 4]], "dog": [[1, 3], [0, 1]]}
    assert skipped == {}


def test_a_class_under_min_items_on_one_side_is_skipped_with_why():
    result = _run("class", ClassImages({0: 12, 1: 12, 2: 12}), ClassImages({0: 12, 1: 12, 2: 1}, seed=1))
    output = _knn(result).output
    assert list(output.outputs) == ["cat", "dog"]
    assert output.skipped == {"bird": "1 item in `tests[cam1]`, fewer than `min_items` 2"}


def test_a_class_missing_from_the_test_source_is_skipped_as_zero_items():
    result = _run("class", ClassImages({0: 12, 1: 12, 2: 12}), ClassImages({0: 12, 1: 12}, seed=1))
    assert _knn(result).output.skipped == {"bird": "0 items in `tests[cam1]`, fewer than `min_items` 2"}


def test_a_class_whose_run_raises_is_skipped_with_its_error_and_the_rest_still_run():
    # `k` is 3, so a class of 3 reference items is too small for `drift-kneighbors`, which raises.
    result = _run("class", ClassImages({0: 12, 1: 12, 2: 3}), ClassImages({0: 12, 1: 12, 2: 3}, seed=1))
    element = _knn(result)
    assert element.status == "ok"
    assert list(element.output.outputs) == ["cat", "dog"]
    assert list(element.output.skipped) == ["bird"]
    assert "k (3) must be less than" in element.output.skipped["bird"]


def test_every_class_raising_fails_the_step_with_the_first_error():
    element = _knn(_run("class", ClassImages({0: 3, 1: 3}), ClassImages({0: 3, 1: 3}, seed=1)))
    assert element.status == "failed"
    assert "k (3) must be less than" in element.errors[0]


def test_groups_key_by_name_may_overlap_and_list_classes_in_no_group():
    by = {"class": {"groups": {"pets": ["cat", "dog"], "felines": [0]}}}
    data = ClassImages({0: 12, 1: 12, 2: 12})
    output = _knn(_run(by, data, ClassImages({0: 12, 1: 12, 2: 12}, seed=1))).output
    assert list(output.outputs) == ["pets", "felines"]
    assert output.label == "group"
    assert output.skipped == {"bird": "in no group"}


def test_a_group_naming_no_class_fails_the_step_naming_both():
    by = {"class": {"groups": {"vehicles": ["lorry"]}}}
    result = _run(by, ClassImages({0: 12, 1: 12}), ClassImages({0: 12, 1: 12}, seed=1))
    element = _knn(result)
    assert element.status == "failed"
    assert "vehicles" in element.errors[0]
    assert "lorry" in element.errors[0]


def test_detection_data_fails_the_step_so_optional_skips_it():
    result = _run("class", BoxImages(), BoxImages(seed=1), optional=True)
    element = _knn(result)
    assert element.status == "skipped"
    assert "one label per item" in (element.reason or "")


def test_unlabelled_data_fails_the_step_so_optional_skips_it():
    reference = ClassImages({0: 12, 1: 12}, labeled=False)
    result = _run("class", reference, ClassImages({0: 12, 1: 12}, seed=1, labeled=False), optional=True)
    assert _knn(result).status == "skipped"


def test_a_shared_index_named_differently_fails_conform_it_first():
    # Its own seed: the cache keys a Dataset by its items, not its names, so a renamed copy of another test's Dataset
    # would be read with that Dataset's names.
    renamed = ClassImages({0: 12, 1: 12}, seed=2)
    renamed.metadata["index2label"] = {0: "cat", 1: "puppy", 2: "bird"}
    element = _knn(_run("class", ClassImages({0: 12, 1: 12}), renamed))
    assert element.status == "failed"
    assert "conform it first" in element.errors[0]


def test_per_class_json_holds_each_classs_own_json():
    result = _run("class", ClassImages({0: 12, 1: 12, 2: 1}), ClassImages({0: 12, 1: 12, 2: 12}, seed=1))
    run = _knn(result).result
    assert run is not None
    body = cast(dict[str, Any], run.to_dict()["output"])
    assert (body["shape"], body["key"]) == ("per_class", "class")
    assert set(body["classes"]) == {"cat", "dog"}
    assert body["classes"]["cat"]["shape"] == "mapping"
    assert body["skipped"] == {"bird": "1 item in `reference`, fewer than `min_items` 2"}


def test_per_class_drift_reports_one_table_of_its_classes_then_the_skipped():
    result = _run("class", ClassImages({0: 12, 1: 12, 2: 1}), ClassImages({0: 12, 1: 12, 2: 12}, seed=1))
    run = _knn(result).result
    assert run is not None
    table, skipped = run._report_output(detailed=False)
    assert isinstance(table, Table)
    assert [row["key"] for row in table.rows] == ["cat", "dog"]
    assert skipped == Paragraph(text="Not assessed: bird (1 item in `reference`, fewer than `min_items` 2).")


def test_by_is_refused_on_a_transform_and_settings_on_a_check():
    with pytest.raises(ValueError, match="by"):
        StepEntry.model_validate({"name": "v", "transform": "view", "input": "a", "view": "x", "by": "class"})
    with pytest.raises(ValueError, match="no settings"):
        StepEntry.model_validate(
            {"name": "c", "check": "outlier-rate", "input": "o", "by": {"class": {"min_items": 3}}}
        )


def test_by_never_reaches_an_inline_steps_settings():
    entry = StepEntry.model_validate({"name": "c", "check": "outlier-rate", "input": "o", "by": "class"})
    assert "by" not in entry.settings


def test_by_is_refused_on_an_evaluator_reading_stats():
    from dataeval_flow.evaluators.quality import OutliersConfig

    workflow = {
        "name": "w",
        "inputs": ["data"],
        "steps": [{"name": "o", "evaluator": "outliers", "input": "data", "by": "class"}],
    }
    # The graph is built when the pipeline loads, so its refusal surfaces as a validation error there.
    with pytest.raises(ValidationError, match="slices embeddings and labels, and `outliers` reads stats"):
        chain_pipeline(
            workflows=[workflow],
            evaluators=[OutliersConfig(name="outliers")],
            datasets={"data": ClassImages({0: 4, 1: 4})},
        )


def test_by_round_trips_through_save():
    entry = StepEntry.model_validate({"name": "k", "evaluator": "knn", "input": ["a", "b"], "by": "class"})
    assert entry.model_dump()["by"] == "class"
    grouped = {"class": {"groups": {"pets": ["cat", "dog"]}}}
    entry = StepEntry.model_validate({"name": "k", "evaluator": "knn", "input": ["a", "b"], "by": grouped})
    assert entry.model_dump()["by"] == grouped
    assert StepEntry.model_validate(entry.model_dump()).by == entry.by


_CHECK = [{"name": "knn-check", "check": "toy-drifted", "input": "knn", "by": "class"}]


def test_a_check_with_by_rolls_its_per_class_findings_into_one(plugins):
    register_toys(plugins)
    counts = {0: 15, 1: 15, 2: 15}
    result = _run("class", ClassImages(counts), ClassImages(counts, seed=1, bright_classes={2}), steps=_CHECK)
    elements = result.steps["knn-check"].elements
    assert elements is not None
    (finding,) = elements["cam1"].output
    assert finding.title == "Drifted by class"
    assert finding.severity == "warning"
    assert finding.brief == "1/3 classes warn"
    assert [block.text for block in finding.blocks] == ["Warned: bird."]


def test_skipped_keys_are_named_with_why_in_the_rollup(plugins):
    register_toys(plugins)
    result = _run("class", ClassImages({0: 15, 1: 15, 2: 15}), ClassImages({0: 15, 1: 15, 2: 1}, seed=1), steps=_CHECK)
    elements = result.steps["knn-check"].elements
    assert elements is not None
    (finding,) = elements["cam1"].output
    expected = "Not assessed: bird (1 item in `tests[cam1]`, fewer than `min_items` 2)."
    assert expected in [b.text for b in finding.blocks]


def test_nothing_assessed_is_not_assessed():
    from dataeval_flow.steps._by import ByConfig, roll_up

    finding = roll_up({}, {"cat": "0 items in `cam1`, fewer than `min_items` 2"}, title="Drifted", by=ByConfig())
    assert (finding.severity, finding.title, finding.brief) == ("info", "Drifted by class", "not assessed")


def test_groups_roll_up_by_group():
    from dataeval_flow.steps._by import ByConfig, roll_up
    from dataeval_flow.workflows import Finding

    by = ByConfig.model_validate({"class": {"groups": {"pets": ["cat"]}}})
    ok = Finding(severity="ok", title="Drifted")
    assert roll_up({"pets": [ok]}, {}, title="Drifted", by=by).brief == "0/1 groups warn"


def test_a_per_class_output_is_refused_to_a_check_without_by(plugins):
    register_toys(plugins)
    steps = [{"name": "knn-check", "check": "toy-drifted", "input": "knn"}]
    with pytest.raises(ValidationError, match="per-class"):
        _run("class", ClassImages({0: 4, 1: 4}), ClassImages({0: 4, 1: 4}, seed=1), steps=steps)


def test_a_check_with_by_is_refused_on_an_output_without_it(plugins):
    register_toys(plugins)
    with pytest.raises(ValidationError, match="one Output"):
        _run(None, ClassImages({0: 4, 1: 4}), ClassImages({0: 4, 1: 4}, seed=1), steps=_CHECK)


def test_a_check_with_by_and_more_than_one_input_is_refused_at_load():
    with pytest.raises(ValidationError, match="maps a check over one input"):
        StepEntry.model_validate(
            {"name": "c", "check": "target-outlier-rate", "input": "a", "labels": "b", "by": "class"}
        )
