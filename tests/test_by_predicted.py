"""`by: predicted`: a step run once per class a model predicts, on unlabelled data (uncertainty-drift spec §5)."""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from dataeval_flow._blocks import Fields, Table
from dataeval_flow._predictions import Predictions
from dataeval_flow.evaluators import EvaluatorInputs, PerClassOutput
from dataeval_flow.evaluators._per_class import split_keys
from dataeval_flow.evaluators._report import per_class_blocks
from dataeval_flow.evaluators.shift import DriftUnivariateConfig
from dataeval_flow.steps import StepEntry
from dataeval_flow.steps._by import ByConfig
from tests.chain_toys import chain_pipeline
from tests.drift_toys import CLASSES, BoxImages
from tests.evaluator_toys import FLAT
from tests.onnx_toys import DETECTOR, Frames, element, install, model_files, run_uncertainty

_INPUTS = ["reference", {"name": "tests", "list": True}]


def _inputs(*per_source: list[list[float]], names: Any = CLASSES, other_names: Any = None) -> list[EvaluatorInputs]:
    """One input per source in `per_source`, each row's logits as given; the reference names `names`."""
    made = []
    for position, (source, scores) in enumerate(zip(("reference", "cam1"), per_source, strict=False)):
        array = np.asarray(scores, dtype=np.float32)
        predictions = Predictions(
            scores=array, rows=np.arange(len(array)), items=len(array), preds_type="logits", confidence=0.3
        )
        label_names = names if position == 0 else other_names
        made.append(EvaluatorInputs(source=source, index2label=label_names, predictions=predictions))
    return made


_CAT, _DOG, _BIRD = [5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]


def test_by_predicted_is_written_bare_or_with_settings_and_round_trips():
    for written in ("predicted", {"predicted": {"threshold": 0.9}}, {"predicted": {"groups": {"pets": ["cat"]}}}):
        assert ByConfig.model_validate(written).model_dump() == written


def test_by_keys_by_exactly_one_kind():
    with pytest.raises(ValidationError, match="exactly one"):
        ByConfig.model_validate({"class": {}, "predicted": {}})


@pytest.mark.parametrize(
    ("written", "label", "plural"),
    [
        ("class", "class", "classes"),
        ({"class": {"groups": {"a": [0]}}}, "group", "groups"),
        ("predicted", "predicted class", "predicted classes"),
        ({"predicted": {"groups": {"a": [0]}}}, "predicted group", "predicted groups"),
    ],
)
def test_a_key_is_labelled_by_its_kind(written, label, plural):
    by = ByConfig.model_validate(written)
    assert (by.label, by.plural) == (label, plural)


def test_a_checks_by_predicted_takes_no_settings():
    with pytest.raises(ValidationError, match="takes no settings"):
        StepEntry.model_validate(
            {"name": "c", "check": "drift", "input": "ks", "by": {"predicted": {"threshold": 0.5}}}
        )


def test_each_predicted_class_selects_its_rows_named_by_the_reference():
    by = ByConfig.model_validate({"predicted": {"min_items": 1}})
    masks, skipped = split_keys(_inputs([_CAT, _DOG, _CAT], [_DOG, _DOG, _CAT]), by)
    selected = {key: [mask.nonzero()[0].tolist() for mask in per_input] for key, per_input in masks.items()}
    assert selected == {"cat": [[0, 2], [2]], "dog": [[1], [0, 1]]}
    assert skipped == {}


def test_a_class_one_test_source_never_predicts_is_skipped_naming_it():  # Review Focus 3
    _, skipped = split_keys(_inputs([_CAT, _CAT, _BIRD, _BIRD], [_CAT, _CAT]), ByConfig.model_validate("predicted"))
    assert skipped == {"bird": "0 detections in `cam1`, fewer than `min_items` 2"}


def test_names_that_do_not_fit_the_models_outputs_leave_keys_as_indices():
    coco = {1: "cat", 2: "dog", 3: "bird"}
    masks, _ = split_keys(_inputs([_CAT, _CAT], [_CAT, _CAT], names=coco), ByConfig.model_validate("predicted"))
    assert list(masks) == ["0"]


def test_a_group_naming_a_class_when_keys_are_indices_is_refused():
    by = ByConfig.model_validate({"predicted": {"groups": {"pets": ["cat"]}}})
    with pytest.raises(ValueError, match="pets.*cat.*by index"):
        split_keys(_inputs([_CAT, _CAT], [_CAT, _CAT], names=None), by)


@pytest.mark.parametrize("index", [5, -1])
def test_a_group_member_outside_the_models_outputs_is_refused(index):
    by = ByConfig.model_validate({"predicted": {"groups": {"odd": [index]}}})
    with pytest.raises(ValueError, match=rf"`odd`.*{index}.*0\u20262"):
        split_keys(_inputs([_CAT, _CAT], [_CAT, _CAT]), by)


def test_a_test_source_naming_a_class_differently_is_not_refused():
    renamed = {0: "kitty", 1: "dog", 2: "bird"}
    masks, _ = split_keys(
        _inputs([_CAT, _CAT], [_CAT, _CAT], other_names=renamed), ByConfig.model_validate("predicted")
    )
    assert list(masks) == ["cat"]


def test_threshold_decides_how_many_classes_a_close_call_joins():
    close = [5.0, 4.99, 0.0]
    for threshold, keys in ((0.99, ["cat", "dog"]), (1.0, ["cat"])):
        by = ByConfig.model_validate({"predicted": {"threshold": threshold, "min_items": 1}})
        masks, _ = split_keys(_inputs([close], [close]), by)
        assert list(masks) == keys


def test_the_per_class_table_is_headed_by_its_key_label():
    serialized = {"key": "predicted class", "classes": {"cat": {}}, "skipped": {}}
    (table,) = per_class_blocks(serialized, lambda _inner: [Fields(items=[("Drifted", "no")])], detailed=False)
    assert isinstance(table, Table)
    assert table.columns[0].header == "Predicted class"


def test_the_per_class_table_forms_over_unchunked_detection_rows():
    from dataeval_flow.evaluators.shift._report import drift_section

    def inner(count: int, images: int) -> dict[str, Any]:
        rows = {"compared": {"reference": count, "cam1": count}, "images": {"reference": images, "cam1": images}}
        data = {"drifted": False, "distance": 0.1, "threshold": 0.3, "metric_name": "ks", "details": None}
        return {"data": data | {"rows": rows | {"unit": "detections", "confidence": 0.25}}}

    serialized = {"key": "predicted class", "classes": {"cat": inner(40, 20), "dog": inner(30, 15)}, "skipped": {}}
    (table,) = per_class_blocks(serialized, drift_section, detailed=False)
    assert isinstance(table, Table)
    assert [column.header for column in table.columns][:2] == ["Predicted class", "Drifted"]
    compared = next(index for index, column in enumerate(table.columns) if column.header == "Compared")
    assert "`reference` 40 in 20 images" in str(table.rows[0][f"f{compared - 1}"])


def _workflow(by: Any = "predicted", *, check: bool = False) -> dict[str, Any]:
    steps: list[dict[str, Any]] = [{"name": "ks", "evaluator": "ks", "input": ["reference", "tests"], "by": by}]
    if check:
        steps.append({"name": "ks-check", "check": "drift", "input": "ks", "by": "predicted"})
    return {"name": "w", "inputs": _INPUTS, "steps": steps}


def _run(tmp_path, monkeypatch, reference, test, *, chunking=None, **workflow: Any):
    install(monkeypatch, DETECTOR)
    model_files(tmp_path, "IMAGE_OBJECT_DETECTION")
    ks = DriftUnivariateConfig(name="ks", chunking=chunking)
    datasets = {"reference": reference, "cam1": test}
    return run_uncertainty(tmp_path, _workflow(**workflow), [ks], datasets, detector=True)


def test_a_run_by_predicted_class_on_unlabelled_frames_rolls_up_by_predicted_class(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    reference, test = Frames(rng.uniform(0.4, 0.6, 40)), Frames(rng.uniform(0.8, 1.0, 40), seed=1, index2label=None)
    result = _run(tmp_path, monkeypatch, reference, test, check=True)
    output = element(result, "ks").output
    assert isinstance(output, PerClassOutput)
    assert (list(output.outputs), output.label) == (["cat", "dog"], "predicted class")
    (finding,) = element(result, "ks-check").output
    assert finding.title.endswith(" by predicted class")
    assert finding.brief in {"1/2 predicted classes warn", "2/2 predicted classes warn"}


def test_labelled_detection_data_runs_by_predicted_class(tmp_path, monkeypatch):
    install(monkeypatch, DETECTOR)
    model_files(tmp_path, "IMAGE_OBJECT_DETECTION")
    datasets = {"reference": BoxImages(40), "cam1": BoxImages(40, seed=1)}
    result = run_uncertainty(
        tmp_path,
        _workflow(),
        [DriftUnivariateConfig(name="ks")],
        datasets,
        detector=True,
        extractor={"confidence": 0.1},
    )
    assert element(result, "ks").status == "ok"


def test_a_chunked_run_by_predicted_class_chunks_each_class_by_image(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    reference, test = Frames(rng.uniform(0.4, 0.6, 30)), Frames(rng.uniform(0.4, 0.6, 30), seed=1)
    result = _run(tmp_path, monkeypatch, reference, test, chunking={"chunk_count": 3})
    output = element(result, "ks").output
    assert output.outputs["cat"].rows["chunk_images"] == [[0, 9], [10, 19], [20, 29]]


def test_by_class_over_a_detectors_rows_is_refused_pointing_at_predicted(tmp_path, monkeypatch):
    from tests.drift_toys import ClassImages

    install(monkeypatch, DETECTOR)
    model_files(tmp_path, "IMAGE_OBJECT_DETECTION")
    ks = DriftUnivariateConfig(name="ks")
    datasets = {"reference": ClassImages({0: 5, 1: 5}), "cam1": ClassImages({0: 5, 1: 5}, seed=1)}
    result = run_uncertainty(tmp_path, _workflow("class"), [ks], datasets, detector=True)
    assert "use `by: predicted`" in element(result, "ks").errors[0]


def test_by_predicted_without_a_model_extractor_is_refused_at_load():
    from dataeval_flow.config.extractors import UncertaintyExtractorConfig

    unc = UncertaintyExtractorConfig(
        name="unc", model_path="model.onnx", metadata_path="model.json", preds_type="logits"
    )
    data = {"reference": Frames([0.5]), "cam1": Frames([0.5], seed=1)}
    task = {"name": "t", "workflow": "w", "sources": ["reference", "cam1"], "extractor": "flat"}

    def load(step_extractor: str | None) -> None:
        workflow = _workflow()
        if step_extractor is not None:
            workflow["steps"][0]["extractor"] = step_extractor
        chain_pipeline(
            workflows=[workflow],
            evaluators=[DriftUnivariateConfig(name="ks")],
            datasets=data,
            tasks=[task],
            extra={"extractors": [unc, FLAT]},
        )

    with pytest.raises(ValidationError, match="`by: predicted`, which needs a model's predictions"):
        load(None)
    load("unc")


def _bare_kinds(schema: dict[str, Any]) -> set[str]:
    """The bare strings a schema's `anyOf` accepts."""
    return {kind for member in schema["anyOf"] for kind in member.get("enum", [])}


def test_the_generated_schema_accepts_the_bare_by_predicted_on_steps_and_checks():
    from dataeval_flow.config._json_schema import registry_twin

    defs = registry_twin(plugins=False).model_json_schema()["$defs"]
    assert _bare_kinds(defs["ByConfig"]) == {"class", "predicted"}
    assert any(member.get("type") == "object" for member in defs["ByConfig"]["anyOf"])
    assert {"$ref": "#/$defs/ByConfig"} in defs["EvaluatorStep"]["properties"]["by"]["anyOf"]
    checks = {name: branch for name, branch in defs.items() if name.startswith("CheckStep_")}
    assert checks
    for name, branch in checks.items():
        assert _bare_kinds(branch["properties"]["by"]) == {"class", "predicted"}, name
