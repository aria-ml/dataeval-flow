"""OOD over the `uncertainty` extractor's rows: a classifier's, one per image, and a detector's, one per detection,
judged per image (ood-detection spec §7)."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dataeval.shift import OODKNeighbors
from pydantic import ValidationError

from dataeval_flow._predictions import compute_predictions, uncertainty_rows
from dataeval_flow.config.extractors import UncertaintyExtractorConfig
from dataeval_flow.evaluators.shift import DriftKNeighborsConfig, OODKNeighborsConfig
from dataeval_flow.evaluators.shift._rows import OODRowsOutput
from tests.drift_toys import ClassImages
from tests.onnx_toys import CLASSIFIER, DETECTOR, Frames, element, install, model_files, run_uncertainty

_INPUTS = ["reference", {"name": "tests", "list": True}]
_KNN = OODKNeighborsConfig(name="knn", k=5, distance_metric="euclidean")
_STEPS = [{"name": "knn", "evaluator": "knn", "input": ["reference", "tests"]}]
_WORKFLOW = {"name": "w", "inputs": _INPUTS, "steps": _STEPS}
_MODEL = {"name": "unc", "model_path": "model.onnx", "metadata_path": "model.json", "batch_size": 8}


def _frames(*spans: tuple[int, float], seed: int = 0) -> Frames:
    """Frames in runs of (count, brightness), each jittered by up to 0.05; a brightness of 0.05 holds no detection at
    confidence 0.3, and one of 0.5 holds two."""
    rng = np.random.default_rng(seed + 100)
    brightness = [max(0.0, level + rng.uniform(-0.05, 0.05)) for count, level in spans for _ in range(count)]
    return Frames(brightness, seed=seed)


def _dataeval(here: Path, monkeypatch: pytest.MonkeyPatch, data: list[Any], **fields: Any) -> list[Any]:
    """The predictions DataEval's own predictor gives each of `data` under the stub model."""
    monkeypatch.chdir(here)
    config = UncertaintyExtractorConfig.model_validate(_MODEL | fields)
    return [compute_predictions(item, config, None, 8) for item in data]


def _detections(tmp_path, monkeypatch, reference, test, **options):
    install(monkeypatch, DETECTOR)
    model_files(tmp_path, "IMAGE_OBJECT_DETECTION")
    datasets = {"reference": reference, "cam1": test}
    return run_uncertainty(tmp_path, _WORKFLOW, [_KNN], datasets, detector=True, **options)


def test_classifier_rows_are_flagged_as_dataeval_flags_their_entropies(tmp_path, monkeypatch):
    install(monkeypatch, CLASSIFIER)
    model_files(tmp_path)
    reference, test = ClassImages({0: 20, 1: 20}), ClassImages({0: 10, 1: 10}, seed=1, bright=True)
    result = run_uncertainty(tmp_path, _WORKFLOW, [_KNN], {"reference": reference, "cam1": test}, detector=False)
    output = element(result, "knn").output
    made = _dataeval(tmp_path, monkeypatch, [reference, test], preds_type="logits")
    expected = OODKNeighbors(k=5, distance_metric="euclidean").fit(uncertainty_rows(made[0]))
    expected = expected.predict(uncertainty_rows(made[1]))
    assert not isinstance(output, OODRowsOutput)
    np.testing.assert_array_equal(output.is_ood, expected.is_ood)
    np.testing.assert_allclose(output.instance_score, expected.instance_score, rtol=1e-5)


def test_detector_rows_are_flagged_as_dataeval_flags_them_then_judged_per_image(tmp_path, monkeypatch):
    reference, test = _frames((20, 0.5)), _frames((8, 0.5), (4, 0.95), seed=1)
    extractor = {"confidence": 0.0, "preds_type": "logits"}
    output = element(_detections(tmp_path, monkeypatch, reference, test, extractor=extractor), "knn").output
    made = _dataeval(tmp_path, monkeypatch, [reference, test], confidence=0.0, preds_type="logits")
    expected = OODKNeighbors(k=5, distance_metric="euclidean").fit(uncertainty_rows(made[0]))
    expected = expected.predict(uncertainty_rows(made[1]))
    assert isinstance(output, OODRowsOutput)
    assert output.rows is not None
    assert [row["is_ood"] for row in output.rows["detections"]] == expected.is_ood.tolist()
    np.testing.assert_allclose([row["score"] for row in output.rows["detections"]], expected.instance_score, rtol=1e-5)
    images = made[1].rows
    for image in range(len(test)):
        own = images == image
        assert bool(output.is_ood[image]) == bool(expected.is_ood[own].any())
        assert float(output.instance_score[image]) == pytest.approx(float(expected.instance_score[own].max()), rel=1e-5)
    assert output.rows["unassessed"] == []


def test_an_image_with_no_detection_is_not_assessed(tmp_path, monkeypatch):
    test = _frames((6, 0.5), (3, 0.05), (6, 0.5), seed=1)
    result = _detections(tmp_path, monkeypatch, _frames((20, 0.5)), test)
    output = element(result, "knn").output
    assert output.rows["unassessed"] == [6, 7, 8]
    assert not output.is_ood[6:9].any()
    assert np.isnan(output.instance_score[6:9]).all()
    assert output.rows["compared"] == {"reference": 40, "tests[cam1]": 24}
    assert output.rows["images"] == {"reference": 20, "tests[cam1]": 15}
    assert output.rows["confidence"] == 0.3
    data = result.to_dict()["steps"]["knn"]["elements"]["cam1"]["output"]["data"]  # type: ignore[index]
    assert data["instance_score"][6] is None


@pytest.mark.parametrize("side", ["reference", "test"])
def test_a_source_with_no_detection_fails_the_detector_s_step_naming_it(tmp_path, monkeypatch, side):
    dark, bright = _frames((10, 0.05), seed=1), _frames((10, 0.5))
    reference, test = (dark, bright) if side == "reference" else (bright, dark)
    knn = element(_detections(tmp_path, monkeypatch, reference, test), "knn")
    named = "reference" if side == "reference" else "tests[cam1]"
    assert knn.status == "failed"
    assert f"No detections at `confidence` ≥ 0.3 in `{named}`" in knn.errors[0]


def test_ood_kneighbors_on_uncertainty_must_say_euclidean(tmp_path, monkeypatch):
    install(monkeypatch, CLASSIFIER)
    model_files(tmp_path)
    data = {"reference": ClassImages({0: 4}), "cam1": ClassImages({0: 4}, seed=1)}
    wanted = "step 'knn' embeds with `unc`, and `ood-kneighbors` cannot rank the one number per row"
    with pytest.raises(ValidationError, match=wanted):
        run_uncertainty(tmp_path, _WORKFLOW, [OODKNeighborsConfig(name="knn")], data, detector=False, tasks=True)


def test_drift_kneighbors_on_uncertainty_refuses_cosine_only_where_written(tmp_path, monkeypatch):
    install(monkeypatch, CLASSIFIER)
    model_files(tmp_path)
    data = {"reference": ClassImages({0: 4}), "cam1": ClassImages({0: 4}, seed=1)}
    workflow = {
        "name": "w",
        "inputs": _INPUTS,
        "steps": [{"name": "kd", "evaluator": "kd", "input": _STEPS[0]["input"]}],
    }
    run_uncertainty(tmp_path, workflow, [DriftKNeighborsConfig(name="kd")], data, detector=False, tasks=True)
    cosine = DriftKNeighborsConfig(name="kd", distance_metric="cosine")
    with pytest.raises(ValidationError, match="`drift-kneighbors` cannot rank the one number per row"):
        run_uncertainty(tmp_path, workflow, [cosine], data, detector=False, tasks=True)
