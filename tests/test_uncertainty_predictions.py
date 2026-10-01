"""The `uncertainty` extractor's model run: each row's class scores and the item it came from (uncertainty-drift spec
§3)."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from dataeval_flow._predictions import Predictions, compute_predictions, membership, runs_model, uncertainty_rows
from dataeval_flow.config.extractors import FlattenExtractorConfig, UncertaintyExtractorConfig
from tests.drift_toys import ClassImages
from tests.onnx_toys import CLASSIFIER, DETECTOR, N_BOXES, Frames, install, model_files


def _config(**fields: Any) -> UncertaintyExtractorConfig:
    data = {"name": "unc", "model_path": "model.onnx", "metadata_path": "model.json", "preds_type": "logits"}
    return UncertaintyExtractorConfig.model_validate(data | fields)


@pytest.fixture
def here(tmp_path, monkeypatch):
    """Run in `tmp_path`, where the model files are written: a config's paths are relative."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_classifier_rows_are_dataevals_uncertainty_one_per_image(here, monkeypatch):
    from dataeval.extractors import ScoresExtractor, UncertaintyExtractor
    from dataeval.models import OnnxImageClassifier

    install(monkeypatch, CLASSIFIER)
    model_files(here)
    images = ClassImages({0: 5, 1: 5})
    predictions = compute_predictions(images, _config(), None, 4)
    scores = ScoresExtractor(OnnxImageClassifier("model.onnx", "model.json"))
    expected = UncertaintyExtractor(scores, preds_type="logits")(images)
    assert predictions.rows is None
    assert predictions.items == 10
    np.testing.assert_allclose(uncertainty_rows(predictions), expected, rtol=1e-5)


def test_detector_rows_at_confidence_zero_are_dataevals_in_order(here, monkeypatch):
    from dataeval.extractors import ScoresExtractor, UncertaintyExtractor
    from dataeval.models import OnnxObjectDetector

    install(monkeypatch, DETECTOR)
    model_files(here, "IMAGE_OBJECT_DETECTION")
    frames = Frames([0.2, 0.5, 0.8])
    predictions = compute_predictions(frames, _config(confidence=0.0), None, 2)
    scores = ScoresExtractor(OnnxObjectDetector("model.onnx", "model.json"))(frames)
    np.testing.assert_allclose(predictions.scores, scores, rtol=1e-6)
    assert predictions.rows is not None
    assert predictions.rows.tolist() == [0] * N_BOXES + [1] * N_BOXES + [2] * N_BOXES
    expected = UncertaintyExtractor(lambda s: s, preds_type="logits")(scores)
    np.testing.assert_allclose(uncertainty_rows(predictions), expected, rtol=1e-5)


def test_confidence_drops_the_padded_boxes_and_each_row_keeps_its_image(here, monkeypatch):
    install(monkeypatch, DETECTOR)
    model_files(here, "IMAGE_OBJECT_DETECTION")
    predictions = compute_predictions(Frames([0.5, 0.05, 0.9]), _config(preds_type="sigmoid", confidence=0.3), None, 2)
    # Each bright frame keeps its two real boxes; the dark frame keeps none; the zero-padded boxes go.
    assert predictions.rows is not None
    assert predictions.rows.tolist() == [0, 0, 2, 2]
    assert (predictions.items, predictions.confidence) == (3, 0.3)


def test_sigmoid_scores_become_the_logits_dataeval_reads(here, monkeypatch):
    from dataeval.extractors import UncertaintyExtractor
    from dataeval.models import OnnxObjectDetector

    install(monkeypatch, DETECTOR)
    model_files(here, "IMAGE_OBJECT_DETECTION")
    frames = Frames([0.5, 0.9])
    predictions = compute_predictions(frames, _config(preds_type="sigmoid", confidence=0.3), None, 2)
    targets = OnnxObjectDetector("model.onnx", "model.json")([frames[i][0] for i in range(len(frames))])
    raw = np.concatenate([np.asarray(target.scores) for target in targets])
    logits = np.log(raw[raw.max(axis=1) >= 0.3] / (1 - raw[raw.max(axis=1) >= 0.3]))
    assert predictions.preds_type == "logits"
    np.testing.assert_allclose(predictions.scores, logits, rtol=1e-4)
    expected = UncertaintyExtractor(lambda s: s, preds_type="logits")(logits)
    np.testing.assert_allclose(uncertainty_rows(predictions), expected, rtol=1e-4)


def test_a_fixed_batch_size_pads_the_last_batch_and_drops_its_predictions(here, monkeypatch):
    session = install(monkeypatch, CLASSIFIER)
    model_files(here, batch_size=4)
    predictions = compute_predictions(ClassImages({0: 3, 1: 3}), _config(), None, None)
    assert session.batches == [4, 4]
    assert len(predictions.scores) == 6


def test_a_batch_size_the_fixed_batch_size_disagrees_with_is_refused_before_inference(here, monkeypatch):
    session = install(monkeypatch, CLASSIFIER)
    model_files(here, batch_size=4)
    with pytest.raises(ValueError, match="batches of 4 exactly"):
        compute_predictions(ClassImages({0: 3}), _config(batch_size=8), None, 8)
    assert session.batches == []


@pytest.mark.parametrize(
    ("task", "fields", "n_classes", "match"),
    [
        ("IMAGE_OBJECT_DETECTION", {}, 3, "set `confidence`"),
        ("IMAGE_CLASSIFICATION", {"confidence": 0.5}, 3, "remove `confidence`"),
        ("IMAGE_CLASSIFICATION", {}, 1, "at least 2"),
    ],
)
def test_what_cannot_run_is_refused_before_inference(here, monkeypatch, task, fields, n_classes, match):
    session = install(monkeypatch, DETECTOR if task == "IMAGE_OBJECT_DETECTION" else CLASSIFIER)
    model_files(here, task, n_classes=n_classes)
    with pytest.raises(ValueError, match=match):
        compute_predictions(Frames([0.5]), _config(**fields), None, 4)
    assert session.batches == []


def test_metadata_for_another_task_is_refused(here, monkeypatch):
    install(monkeypatch, CLASSIFIER)
    model_files(here, "IMAGE_SEGMENTATION")
    with pytest.raises(ValueError, match="io.interface"):
        compute_predictions(Frames([0.5]), _config(), None, 4)


def test_probs_that_do_not_sum_to_one_are_refused_naming_sigmoid(here, monkeypatch):
    install(monkeypatch, DETECTOR)
    model_files(here, "IMAGE_OBJECT_DETECTION")
    with pytest.raises(ValueError, match="`sigmoid`"):
        compute_predictions(Frames([0.5]), _config(preds_type="probs", confidence=0.3), None, 4)


def test_membership_reproduces_dataevals_classwise_arrays():
    from dataeval.extractors import ClasswiseUncertaintyExtractor

    scores = np.random.default_rng(0).normal(0, 2, (40, 4)).astype(np.float32)
    mask = membership(scores, 0.9)
    expected = ClasswiseUncertaintyExtractor(lambda s: s, preds_type="logits", threshold=0.9)(scores)
    assert sorted(expected) == [cls for cls in range(4) if mask[:, cls].any()]
    for cls, entropies in expected.items():
        own = Predictions(scores=scores[mask[:, cls]], rows=None, items=40, preds_type="logits", confidence=None)
        np.testing.assert_allclose(uncertainty_rows(own), entropies, rtol=1e-6)


def test_predictions_round_trip_through_their_arrays():
    made = Predictions(
        scores=np.ones((3, 2), dtype=np.float32), rows=np.array([0, 0, 2]), items=4, preds_type="logits", confidence=0.3
    )
    back = Predictions.from_arrays(made.to_arrays(), _config(preds_type="sigmoid", confidence=0.3))
    assert (back.rows.tolist(), back.items, back.preds_type, back.confidence) == ([0, 0, 2], 4, "logits", 0.3)  # type: ignore[union-attr]
    np.testing.assert_array_equal(back.scores, made.scores)


def test_only_an_uncertainty_entry_runs_a_model():
    assert runs_model(_config())
    assert not runs_model(FlattenExtractorConfig(name="flat"))
    assert not runs_model(None)


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"metadata_path": None}, "metadata_path"),
        ({"preds_type": None}, "preds_type"),
        ({"image_height": 16}, "together"),
        ({"metadata_path": "/models/model.json"}, "relative"),
    ],
)
def test_the_config_requires_its_metadata_and_output_format(fields, match):
    data = {"name": "unc", "model_path": "model.onnx", "metadata_path": "model.json", "preds_type": "logits"} | fields
    with pytest.raises(ValidationError, match=match):
        UncertaintyExtractorConfig.model_validate({key: value for key, value in data.items() if value is not None})


def test_both_paths_resolve_against_the_data_root(tmp_path):
    from dataeval_flow._orchestrator import _resolve_extractor_paths

    model_files(tmp_path)
    resolved = _resolve_extractor_paths(_config(), tmp_path)
    assert (resolved.model_path, resolved.metadata_path) == (str(tmp_path / "model.onnx"), str(tmp_path / "model.json"))


def test_the_embeddings_path_refuses_it_naming_drift():
    from dataeval_flow._embeddings import build_embeddings

    with pytest.raises(ValueError, match="only drift evaluators"):
        build_embeddings(MagicMock(), _config())
