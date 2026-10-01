"""A model's predictions reaching drift: cached, each row's entropy as the step's embeddings (uncertainty-drift spec
§3.3, §4.2)."""

import numpy as np
import pytest

from dataeval_flow import run_task
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators.shift import DriftUnivariateConfig
from tests.chain_toys import chain_pipeline
from tests.drift_toys import ClassImages
from tests.onnx_toys import CLASSIFIER, DETECTOR, Frames, element, install, model_files, run_uncertainty

KS = {
    "name": "w",
    "inputs": ["reference", {"name": "tests", "list": True}],
    "steps": [{"name": "ks", "evaluator": "ks", "input": ["reference", "tests"]}],
}


def _ks(tmp_path, reference, test, **options):
    return run_uncertainty(
        tmp_path, KS, [DriftUnivariateConfig(name="ks")], {"reference": reference, "cam1": test}, **options
    )


def test_a_brightened_source_drifts_in_its_classifier_uncertainty(tmp_path, monkeypatch):
    install(monkeypatch, CLASSIFIER)
    model_files(tmp_path)
    reference, test = ClassImages({0: 30, 1: 30}), ClassImages({0: 30, 1: 30}, seed=1, bright=True)
    assert element(_ks(tmp_path, reference, test, detector=False), "ks").output.drifted


def test_a_source_like_the_reference_does_not_drift(tmp_path, monkeypatch):
    install(monkeypatch, CLASSIFIER)
    model_files(tmp_path)
    reference, test = ClassImages({0: 30, 1: 30}), ClassImages({0: 30, 1: 30}, seed=1)
    assert not element(_ks(tmp_path, reference, test, detector=False), "ks").output.drifted


def test_a_brightened_source_drifts_in_its_detection_uncertainty(tmp_path, monkeypatch):
    install(monkeypatch, DETECTOR)
    model_files(tmp_path, "IMAGE_OBJECT_DETECTION")
    rng = np.random.default_rng(0)
    reference, test = Frames(rng.uniform(0.4, 0.6, 40)), Frames(rng.uniform(0.8, 1.0, 40), seed=1)
    assert element(_ks(tmp_path, reference, test, detector=True), "ks").output.drifted


def test_an_evaluator_task_reads_it_too(tmp_path, monkeypatch):
    from dataeval_flow.config.extractors import UncertaintyExtractorConfig

    install(monkeypatch, CLASSIFIER)
    model_files(tmp_path)
    uncertainty = UncertaintyExtractorConfig(
        name="unc", model_path="model.onnx", metadata_path="model.json", preds_type="logits", batch_size=8
    )
    datasets = {"reference": ClassImages({0: 30, 1: 30}), "cam1": ClassImages({0: 30, 1: 30}, seed=1, bright=True)}
    config = chain_pipeline(
        evaluators=[DriftUnivariateConfig(name="ks")], datasets=datasets, extra={"extractors": [uncertainty]}
    )
    result = run_task(
        TaskConfig(name="t", workflow="ks", kind="evaluator", sources=["reference", "cam1"], extractor="unc"),
        config,
        data_dir=tmp_path,
    )
    assert result.success
    assert result.output.drifted  # type: ignore[union-attr]


def test_a_second_run_reads_the_predictions_from_cache(tmp_path, monkeypatch):
    session = install(monkeypatch, CLASSIFIER)
    model_files(tmp_path)
    data = ClassImages({0: 10, 1: 10}), ClassImages({0: 10, 1: 10}, seed=1)
    _ks(tmp_path, *data, detector=False, cache=True)
    calls = len(session.batches)
    assert calls > 0
    assert list((tmp_path / "cache").rglob("predictions_*.npz"))
    _ks(tmp_path, *data, detector=False, cache=True)
    assert len(session.batches) == calls


@pytest.mark.parametrize("change", ["confidence", "model", "metadata"])  # Review Focus 5 is "model"
def test_a_changed_confidence_model_or_metadata_misses_the_cache(tmp_path, monkeypatch, change):
    session = install(monkeypatch, DETECTOR)
    model_files(tmp_path, "IMAGE_OBJECT_DETECTION")
    data = Frames([0.5] * 10), Frames([0.6] * 10, seed=1)
    _ks(tmp_path, *data, detector=True, cache=True)
    calls = len(session.batches)
    assert calls > 0
    extractor = {"confidence": 0.4} if change == "confidence" else {}
    if change == "model":
        (tmp_path / "model.onnx").write_bytes(b"retrained")
    if change == "metadata":
        (tmp_path / "model.json").write_text((tmp_path / "model.json").read_text() + "\n")
    _ks(tmp_path, *data, detector=True, cache=True, extractor=extractor)
    assert len(session.batches) > calls
