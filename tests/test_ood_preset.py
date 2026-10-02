"""The `ood-detection` preset: the chain its settings expand to, what load refuses, and what it gives per test
source (ood-detection spec §3, §4, §9)."""

from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import run_task
from dataeval_flow._cache import DatasetCache
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators.shift import OODKNeighborsConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.ood_detection import OODDetectionConfig, OODDetectionWorkflow
from tests.chain_toys import chain_pipeline
from tests.drift_toys import ClassImages
from tests.evaluator_toys import FLAT
from tests.onnx_toys import CLASSIFIER, element, install, model_files, run_uncertainty
from tests.ood_toys import FactorImages

_KNN = {"type": "ood-kneighbors", "k": 5, "distance_metric": "euclidean"}
_DC = {"type": "ood-domain-classifier", "n_folds": 3, "n_repeats": 2}
_LIMITS = {"warning": 10.0, "info": 1.0}


@pytest.fixture(autouse=True)
def _fresh_caches() -> Iterator[None]:
    """Each run reads its own metadata, so each binning warning is raised in the test that causes it."""
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _config(**settings: Any) -> OODDetectionConfig:
    return OODDetectionConfig.model_validate({"name": "ood", **settings})


def _steps(**settings: Any) -> list[dict[str, Any]]:
    return [dict(step) for step in OODDetectionWorkflow.chain(_config(**settings)).steps]


def test_two_detectors_expand_to_their_checks_the_agreement_and_the_factor_steps() -> None:
    steps = _steps(detectors=[_KNN, _DC])
    assert [step["name"] for step in steps] == [
        "ood-kneighbors",
        "ood-kneighbors-check",
        "ood-domain-classifier",
        "ood-domain-classifier-check",
        "agreement",
        "agreement-check",
        "factor-predictors",
        "factor-deviation",
    ]
    assert steps[0] == {"name": "ood-kneighbors", "evaluator": "ood-kneighbors", "input": ["reference", "tests"]}
    assert steps[1] == {
        "name": "ood-kneighbors-check",
        "check": "ood",
        "input": "ood-kneighbors",
        "subject": "OOD (K-Neighbors)",
        **_LIMITS,
    }
    assert steps[4] == {
        "name": "agreement",
        "combine": "ood-union",
        "input": ["ood-kneighbors", "ood-domain-classifier"],
    }
    assert steps[5] == {"name": "agreement-check", "check": "ood-agreement", "input": "agreement", **_LIMITS}
    factors = {"ood": "agreement", "reference": "reference", "input": "tests", "optional": True}
    assert steps[6] == {"name": "factor-predictors", "combine": "factor-predictors", **factors}
    assert steps[7] == {"name": "factor-deviation", "combine": "factor-deviation", **factors, "max_items": 50}


def test_one_detector_has_no_agreement_check_and_insights_off_drop_the_factor_steps() -> None:
    names = [step["name"] for step in _steps(detectors=[_KNN], metadata_insights=False)]
    assert names == ["ood-kneighbors", "ood-kneighbors-check", "agreement"]


def test_settings_reach_their_steps() -> None:
    steps = _steps(
        detectors=[{**_KNN, "name": "unc", "extractor": "yolo-uncertainty"}],
        metadata="m",
        stats="s",
        factor_deviation={"max_items": 5},
        health_thresholds={"ood": {"warning": 20.0}, "ood-agreement": {"info": None}},
    )
    by_name = {step["name"]: step for step in steps}
    assert by_name["unc"]["extractor"] == "yolo-uncertainty"
    assert by_name["unc-check"]["subject"] == "OOD (K-Neighbors) · unc"
    assert (by_name["unc-check"]["warning"], by_name["unc-check"]["info"]) == (20.0, 1.0)
    assert by_name["factor-deviation"]["max_items"] == 5
    assert (by_name["factor-predictors"]["metadata"], by_name["factor-predictors"]["stats"]) == ("m", "s")


def test_the_agreement_thresholds_are_keyed_by_check_type() -> None:
    config = _config(detectors=[_KNN, _DC], health_thresholds={"ood-agreement": {"warning": 50.0}})
    check = next(step for step in OODDetectionWorkflow.chain(config).steps if dict(step)["name"] == "agreement-check")
    assert dict(check)["warning"] == 50.0
    assert config.health_thresholds.ood_agreement.warning == 50.0


def test_a_detector_s_extractor_goes_on_its_step_not_its_evaluator_entry() -> None:
    chain = OODDetectionWorkflow.chain(_config(detectors=[{**_KNN, "name": "unc", "extractor": "yolo"}]))
    (entry,) = chain.evaluators
    assert type(entry) is OODKNeighborsConfig
    assert entry.name == "unc"
    assert dict(chain.steps[0])["extractor"] == "yolo"


@pytest.mark.parametrize(
    ("detector", "wanted"),
    [
        ({"method": "kneighbors"}, "Legacy's `method: kneighbors` is now `type: ood-kneighbors`"),
        ({"k": 5}, "Each detector needs a `type`, one of ood-kneighbors, ood-domain-classifier"),
        ({"type": "drift-mmd"}, "`detectors:` takes ood-kneighbors, ood-domain-classifier entries, not `drift-mmd`"),
    ],
)
def test_a_detector_that_is_not_an_ood_entry_is_refused(detector: dict[str, Any], wanted: str) -> None:
    with pytest.raises(ValidationError, match=wanted):
        _config(detectors=[detector])


def test_two_unnamed_detectors_of_one_type_are_refused() -> None:
    with pytest.raises(ValidationError, match="two detectors named `ood-kneighbors`: give each a distinct `name`"):
        _config(detectors=[_KNN, {**_KNN, "k": 3}])


@pytest.mark.parametrize("name", ["agreement", "factor-predictors", "factor-deviation", "knn-check"])
def test_a_detector_named_as_a_preset_step_is_refused(name: str) -> None:
    with pytest.raises(ValidationError, match=f"Detector `{name}` is a name the preset's own steps use"):
        _config(detectors=[{**_KNN, "name": name}])


@pytest.mark.parametrize(
    "legacy",
    [
        {"health_thresholds": {"ood_pct_warning": 5.0}},
        {"max_ood_insights": 10},
        {"value_range": [0.0, 1.0]},
        {"metadata_auto_bin_method": "clusters"},
    ],
)
def test_legacy_settings_are_refused(legacy: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _config(detectors=[_KNN], **legacy)


def test_a_detector_reading_uncertainty_by_cosine_distance_is_refused_at_load(tmp_path, monkeypatch) -> None:
    install(monkeypatch, CLASSIFIER)
    model_files(tmp_path)
    preset = {
        "name": "ood",
        "type": "ood-detection",
        "detectors": [{"name": "u", "type": "ood-kneighbors", "extractor": "unc"}],
    }
    data = {"reference": ClassImages({0: 4}), "cam1": ClassImages({0: 4}, seed=1)}
    wanted = "step 'u' embeds with `unc`, and `ood-kneighbors` cannot rank the one number per row"
    with pytest.raises(ValidationError, match=wanted):
        run_uncertainty(tmp_path, preset, [], data, detector=False, tasks=True)


def _run(datasets: dict[str, Any], **settings: Any) -> ChainResult:
    preset = {"name": "ood", "type": "ood-detection", "detectors": [_KNN], **settings}
    config = chain_pipeline(workflows=[preset], datasets=datasets, extractor=True)
    result = run_task(TaskConfig(name="t", workflow="ood", sources=list(datasets), extractor="flat"), config)
    assert isinstance(result, ChainResult)
    return result


def test_each_test_source_gets_its_own_findings() -> None:
    shifted = FactorImages(40, seed=1, shifted=range(0, 40, 5))
    with pytest.warns(UserWarning, match="binned automatically"):
        result = _run({"reference": FactorImages(40), "first": shifted, "second": FactorImages(40, seed=2)})
    assert result.success, result.errors
    assert result.type == "ood-detection"
    assert sorted(str(finding.step) for finding in result.findings) == [
        "ood-kneighbors-check[first]",
        "ood-kneighbors-check[second]",
    ]
    assert list(result.steps["factor-predictors"].elements or {}) == ["first", "second"]


def test_a_detector_that_raises_fails_its_step_and_the_task(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataeval.shift import OODKNeighbors

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("cannot fit")

    monkeypatch.setattr(OODKNeighbors, "fit", refuse)
    result = _run({"reference": FactorImages(40), "cam1": FactorImages(40, seed=1)})
    assert not result.success
    assert (result.steps["ood-kneighbors"].elements or {})["cam1"].status == "failed"


def test_a_detector_on_uncertainty_agrees_with_one_on_embeddings(tmp_path, monkeypatch) -> None:  # Review Focus 2
    install(monkeypatch, CLASSIFIER)
    model_files(tmp_path)
    preset = {
        "name": "ood",
        "type": "ood-detection",
        "metadata_insights": False,
        "detectors": [{**_KNN, "name": "flat-knn"}, {**_KNN, "name": "unc-knn", "extractor": "unc"}],
    }
    datasets = {"reference": ClassImages({0: 20, 1: 20}), "cam1": ClassImages({0: 20, 1: 20}, seed=1, bright=True)}
    result = run_uncertainty(tmp_path, preset, [], datasets, detector=False, task_extractor="flat", extractors=[FLAT])
    assert result.success, result.errors
    union = element(result, "agreement").output
    assert (union.detectors, union.images) == (["flat-knn", "unc-knn"], 40)
