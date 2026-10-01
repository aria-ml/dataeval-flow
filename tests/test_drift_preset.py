"""The drift-monitoring preset: each test source against the reference, detectors as evaluator entries (§10.11)."""

from collections.abc import Mapping
from typing import Any, cast

import pytest

from dataeval_flow import run_task
from dataeval_flow.config import TaskConfig
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.drift_monitoring import DriftMonitoringConfig, DriftMonitoringWorkflow
from tests.chain_toys import chain_pipeline
from tests.drift_toys import BoxImages
from tests.evaluator_toys import ToyImages

# Verdicts below hold on the CPU; an unset device runs on CUDA where torch sees it, and CI has none.
_CPU = {"device": "cpu"}


def _chain(**settings: Any) -> tuple[list[Mapping[str, Any]], list[Any]]:
    """The steps and evaluator entries an entry with `settings` expands to."""
    chain = DriftMonitoringWorkflow.chain(DriftMonitoringConfig.model_validate({"name": "drift", **settings}))
    return [cast("Mapping[str, Any]", step) for step in chain.steps], list(chain.evaluators)


def test_each_detector_is_a_step_and_its_check():
    steps, _ = _chain(detectors=[{"type": "drift-mmd"}, {"name": "ks", "type": "drift-univariate"}])
    assert [step["name"] for step in steps] == ["drift-mmd", "drift-mmd-check", "ks", "ks-check"]
    assert steps[0]["input"] == ["reference", "tests"]


def test_classwise_runs_follow_every_whole_set_detector():
    steps, _ = _chain(detectors=[{"type": "drift-mmd"}, {"type": "drift-kneighbors"}], classwise=["drift-mmd"])
    names = [step["name"] for step in steps]
    assert names[-2:] == ["drift-mmd-classes", "drift-mmd-classes-check"]
    classes = steps[-2]
    assert (classes["by"], classes["optional"], classes["evaluator"]) == ("class", True, "drift-mmd")
    assert steps[-1]["subject"] == "Drift (MMD)"


def test_a_chunked_classwise_detector_runs_an_unchunked_copy():
    steps, evaluators = _chain(
        detectors=[{"name": "ks", "type": "drift-univariate", "chunking": {"chunk_count": 5}}], classwise=["ks"]
    )
    copy = next(entry for entry in evaluators if entry.name == "ks-unchunked")
    assert copy.chunking is None
    assert steps[-2]["evaluator"] == "ks-unchunked"
    assert steps[-1]["subject"] == "Drift (Univariate) · ks"


def test_health_thresholds_reach_every_check():
    steps, _ = _chain(
        detectors=[{"type": "drift-mmd"}],
        classwise=["drift-mmd"],
        health_thresholds={"drift": {"chunk_percent": 25.0}},
    )
    assert all(step["chunk_percent"] == 25.0 for step in steps if step.get("check") == "drift")


def _preset_run(datasets: dict[str, Any], **settings: Any) -> ChainResult:
    """The preset over `datasets`, the first the reference and the rest its tests, with the flatten extractor."""
    entry = {
        "name": "drift",
        "type": "drift-monitoring",
        "detectors": [{"type": "drift-kneighbors", "k": 3}],
        **settings,
    }
    config = chain_pipeline(workflows=[entry], datasets=datasets, extractor=True, extra=_CPU)
    result = run_task(TaskConfig(name="t", workflow="drift", sources=list(datasets), extractor="flat"), config)
    assert isinstance(result, ChainResult)
    return result


def test_each_test_source_is_tested_on_its_own():
    result = _preset_run(
        {"reference": ToyImages(40), "cam1": ToyImages(40, seed=1, bright=True), "cam2": ToyImages(40, seed=2)}
    )
    assert list(result.steps["drift-kneighbors"].elements or {}) == ["cam1", "cam2"]
    checks = result.steps["drift-kneighbors-check"].elements or {}
    severities = {key: element.output[0].severity for key, element in checks.items()}
    assert severities == {"cam1": "warning", "cam2": "ok"}


def test_a_detector_that_raises_fails_its_step_and_the_task_but_not_the_others():  # §10.6's visible failures
    # A reference of 40 items cannot split into the 3 chunks of 1000 that chunking needs, so DataEval raises.
    detectors = [
        {"type": "drift-kneighbors", "k": 3},
        {"name": "big", "type": "drift-mmd", "chunking": {"chunk_size": 1000}},
    ]
    result = _preset_run({"reference": ToyImages(40), "cam1": ToyImages(40, seed=1, bright=True)}, detectors=detectors)
    assert result.steps["big"].status == "failed"
    assert not result.success
    assert (result.steps["drift-kneighbors-check"].elements or {})["cam1"].output[0].severity == "warning"


def test_an_empty_test_source_fails_only_its_own_element():  # Review Focus 1
    result = _preset_run({"reference": ToyImages(40), "cam1": ToyImages(0), "cam2": ToyImages(40, seed=1, bright=True)})
    elements = result.steps["drift-kneighbors"].elements or {}
    assert elements["cam1"].status == "failed"
    assert elements["cam2"].status == "ok"
    assert (result.steps["drift-kneighbors-check"].elements or {})["cam2"].output[0].severity == "warning"


def test_classwise_on_unlabelled_data_is_not_assessed():  # Review Focus 3, through the preset
    datasets = {"reference": ToyImages(40, labeled=False), "cam1": ToyImages(40, seed=1, labeled=False)}
    result = _preset_run(datasets, classwise=["drift-kneighbors"])
    assert (result.steps["drift-kneighbors-classes"].elements or {})["cam1"].status == "skipped"
    assert (result.steps["drift-kneighbors-check"].elements or {})["cam1"].output[0].severity == "ok"
    (finding,) = (result.steps["drift-kneighbors-classes-check"].elements or {})["cam1"].output
    assert (finding.severity, finding.brief) == ("info", "not assessed")


def test_the_crop_recipe_runs_the_preset_on_detections():
    preset = {
        "name": "drift",
        "type": "drift-monitoring",
        "detectors": [{"type": "drift-kneighbors", "k": 3}],
        "classwise": ["drift-kneighbors"],
    }
    recipe = {
        "name": "object_drift",
        "inputs": ["reference", {"name": "tests", "list": True}],
        "steps": [
            {"name": "ref-crops", "transform": "wrap", "input": "reference", "wrapper": "DetectionCrops"},
            {"name": "test-crops", "transform": "wrap", "input": "tests", "wrapper": "DetectionCrops"},
            {"name": "drift", "workflow": "drift", "input": ["ref-crops", "test-crops"]},
        ],
    }
    datasets = {"reference": BoxImages(40), "cam1": BoxImages(40, seed=1, bright=True)}
    config = chain_pipeline(workflows=[preset, recipe], datasets=datasets, extractor=True, extra=_CPU)
    task = TaskConfig(name="t", workflow="object_drift", sources=["reference", "cam1"], extractor="flat")
    # The binning record reads the crops' metadata, where DataEval bins the `source_id` DetectionCrops adds to each.
    with pytest.warns(UserWarning, match="`source_id` was binned automatically"):
        result = run_task(task, config)
    assert isinstance(result, ChainResult)
    (finding,) = (result.steps["drift/drift-kneighbors-classes-check"].elements or {})["cam1"].output
    assert finding.title == "Drift (K-Neighbors) by class"
    assert finding.brief.endswith("/3 classes warn")
