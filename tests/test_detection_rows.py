"""Rows that may be detections reach drift only: load refuses every other reader (uncertainty-drift spec §4.1)."""

from typing import Any

import pytest
from pydantic import ValidationError

from dataeval_flow import run_task
from dataeval_flow.config import TaskConfig
from dataeval_flow.config.extractors import UncertaintyExtractorConfig
from dataeval_flow.evaluators.quality import OutliersConfig
from dataeval_flow.evaluators.scope import PrioritizationConfig
from dataeval_flow.evaluators.shift import DriftUnivariateConfig, OODKNeighborsConfig
from dataeval_flow.workflows.drift_monitoring import DriftMonitoringConfig
from tests.chain_toys import chain_pipeline
from tests.drift_toys import ClassImages
from tests.evaluator_toys import FLAT

_UNC = UncertaintyExtractorConfig(name="unc", model_path="model.onnx", metadata_path="model.json", preds_type="logits")
_DATA = {"reference": ClassImages({0: 4}), "cam1": ClassImages({0: 4}, seed=1)}
_INPUTS = ["reference", {"name": "tests", "list": True}]


def _load(*, task: dict[str, Any], evaluators=(), workflows=()):
    return chain_pipeline(
        workflows=workflows, evaluators=evaluators, datasets=_DATA, tasks=[task], extra={"extractors": [_UNC, FLAT]}
    )


def _task(target: str, extractor: str, kind: str = "workflow") -> dict[str, Any]:
    return {"name": "t", kind: target, "sources": ["reference", "cam1"], "extractor": extractor}


def _workflow(step: dict[str, Any]) -> dict[str, Any]:
    return {"name": "w", "inputs": _INPUTS, "steps": [{"name": "s", "input": ["reference", "tests"], **step}]}


def test_an_evaluator_task_reading_one_row_per_item_refuses_a_model_extractor():
    with pytest.raises(ValidationError, match="extractor `unc` runs a model.*only drift and OOD evaluators read it"):
        _load(task=_task("rank", "unc", "evaluator"), evaluators=[PrioritizationConfig(name="rank")])


def test_a_drift_evaluator_task_takes_it():
    _load(task=_task("ks", "unc", "evaluator"), evaluators=[DriftUnivariateConfig(name="ks")])


def test_a_step_naming_a_model_extractor_is_refused_unless_it_drifts():
    with pytest.raises(ValidationError, match="step 's' embeds with `unc`.*only drift and OOD evaluators read it"):
        _load(
            task=_task("w", "flat"),
            evaluators=[PrioritizationConfig(name="rank")],
            workflows=[_workflow({"evaluator": "rank", "extractor": "unc"})],
        )
    _load(
        task=_task("w", "flat"),
        evaluators=[DriftUnivariateConfig(name="ks")],
        workflows=[_workflow({"evaluator": "ks", "extractor": "unc"})],
    )


def test_the_tasks_model_extractor_reaching_a_non_drift_step_is_refused():
    with pytest.raises(ValidationError, match="step 's' embeds with `unc`"):
        _load(
            task=_task("w", "unc"),
            evaluators=[PrioritizationConfig(name="rank")],
            workflows=[_workflow({"evaluator": "rank"})],
        )


def test_a_step_that_embeds_nothing_ignores_the_tasks_model_extractor():
    _load(task=_task("w", "unc"), evaluators=[OutliersConfig(name="out")], workflows=[_workflow({"evaluator": "out"})])


def test_a_drift_monitoring_task_takes_it():
    preset = DriftMonitoringConfig(name="drift", detectors=[{"type": "drift-univariate"}])  # type: ignore[list-item]
    _load(task=_task("drift", "unc"), workflows=[preset])


def test_run_refuses_it_for_a_non_drift_evaluator():
    config = chain_pipeline(
        evaluators=[PrioritizationConfig(name="rank")], datasets=_DATA, extra={"extractors": [_UNC]}
    )
    result = run_task(
        config, TaskConfig(name="t", workflow="rank", kind="evaluator", sources=["reference", "cam1"], extractor="unc")
    )
    assert not result.success
    assert "only drift and OOD evaluators read it" in result.errors[0]


def test_an_ood_evaluator_task_takes_it():
    _load(
        task=_task("knn", "unc", "evaluator"), evaluators=[OODKNeighborsConfig(name="knn", distance_metric="euclidean")]
    )


def test_an_ood_evaluator_task_ranking_by_cosine_distance_is_refused():
    wanted = "runs evaluator 'knn' .*, which cannot rank the one number per row an uncertainty extractor gives"
    with pytest.raises(ValidationError, match=wanted):
        _load(task=_task("knn", "unc", "evaluator"), evaluators=[OODKNeighborsConfig(name="knn")])
