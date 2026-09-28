"""Each evaluator gives the same answer as the workflow that makes the same DataEval call on the same data.

The workflow and the evaluator run as two tasks of one pipeline, over the same sources, with settings that mean the
same call. DataEval and torch are seeded before each run, so the random detectors agree too.
"""

import json
from typing import Any

import pytest
import torch
from dataeval.config import set_seed

from dataeval_flow import PipelineConfig, Result, run_task
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators.bias import BalanceConfig, DiversityConfig
from dataeval_flow.evaluators.scope import CoverageConfig, PrioritizeConfig, RepresentationConfig
from dataeval_flow.evaluators.shift import (
    DriftDomainClassifierConfig,
    DriftKNeighborsConfig,
    DriftMMDConfig,
    DriftUnivariateConfig,
    OODDomainClassifierConfig,
    OODKNeighborsConfig,
)
from dataeval_flow.workflows.data_analysis import DataAnalysisConfig, DataAnalysisResult
from dataeval_flow.workflows.data_coverage import DataCoverageConfig, DataCoverageResult
from dataeval_flow.workflows.data_prioritization import DataPrioritizationConfig, DataPrioritizationResult
from dataeval_flow.workflows.drift_monitoring import (
    DriftDetectorDomainClassifier,
    DriftDetectorKNeighbors,
    DriftDetectorMMD,
    DriftDetectorUnivariate,
    DriftMonitoringConfig,
    DriftMonitoringResult,
)
from dataeval_flow.workflows.ood_detection import (
    OODDetectionConfig,
    OODDetectionResult,
    OODDetectorDomainClassifier,
    OODDetectorKNeighbors,
)
from tests.evaluator_toys import ToyFactors, ToyImages, output_json, shifted_sources, toy_pipeline


def _run(config: PipelineConfig, task: TaskConfig) -> "Result[Any, Any]":
    """`task`'s result, with DataEval and torch seeded first."""
    set_seed(0)
    torch.manual_seed(0)
    result = run_task(task, config)
    assert result.success, result.errors
    return result


def _json(value: Any) -> Any:
    """`value` as JSON reads it back, so NumPy and Python numbers compare alike."""
    return json.loads(json.dumps(value, default=float))


def _tasks(workflow: str, evaluator: str, sources: list[str], *, extractor: bool) -> list[TaskConfig]:
    """The workflow's task and the evaluator's, over the same sources and extractor."""
    named = "flat" if extractor else None
    return [
        TaskConfig(name="workflow", workflow=workflow, sources=sources, extractor=named),
        TaskConfig(name="evaluator", workflow=evaluator, sources=sources, kind="evaluator", extractor=named),
    ]


def test_coverage_agrees_with_data_coverage():
    settings = {"num_observations": 5, "min_class_samples": 5, "near_duplicate_factor": 0.5}
    workflow = DataCoverageConfig(
        name="wf", coverage_method="adaptive", coverage_percent=0.01, run_completeness=False, **settings
    )
    evaluator = CoverageConfig(name="ev", method="adaptive", percent=0.01, **settings)
    tasks = _tasks("wf", "ev", ["src"], extractor=True)
    config = toy_pipeline(
        workflows=[workflow], evaluators=[evaluator], tasks=tasks, dataset=ToyImages(count=40), extractor=True
    )
    workflow_result, evaluator_result = _run(config, tasks[0]), _run(config, tasks[1])
    assert isinstance(workflow_result, DataCoverageResult)
    coverage = workflow_result.output.raw.coverage
    assert coverage is not None
    table = output_json(evaluator_result)
    assert table["rows"] == _json(
        [
            {
                "class": row.class_name,
                "count": row.count,
                "uncovered": row.uncovered,
                "uncovered_fraction": row.uncovered_fraction,
                "dispersion": row.dispersion,
                "isotropy": row.isotropy,
                "near_duplicate_fraction": row.near_duplicate_fraction,
                "assessable": row.assessable,
            }
            for row in coverage.per_class
        ]
    )
    assert table["extras"]["coverage_radius"] == pytest.approx(coverage.coverage_radius)
    assert len(table["extras"]["uncovered_indices"]) == coverage.uncovered_count


@pytest.mark.parametrize(
    ("evaluator", "summary", "tables"),
    [
        (BalanceConfig(name="ev"), "balance_summary", ("balance", "factors", "classwise")),
        (DiversityConfig(name="ev", method="shannon"), "diversity_summary", ("factors", "classwise")),
    ],
)
def test_bias_agrees_with_data_analysis(evaluator: Any, summary: str, tables: tuple[str, ...]):
    workflow = DataAnalysisConfig(
        name="wf", outlier_method="zscore", outlier_flags=["pixel"], balance=True, diversity_method="shannon"
    )
    tasks = _tasks("wf", "ev", ["src"], extractor=False)
    config = toy_pipeline(workflows=[workflow], evaluators=[evaluator], tasks=tasks, dataset=ToyFactors())
    workflow_result, evaluator_result = _run(config, tasks[0]), _run(config, tasks[1])
    assert isinstance(workflow_result, DataAnalysisResult)
    expected = getattr(workflow_result.output.raw.splits["src"].bias, summary)
    data = output_json(evaluator_result)["data"]
    for table in tables:
        assert data[table]["rows"] == _json(expected[table]), table


def test_representation_agrees_with_data_coverage():
    toy = ToyImages(count=40)
    toy.metadata["index2label"] = {0: "a", 1: "b", 2: "c"}  # type: ignore[reportTypedDictNotRequiredAccess]
    ontology = {"animal": ["a", "b", "c", "d"]}
    workflow = DataCoverageConfig(name="wf", ontology=ontology, ontology_expected={"a": 0.6}, run_completeness=False)
    evaluator = RepresentationConfig(name="ev", ontology=ontology, expected={"a": 0.6})
    tasks = _tasks("wf", "ev", ["src"], extractor=False)
    config = toy_pipeline(workflows=[workflow], evaluators=[evaluator], tasks=tasks, dataset=toy)
    workflow_result, evaluator_result = _run(config, tasks[0]), _run(config, tasks[1])
    assert isinstance(workflow_result, DataCoverageResult)
    assert workflow_result.output.raw.ontology is not None
    representation = workflow_result.output.raw.ontology.representation
    table = output_json(evaluator_result)
    assert table["rows"] == _json([row.model_dump() for row in representation.worklist])
    assert table["extras"]["leaf_coverage"] == pytest.approx(representation.leaf_coverage)
    assert table["extras"]["total_deficit"] == representation.total_deficit
    assert table["extras"]["violations"]["rows"] == _json([row.model_dump() for row in representation.violations])
    assert table["extras"]["dark_branches"]["rows"] == _json([row.model_dump() for row in representation.dark_branches])


def test_prioritize_agrees_with_data_prioritization():
    """The workflow reads the reference first; the evaluator reads it second."""
    datasets = {"labeled": ToyImages(count=40), "unlabeled": ToyImages(count=40, seed=1)}
    tasks = [
        TaskConfig(name="workflow", workflow="wf", sources=["labeled", "unlabeled"], extractor="flat"),
        TaskConfig(
            name="evaluator", workflow="ev", sources=["unlabeled", "labeled"], kind="evaluator", extractor="flat"
        ),
    ]
    config = toy_pipeline(
        workflows=[DataPrioritizationConfig(name="wf", order="hard_first")],
        evaluators=[PrioritizeConfig(name="ev", order="hard_first")],
        tasks=tasks,
        datasets=datasets,
        extractor=True,
    )
    workflow_result, evaluator_result = _run(config, tasks[0]), _run(config, tasks[1])
    assert isinstance(workflow_result, DataPrioritizationResult)
    (ranking,) = workflow_result.output.raw.prioritizations
    assert output_json(evaluator_result)["data"] == ranking["prioritized_indices"]
    assert output_json(evaluator_result)["extras"]["scores"] == pytest.approx(ranking["scores"])


@pytest.mark.parametrize(
    ("detector", "evaluator", "key"),
    [
        (
            DriftDetectorUnivariate(test="ks", p_val=0.05),
            DriftUnivariateConfig(name="ev", method="ks", p_val=0.05),
            "univariate",
        ),
        (
            DriftDetectorMMD(p_val=0.05, n_permutations=50),
            DriftMMDConfig(name="ev", p_val=0.05, n_permutations=50),
            "mmd",
        ),
        (
            DriftDetectorKNeighbors(k=5, distance_metric="euclidean", p_val=0.05),
            DriftKNeighborsConfig(name="ev", k=5, distance_metric="euclidean", p_val=0.05),
            "kneighbors",
        ),
        (
            DriftDetectorDomainClassifier(n_folds=3, threshold=0.55),
            DriftDomainClassifierConfig(name="ev", n_folds=3, threshold=0.55),
            "domain_classifier",
        ),
    ],
)
def test_drift_agrees_with_drift_monitoring(detector: Any, evaluator: Any, key: str):
    tasks = _tasks("wf", "ev", ["reference", "test"], extractor=True)
    config = toy_pipeline(
        workflows=[DriftMonitoringConfig(name="wf", detectors=[detector])],
        evaluators=[evaluator],
        tasks=tasks,
        datasets=shifted_sources(),
        extractor=True,
    )
    workflow_result, evaluator_result = _run(config, tasks[0]), _run(config, tasks[1])
    assert isinstance(workflow_result, DriftMonitoringResult)
    expected = workflow_result.output.raw.detectors[key]
    data = output_json(evaluator_result)["data"]
    assert data["drifted"] == expected["drifted"]
    assert data["metric_name"] == expected["metric_name"]
    assert data["distance"] == pytest.approx(expected["distance"])
    assert data["threshold"] == pytest.approx(expected["threshold"])
    if "p_val" in expected["details"]:  # type: ignore[reportTypedDictNotRequiredAccess]
        assert data["details"]["p_val"] == pytest.approx(
            expected["details"]["p_val"]  # type: ignore[reportTypedDictNotRequiredAccess]
        )


@pytest.mark.parametrize(
    ("detector", "evaluator", "key"),
    [
        (
            OODDetectorKNeighbors(k=5, distance_metric="euclidean", threshold_perc=95.0),
            OODKNeighborsConfig(name="ev", k=5, distance_metric="euclidean", threshold_perc=95.0),
            "kneighbors",
        ),
        (
            OODDetectorDomainClassifier(n_folds=3, n_repeats=2, n_std=2.0, threshold_perc=95.0),
            OODDomainClassifierConfig(name="ev", n_folds=3, n_repeats=2, n_std=2.0, threshold_perc=95.0),
            "domain_classifier",
        ),
    ],
)
def test_ood_agrees_with_ood_detection(detector: Any, evaluator: Any, key: str):
    tasks = _tasks("wf", "ev", ["reference", "test"], extractor=True)
    config = toy_pipeline(
        workflows=[OODDetectionConfig(name="wf", detectors=[detector])],
        evaluators=[evaluator],
        tasks=tasks,
        datasets=shifted_sources(),
        extractor=True,
    )
    workflow_result, evaluator_result = _run(config, tasks[0]), _run(config, tasks[1])
    assert isinstance(workflow_result, OODDetectionResult)
    samples = sorted(
        workflow_result.output.raw.detectors[key]["samples"],  # type: ignore[reportTypedDictNotRequiredAccess]
        key=lambda sample: sample["index"],
    )
    data = output_json(evaluator_result)["data"]
    assert data["is_ood"] == [sample["is_ood"] for sample in samples]
    assert data["instance_score"] == pytest.approx([sample["score"] for sample in samples])
