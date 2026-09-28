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
from dataeval_flow.evaluators.scope import CoverageConfig
from dataeval_flow.workflows.data_coverage import DataCoverageConfig, DataCoverageResult
from tests.evaluator_toys import ToyImages, output_json, toy_pipeline


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
