"""A preset's prepared copy, run in its task's place, runs as the preset does (preset decomposition spec §5.4)."""

from typing import Any

import pytest

from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow._service._decompose import DecomposeReady, DecomposeRequest, prepare
from dataeval_flow.config import PipelineConfig
from dataeval_flow.steps import ChainResult, StepResult
from tests.chain_toys import chain_pipeline
from tests.evaluator_toys import ToyImages
from tests.golden.coverage import CoverageImages
from tests.test_audit_preset import _OUTLIERS
from tests.test_chain_verdict_report import _outline
from tests.test_naming_conventions import _MINIMAL
from tests.test_preset_expansion_parity import _outcome
from tests.triage_toys import MixedWeightDataset

_TWENTY = {"src": ToyImages(count=20)}
_SHIFT = {"reference": ToyImages(count=20), "test": ToyImages(count=20, seed=1)}
_MMD, _KNN = _MINIMAL["shift"]["detectors"][:2]
_FACTORS = {"src": CoverageImages()}
_SPLITS = {"train": CoverageImages(), "test": CoverageImages(count=40)}

# (type, settings, datasets, task extras): every built-in, and the structural branches that change its steps.
# Data with metadata factors (`CoverageImages`) where a preset judges them, so its factor steps run rather than skip.
_CASES: dict[str, tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = {
    "audit-default": ("audit", _OUTLIERS, _SPLITS, {}),
    "audit-ontology": ("audit", {**_OUTLIERS, "ontology": {"animal": {"cat": None}}}, _FACTORS, {}),
    "audit-factor-leakage": ("audit", {**_OUTLIERS, "factor-leakage": {"factors": ["site"]}}, _SPLITS, {}),
    "audit-no-gaps": ("audit", {**_OUTLIERS, "factor-gaps": False}, _SPLITS, {}),
    "audit-naive-coverage": ("audit", {**_OUTLIERS, "coverage": {"method": "naive"}}, _SPLITS, {"extractor": "flat"}),
    "bias": ("bias", {}, _FACTORS, {}),
    "bias-no-gaps": ("bias", {"factor-gaps": False}, _FACTORS, {}),
    "prioritization": (
        "prioritization",
        {},
        {"ref": ToyImages(count=16), "pool": ToyImages(count=20, seed=1)},
        {"extractor": "flat"},
    ),
    "quality": ("quality", _MINIMAL["quality"], _TWENTY, {}),
    "scope": ("scope", {}, _FACTORS, {"extractor": "flat"}),
    "scope-naive": ("scope", {"coverage": {"method": "naive"}}, _FACTORS, {"extractor": "flat"}),
    "scope-no-completeness": ("scope", {"completeness": False}, _FACTORS, {"extractor": "flat"}),
    "shift": ("shift", _MINIMAL["shift"], _SHIFT, {"extractor": "flat"}),
    "shift-drift-only": ("shift", {"detectors": [_MMD]}, _SHIFT, {"extractor": "flat"}),
    "shift-one-ood": (
        "shift",
        {"detectors": [_KNN], "factor-predictors": False, "factor-deviation": False},
        _SHIFT,
        {"extractor": "flat"},
    ),
    "splits": ("splits", {}, {"src": ToyImages(count=40)}, {}),
    "splits-folds": ("splits", {"folds": 2}, {"src": ToyImages(count=40)}, {}),
    "splits-no-val": ("splits", {"val_frac": 0, "rebalance": "interclass"}, {"src": ToyImages(count=40)}, {}),
    "splits-no-test": ("splits", {"test_frac": 0}, {"src": ToyImages(count=40)}, {}),
    "taxonomy": ("taxonomy", _MINIMAL["taxonomy"], _TWENTY, {}),
    "triage": ("triage", {}, {"src": MixedWeightDataset()}, {}),
}


@pytest.fixture(autouse=True)
def _fresh_caches():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _run(pipeline: PipelineConfig) -> ChainResult:
    DatasetCache.clear_instances()
    result = run_tasks(pipeline)["t"]
    assert isinstance(result, ChainResult)
    return result


def _draft(
    datasets: dict[str, Any], *, stand_in: bool = False, extra: dict[str, Any] | None = None, **parts: Any
) -> PipelineConfig:
    """The pipeline over the in-memory `datasets`; with `stand_in`, over same-named on-disk ones JSON can carry."""
    extra = extra or {}
    if stand_in:  # preparation reads no data, so a path nothing is at serves
        extra = {
            "datasets": [{"name": f"{name}_data", "format": "coco", "path": name} for name in datasets],
            "sources": [{"name": name, "dataset": f"{name}_data"} for name in datasets],
            **extra,
        }
    return chain_pipeline(datasets={} if stand_in else datasets, extra=extra, **parts)


def _failed(record: StepResult) -> bool:
    """Whether a step failed, or skipped because its evaluator failed."""
    return record.status == "failed" or (record.reason or "").startswith("failed:")


def _selections(result: ChainResult) -> dict[str, str]:
    """Each selection a step made, by where it is: what Prioritization ranks, and what Splits partitions to."""
    from dataeval.data import View

    found: dict[str, str] = {}

    def walk(value: Any, at: str) -> None:
        if isinstance(value, View):
            found[at] = repr(value.operation_groups)
        elif isinstance(value, dict):
            for key, inner in value.items():
                walk(inner, f"{at}/{key}")

    for name, record in result.steps.items():
        walk(record.output, name)
        walk({key: element.output for key, element in (record.elements or {}).items()}, name)
    return found


@pytest.mark.parametrize("case", sorted(_CASES))
def test_a_prepared_copy_runs_as_its_preset_does(case: str) -> None:
    type_id, settings, datasets, task = _CASES[case]
    parts = {
        "workflows": [{"name": "p", "type": type_id, **settings}],
        "tasks": [{"name": "t", "workflow": "p", "sources": list(datasets), **task}],
        "extractor": "extractor" in task,
    }
    request = DecomposeRequest.model_validate(
        {
            "pipeline": {},
            "operation": "copy_definition",
            "target_pointer": "/workflows/0",
            "new_name": "c",
            "reassign_tasks": ["t"],
        }
    )
    answer = prepare(_draft(datasets, stand_in=True, graphs=False, **parts), request)
    assert isinstance(answer, DecomposeReady), answer
    assert answer.valid, answer.diagnostics

    preset_result = _run(_draft(datasets, **parts))
    assert preset_result.success, preset_result.errors
    failed = [name for name, record in preset_result.steps.items() if _failed(record)]
    assert not failed, failed
    assert preset_result.findings or _selections(preset_result)  # Prioritization makes no findings, by design

    candidate = {key: value for key, value in answer.pipeline.items() if key not in ("datasets", "sources")}
    copy_result = _run(_draft(datasets, extra=candidate, **parts))
    assert copy_result.type == "c"  # the copy ran, not the preset
    assert _outcome(copy_result) == _outcome(preset_result)
    assert _outline(copy_result) == _outline(preset_result)
    assert _selections(copy_result) == _selections(preset_result)
