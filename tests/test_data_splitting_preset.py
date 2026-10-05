"""The data-splitting preset: the chain its settings expand to, its refusals, and runs (data-splitting spec §3–§8)."""

import re
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

import dataeval_flow._cache as cache_module
from dataeval_flow import run_tasks
from dataeval_flow._cache import DatasetCache
from dataeval_flow.steps import ChainResult
from dataeval_flow.workflows.data_splitting import DataSplittingConfig, DataSplittingWorkflow
from tests.chain_toys import ToyDetections, chain_pipeline
from tests.evaluator_toys import ToyFactors

_NO_EXTRACTOR = "requires an extractor"


@pytest.fixture(autouse=True)
def _fresh_cache():
    DatasetCache.clear_instances()
    yield
    DatasetCache.clear_instances()


def _steps(**settings: Any) -> list[dict[str, Any]]:
    return [dict(step) for step in DataSplittingWorkflow.chain(DataSplittingConfig(**settings)).steps]


def _names(**settings: Any) -> list[str]:
    return [step["name"] for step in _steps(**settings)]


_WHOLE = ["labels", "labels-check", "balance", "diversity", "coverage", "split"]


def test_the_default_chain() -> None:
    assert _names() == [
        *_WHOLE,
        "labels-train",
        "labels-val",
        "labels-test",
        "stratification",
        "coverage-train",
        "coverage-val",
        "coverage-test",
    ]


def test_rebalancing_adds_a_view_its_labels_and_a_shown_column() -> None:
    chain = DataSplittingWorkflow.chain(DataSplittingConfig(rebalance="interclass"))
    steps = {dict(step)["name"]: dict(step) for step in chain.steps}
    assert steps["rebalance"]["operations"] == [{"type": "ClassBalance", "params": {"method": "interclass"}}]
    assert steps["stratification"]["shown"] == "labels-rebalanced"
    assert steps["coverage-train"]["input"] == "rebalance"
    assert chain.outputs == {"train": "rebalance", "val": "split.val", "test": "split.test"}


def test_naive_coverage_adds_an_uncovered_check_per_coverage_step() -> None:
    names = _names(coverage={"method": "naive"})
    assert [name for name in names if name.startswith("uncovered")] == [
        "uncovered",
        "uncovered-train",
        "uncovered-val",
        "uncovered-test",
    ]


def test_two_folds_or_more_run_kfold() -> None:
    split = next(step for step in _steps(folds=3) if step["name"] == "split")
    assert (split["transform"], split["folds"]) == ("kfold", 3)


@pytest.mark.parametrize(("settings", "gone"), [({"val_frac": 0.0}, "val"), ({"folds": 3, "test_frac": 0.0}, "test")])
def test_a_part_the_settings_leave_empty_gets_no_steps(settings: dict[str, Any], gone: str) -> None:
    names = _names(**settings)
    assert f"labels-{gone}" not in names
    assert f"coverage-{gone}" not in names


def test_val_frac_with_folds_is_refused() -> None:
    with pytest.raises(ValidationError, match=re.escape("applies with `folds: 1` only")):
        DataSplittingConfig(folds=3, val_frac=0.2)


def test_no_folds_are_refused_by_the_field_s_bound() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        DataSplittingConfig(folds=0)


def test_a_whole_dump_of_a_kfold_entry_reloads() -> None:
    entry = DataSplittingConfig(folds=3)
    assert DataSplittingConfig.model_validate(entry.model_dump(mode="json")) == entry


def test_thresholds_are_keyed_by_check_type() -> None:
    entry = DataSplittingConfig.model_validate(
        {"health_thresholds": {"class-imbalance": {"warning": 3}, "uncovered-items": {"warning": 1}}}
    )
    dumped = entry.model_dump(mode="json")["health_thresholds"]
    assert (dumped["class-imbalance"]["warning"], dumped["uncovered-items"]["warning"]) == (3, 1)


def test_a_partial_coverage_keeps_legacy_s_defaults() -> None:
    assert DataSplittingConfig.model_validate({"coverage": {"method": "naive"}}).coverage.num_observations == 50


def _task(dataset: Any, *, extractor: bool = False, **settings: Any) -> ChainResult:
    task: dict[str, Any] = {"name": "t", "workflow": "split", "sources": ["src"]}
    if extractor:
        task["extractor"] = "flat"
    config = chain_pipeline(
        workflows=[{"name": "split", "type": "data-splitting", **settings}],
        tasks=[task],
        datasets={"src": dataset},
        extractor=extractor,
        extra={"seed": 0},
    )
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    return result


def test_a_task_splits_and_records_the_parts() -> None:
    result = _task(ToyFactors(count=60))
    assert result.success, result.errors
    indices = (result.steps["split"].details or {})["indices"]
    assert sorted(i for part in indices.values() for i in part) == list(range(60))
    assert {"Class Imbalance", "Stratification"} <= {finding.title for finding in result.findings}
    assert (result.steps["coverage"].status, result.steps["coverage"].reason) == ("skipped", _NO_EXTRACTOR)


def test_kfold_judges_stratification_per_fold() -> None:
    result = _task(ToyFactors(count=60), folds=3)
    assert result.success, result.errors
    judged = [finding.step for finding in result.findings if finding.title == "Stratification"]
    assert judged == ["stratification[0]", "stratification[1]", "stratification[2]"]


def test_with_an_extractor_the_whole_set_is_embedded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    sizes: list[int] = []
    original = cache_module.get_or_compute_embeddings

    def counting(dataset: Any, *args: Any, **kwargs: Any) -> Any:
        sizes.append(len(dataset))
        return original(dataset, *args, **kwargs)

    monkeypatch.setattr(cache_module, "get_or_compute_embeddings", counting)
    result = _task(ToyFactors(count=90), extractor=True, folds=3, coverage={"num_observations": 3})
    assert result.success, result.errors
    assert sizes == [90]


class _NoFactors:
    """Labelled images whose items carry no metadata at all."""

    def __init__(self, count: int = 30) -> None:
        self._images = [np.full((3, 8, 8), index, dtype=np.uint8) for index in range(count)]
        self.metadata = {"id": f"nofactors-{count}", "index2label": {0: "a", 1: "b"}}

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        target = np.zeros(2, dtype=np.float32)
        target[index % 2] = 1.0
        return self._images[index], target, {}


def test_a_dataset_with_no_factors_still_splits() -> None:
    result = _task(_NoFactors())
    assert result.success, result.errors
    for step in ("balance", "diversity"):
        assert result.steps[step].status == "skipped"
        assert "No factors" in (result.steps[step].reason or "")
    assert result.steps["split"].status == "ok"


def test_too_few_items_to_stratify_fail_the_split() -> None:
    result = _task(ToyFactors(count=6), folds=3)
    assert not result.success
    assert result.steps["split"].status == "failed"
    assert any("stratify" in error for error in result.steps["split"].errors)


def test_holding_out_nothing_is_refused_with_split_s_own_message() -> None:
    with pytest.raises(ValidationError, match=re.escape("`split` holds nothing out")):
        _task(ToyFactors(count=60), test_frac=0.0, val_frac=0.0)


def test_a_split_on_factor_the_metadata_lacks_fails_the_split() -> None:
    result = _task(ToyFactors(count=60), split_on=["scene"])
    assert not result.success
    assert result.steps["split"].status == "failed"
    assert any("not among this metadata's factors" in error for error in result.steps["split"].errors)


def test_a_coverage_step_that_raises_is_skipped_and_the_rest_runs() -> None:
    # 20 neighbors fit in the whole set's 60 items and the train's, not in the val's 4 or the test's 12.
    result = _task(ToyFactors(count=60), extractor=True, coverage={"num_observations": 20})
    assert result.success, result.errors
    assert [result.steps[step].status for step in ("coverage", "coverage-train")] == ["ok", "ok"]
    for step in ("coverage-val", "coverage-test"):
        assert result.steps[step].status == "skipped"
        assert (result.steps[step].reason or "").startswith("failed:")


def _outer(steps: list[dict[str, Any]], dataset: Any, **entry: Any) -> Any:
    return chain_pipeline(
        workflows=[
            {"name": "split", "type": "data-splitting", **entry},
            {"name": "outer", "inputs": ["data"], "steps": steps},
        ],
        evaluators=[{"name": "dupes", "type": "duplicates"}],
        tasks=[{"name": "t", "workflow": "outer", "sources": ["src"]}],
        datasets={"src": dataset},
        extra={"seed": 0},
    )


_SPLITS = {"name": "splits", "workflow": "split", "input": "data"}


def test_a_custom_workflow_reads_the_train() -> None:
    config = _outer([_SPLITS, {"name": "dupes", "evaluator": "dupes", "input": "splits.train"}], ToyFactors(count=60))
    result = run_tasks(config)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    assert result.steps["dupes"].status == "ok"


def test_reading_an_empty_val_is_refused() -> None:
    message = "Step 'dupes' reads `splits.val`, which step 'splits' leaves empty with its settings."
    steps = [_SPLITS, {"name": "dupes", "evaluator": "dupes", "input": "splits.val"}]
    with pytest.raises(ValidationError, match=re.escape(message)):
        _outer(steps, ToyFactors(count=60), val_frac=0.0)


# Two boxes per image, more boxes than images, so DataEval splits it as detection data, stratified per image.
_BOXES = ToyDetections([[index % 2, (index // 2) % 2] for index in range(24)], {0: "car", 1: "person"})


def test_detection_parts_are_exported(tmp_path: Any) -> None:
    steps = [_SPLITS, {"name": "out", "transform": "export", "input": "splits.train"}]
    result = run_tasks(_outer(steps, _BOXES, test_frac=0.25, val_frac=0.25), output_dir=tmp_path)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    train = (result.steps["splits/split"].details or {})["indices"]["train"]
    assert len(train) == 14
    assert result.steps["out"].status == "ok"


def test_each_fold_s_detection_train_is_exported(tmp_path: Any) -> None:
    steps = [_SPLITS, {"name": "out", "transform": "export", "input": "splits.train"}]
    result = run_tasks(_outer(steps, _BOXES, folds=3, test_frac=0.25), output_dir=tmp_path)["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    trains = (result.steps["splits/split"].details or {})["indices"]["train"]
    assert [len(train) for train in trains.values()] == [12, 12, 12]
    assert list(result.steps["out"].elements or {}) == ["0", "1", "2"]
    assert result.steps["out"].status == "ok"
