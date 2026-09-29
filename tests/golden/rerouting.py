"""The tasks the rerouting golden files record, and how their JSON is normalized before comparing."""

from collections.abc import Callable
from typing import Any

from dataeval_flow import run_task
from dataeval_flow.config import TaskConfig, ViewConfig, ViewOperation
from dataeval_flow.evaluators.quality import DuplicatesConfig
from dataeval_flow.workflows.data_cleaning import DataCleaningConfig
from dataeval_flow.workflows.drift_monitoring import DriftDetectorMMD, DriftMonitoringConfig
from tests.evaluator_toys import ToyImages, shifted_sources, toy_pipeline

_VOLATILE = {"timestamp", "execution_time_s", "tool_version", "execution_time", "execution_duration"}


def normalized(payload: Any) -> Any:
    """`payload` without the fields that differ from run to run or build to build.

    The DataEval build an evaluator result names is one of them: it changes with every lock update.
    """
    if isinstance(payload, dict):
        kept = {key: normalized(value) for key, value in payload.items() if key not in _VOLATILE}
        if isinstance(kept.get("dataeval"), dict):
            kept["dataeval"] = {key: value for key, value in kept["dataeval"].items() if key != "version"}
        return kept
    if isinstance(payload, list):
        return [normalized(item) for item in payload]
    return payload


def _duplicates() -> dict[str, Any]:
    task = TaskConfig(name="t", workflow="dupes", kind="evaluator", sources="src")
    config = toy_pipeline(evaluators=[DuplicatesConfig(name="dupes")], tasks=[task])
    return run_task(task, config, report_images=True).to_dict()


def _cleaning_with_a_view() -> dict[str, Any]:
    clean = DataCleaningConfig(name="clean", outlier_method="zscore", outlier_flags=["pixel"])
    task = TaskConfig(name="t", workflow="clean", sources="src")
    config = toy_pipeline(workflows=[clean], tasks=[task], dataset=ToyImages(count=24))
    view = ViewConfig(
        name="shuffled",
        operations=[
            ViewOperation(type="Shuffle", params={"seed": 3}),
            ViewOperation(type="Limit", params={"size": 16}),
        ],
    )
    config = config.model_copy(
        update={"views": [view], "sources": [config.sources[0].model_copy(update={"view": "shuffled"})]}  # type: ignore[index]
    )
    return run_task(task, config, report_images=True).to_dict()


def _drift() -> dict[str, Any]:
    drift = DriftMonitoringConfig(name="drift", detectors=[DriftDetectorMMD(method="mmd", n_permutations=20)])
    task = TaskConfig(name="t", workflow="drift", sources=["reference", "test"], extractor="flat")
    config = toy_pipeline(workflows=[drift], tasks=[task], datasets=shifted_sources(count=24), extractor=True)
    config = config.model_copy(update={"seed": 0})
    return run_task(task, config, report_images=False).to_dict()


CASES: dict[str, Callable[[], dict[str, Any]]] = {
    "duplicates": _duplicates,
    "cleaning_with_a_view": _cleaning_with_a_view,
    "drift": _drift,
}
