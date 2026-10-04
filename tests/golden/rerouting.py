"""The tasks the rerouting golden files record, and how their JSON is normalized before comparing."""

from collections.abc import Callable
from typing import Any

import pytest

from dataeval_flow import run_task
from dataeval_flow.config import TaskConfig
from dataeval_flow.evaluators.quality import DuplicatesConfig
from tests.evaluator_toys import toy_pipeline

_VOLATILE = {
    "timestamp",
    "execution_time_s",
    "tool_version",
    "library_versions",
    "execution_time",
    "execution_duration",
}


def normalized(payload: Any) -> Any:
    """`payload` without the fields that differ from run to run or build to build.

    The DataEval build an evaluator result names is one of them: it changes with every lock update. So are each
    thumbnail's encoded bytes, which the image codec's build decides.
    """
    if isinstance(payload, dict):
        kept = {key: normalized(value) for key, value in payload.items() if key not in _VOLATILE}
        if isinstance(kept.get("dataeval"), dict):
            kept["dataeval"] = {key: value for key, value in kept["dataeval"].items() if key != "version"}
        if isinstance(kept.get("assets"), list):
            kept["assets"] = [
                {key: value for key, value in asset.items() if key != "data"} if isinstance(asset, dict) else asset
                for asset in kept["assets"]
            ]
        return kept
    if isinstance(payload, list):
        return [normalized(item) for item in payload]
    return payload


def approximately(expected: Any) -> Any:
    """`expected` with each float compared to a relative 1e-4, and everything else exactly.

    torch's float32 arithmetic differs in its last digits from one CPU to another, so a drift distance recorded on
    one machine reads differently on a CI runner. The golden files pin what the rerouting could change: every key,
    string, count and flag, and each measurement to four significant figures.
    """
    if isinstance(expected, dict):
        return {key: approximately(value) for key, value in expected.items()}
    if isinstance(expected, list):
        return [approximately(item) for item in expected]
    if isinstance(expected, float):
        return pytest.approx(expected, rel=1e-4)
    return expected


def _duplicates() -> dict[str, Any]:
    task = TaskConfig(name="t", workflow="dupes", kind="evaluator", sources="src")
    config = toy_pipeline(evaluators=[DuplicatesConfig(name="dupes")], tasks=[task])
    return run_task(task, config, report_images=True).to_dict()


CASES: dict[str, Callable[[], dict[str, Any]]] = {
    "duplicates": _duplicates,
}
