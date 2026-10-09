"""Shared builders for the audit tests: toy splits and a one-task audit pipeline."""

from __future__ import annotations

from typing import Any

from dataeval_flow import run_tasks
from dataeval_flow.config import PipelineConfig
from dataeval_flow.steps import ChainResult
from verification.functional.chains._toys import Images, pipeline

OUTLIERS = {"outliers": {"flags": ["pixel"], "outlier_threshold": "zscore"}}
READY_CHECKS = {"dimensional-completeness": {"warning": 0.05, "info": 0.06}}

QUESTIONS = [
    "Is the data clean?",
    "Are the labels sound?",
    "Does the data cover what the model must handle?",
    "Could the model learn a shortcut?",
    "Are the splits fit to evaluate on?",
]


class Items:
    """A dataset of exactly the items given, with the metadata of another dataset."""

    def __init__(self, items: list[Any], like: Images) -> None:
        self._items = items
        self.metadata = {**like.metadata, "id": f"items-{len(items)}-of-{like.metadata['id']}"}

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Any:
        return self._items[index]


def clean_pair() -> dict[str, Images]:
    """Train and test with no duplicates, outliers or shared images, and enough items per class."""
    return {"train": Images(80, seed=10, planted=False), "test": Images(60, seed=11, planted=False)}


def leaky_pair() -> dict[str, Any]:
    """Train, and a test split whose first item is train's item 3: one image in two splits."""
    train, fresh = Images(30, seed=1, planted=False), Images(10, seed=2, planted=False)
    return {"train": train, "test": Items([train[3], *(fresh[i] for i in range(len(fresh)))], train)}


def three_splits() -> dict[str, Images]:
    """Train, validation and test, each with a planted duplicate and a white outlier, no image shared."""
    return {"train": Images(12, seed=0), "val": Images(12, seed=1), "test": Images(12, seed=2)}


def audit_pipeline(
    datasets: dict[str, Any],
    entry: dict[str, Any] | None = None,
    *,
    extractor: bool = False,
    extra: dict[str, Any] | None = None,
) -> PipelineConfig:
    """One `audit` task `t` over every dataset, in order, as the sources train, then each evaluation split."""
    task: dict[str, Any] = {"name": "t", "workflow": "w", "sources": list(datasets)}
    if extractor:
        task["extractor"] = "flat"
    workflow = {"name": "w", "type": "audit", **OUTLIERS, **(entry or {})}
    return pipeline(datasets, workflows=[workflow], tasks=[task], extractor=extractor, extra=extra)


def audit(datasets: dict[str, Any], entry: dict[str, Any] | None = None, *, extractor: bool = False) -> ChainResult:
    result = run_tasks(audit_pipeline(datasets, entry, extractor=extractor))["t"]
    assert isinstance(result, ChainResult)
    assert result.success, result.errors
    return result


def section(report: str, heading: str) -> str:
    """The text of the report section headed *heading*, up to the next section."""
    lines = report.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().upper().startswith(heading.upper()))
    body: list[str] = []
    for line in lines[start + 2 :]:
        if line.startswith("=" * 20):
            break
        body.append(line)
    return "\n".join(body)
