"""The ``data-splitting`` preset: the whole set's labels, bias and coverage, the split or folds, each train
rebalanced where set, and each part's labels and coverage (data-splitting spec §4)."""

__all__ = ["DataSplittingWorkflow"]

from typing import Any, ClassVar

from dataeval_flow.evaluators.bias import BalanceConfig, DiversityConfig
from dataeval_flow.evaluators.quality import LabelHealthConfig
from dataeval_flow.evaluators.scope import CoverageConfig
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._preset import Preset, PresetChain
from dataeval_flow.workflows.data_splitting._config import DataSplittingConfig

# Legacy's val share with one fold, where the entry sets none.
_VAL_FRAC = 0.1
_PARTS = ("train", "val", "test")


class DataSplittingWorkflow(Preset, Workflow[DataSplittingConfig, ChainResult]):
    """Splits the task's one source, ``data``, and judges the split.

    The settings expand to:

    - ``label-health`` and ``class-imbalance`` on the whole set; ``balance`` and
      ``diversity``, optional; ``coverage``, optional, which embeds the whole set once for every part, and
      ``uncovered-items`` under ``naive`` coverage;
    - ``split`` (``split``, or ``kfold`` with ``folds`` of 2 or more), and ``rebalanced`` (a ``view`` holding
      ``ClassBalance``) on each train where ``rebalance`` is set;
    - ``label-health-<part>`` on each part the settings fill, ``label-health-rebalanced`` where rebalancing, and
      ``stratification``, judging the parts before rebalancing;
    - ``coverage-<part>`` on each part as handed on, optional, and ``uncovered-items-<part>`` under ``naive`` coverage.

    Under ``kfold`` every step on a train or val runs once per fold. Run as a step of a custom workflow,
    ``<step>.train`` (the rebalanced train where set), ``<step>.val`` and ``<step>.test`` read the parts; under
    ``kfold``, ``train`` and ``val`` are lists keyed ``"0"`` to ``"k-1"``.
    """

    name: ClassVar[str] = "data-splitting"
    title: ClassVar[str] = "Data Splitting"
    description: ClassVar[str] = (
        "Splits a Dataset into train, val and test, or k folds, and judges its balance, stratification and coverage; "
        "with `folds` of 2 or more, `train` and `val` are lists keyed by fold."
    )
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)
    outputs: ClassVar[tuple[Port, ...]] = tuple(Port(part, DataType.DATASET) for part in _PARTS)

    @classmethod
    def chain(cls, config: DataSplittingConfig) -> PresetChain:
        """The whole set's steps, the split, and each part's."""
        limits = config.checks
        naive = config.coverage.method == "naive"
        rate = limits.uncovered_items.warning
        evaluators: list[Any] = [
            LabelHealthConfig(name="label-health", metadata=config.metadata),
            BalanceConfig(name="balance", metadata=config.metadata),
            DiversityConfig(name="diversity", metadata=config.metadata),
            CoverageConfig(name="coverage", **config.coverage.model_dump()),
        ]
        steps: list[dict[str, Any]] = [
            {"name": "label-health", "evaluator": "label-health", "input": "data"},
            {
                "name": "class-imbalance",
                "check": "class-imbalance",
                "input": "label-health",
                "warning": limits.class_imbalance.warning,
            },
            {"name": "balance", "evaluator": "balance", "input": "data", "optional": True},
            {"name": "diversity", "evaluator": "diversity", "input": "data", "optional": True},
            *_coverage("coverage", "data", "uncovered-items", naive, rate),
            _split(config),
        ]
        handed = {part: f"split.{part}" for part in _PARTS}
        if config.rebalance is not None:
            operations = [{"type": "ClassBalance", "params": {"method": config.rebalance}}]
            steps.append({"name": "rebalanced", "transform": "view", "input": "split.train", "operations": operations})
            handed["train"] = "rebalanced"
        parts = _filled(config)
        steps += [
            {"name": f"label-health-{part}", "evaluator": "label-health", "input": f"split.{part}"} for part in parts
        ]
        shown: dict[str, Any] = {}
        if config.rebalance is not None:
            steps.append({"name": "label-health-rebalanced", "evaluator": "label-health", "input": "rebalanced"})
            shown = {"shown": "label-health-rebalanced"}
        steps.append(
            {
                "name": "stratification",
                "check": "stratification",
                "input": "label-health",
                "parts": [f"label-health-{part}" for part in parts],
                **shown,
                **limits.stratification.model_dump(),
            }
        )
        for part in parts:
            steps += _coverage(f"coverage-{part}", handed[part], f"uncovered-items-{part}", naive, rate)
        return PresetChain(steps=steps, evaluators=evaluators, outputs=handed)


def _val_frac(config: DataSplittingConfig) -> float:
    return _VAL_FRAC if config.val_frac is None else config.val_frac


def _split(config: DataSplittingConfig) -> dict[str, Any]:
    """The `split` step: `split` with one fold, `kfold` with more."""
    common = {
        "name": "split",
        "input": "data",
        "test_frac": config.test_frac,
        "stratify": config.stratify,
        "split_on": config.split_on,
        "metadata": config.metadata,
    }
    if config.folds == 1:
        return {**common, "transform": "split", "val_frac": _val_frac(config)}
    return {**common, "transform": "kfold", "folds": config.folds}


def _filled(config: DataSplittingConfig) -> list[str]:
    """The parts these settings fill: `val` is empty with one fold and `val_frac: 0`, `test` with `test_frac: 0`."""
    empty = {"val"} if config.folds == 1 and _val_frac(config) == 0 else set()
    if config.test_frac == 0:
        empty.add("test")
    return [part for part in _PARTS if part not in empty]


def _coverage(name: str, source: str, check: str, naive: bool, rate: float | None) -> list[dict[str, Any]]:
    """A coverage step, optional, and under `naive` coverage the check judging it."""
    steps: list[dict[str, Any]] = [{"name": name, "evaluator": "coverage", "input": source, "optional": True}]
    if naive:
        steps.append({"name": check, "check": "uncovered-items", "input": name, "warning": rate})
    return steps
