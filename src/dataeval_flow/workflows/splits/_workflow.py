"""The ``splits`` preset: the whole set's labels, the split or folds, each train rebalanced where set, and
each part's labels (data-splitting spec §4). The whole set's class balance and metadata factors are `bias`'s."""

__all__ = ["SplitsWorkflow"]

from typing import Any, ClassVar

from dataeval_flow.evaluators.quality import LabelHealthConfig
from dataeval_flow.steps._port import DataType, Port
from dataeval_flow.steps._result import ChainResult
from dataeval_flow.steps._workflow import InputSlot
from dataeval_flow.steps.transforms._split import SplitConfig
from dataeval_flow.workflows._base import Workflow
from dataeval_flow.workflows._preset import Preset, PresetChain
from dataeval_flow.workflows.splits._config import SplitsConfig

# The `split` step's `val_frac` default.
_VAL_FRAC: float = SplitConfig.model_fields["val_frac"].default
_PARTS = ("train", "val", "test")


class SplitsWorkflow(Preset, Workflow[SplitsConfig, ChainResult]):
    """Splits the task's one source, ``data``, and judges the split.

    The settings expand to:

    - ``label-health`` on the whole set, which ``class-stratification`` compares each part to;
    - ``split`` (``split``, or ``kfold`` with ``folds`` of 2 or more), and ``rebalanced`` (a ``view`` holding
      ``ClassBalance``) on each train where ``rebalance`` is set;
    - ``label-health-<part>`` on each part the settings fill, ``label-health-rebalanced`` where rebalancing, and
      ``class-stratification``, judging the parts before rebalancing.

    Under ``kfold`` every step on a train or val runs once per fold. Run as a step of a custom workflow,
    ``<step>.train`` (the rebalanced train where set), ``<step>.val`` and ``<step>.test`` read the parts; under
    ``kfold``, ``train`` and ``val`` are lists keyed ``"0"`` to ``"k-1"``.
    """

    name: ClassVar[str] = "splits"
    title: ClassVar[str] = "Splits"
    description: ClassVar[str] = (
        "Splits a Dataset into train, val and test, or k folds, and judges each part's stratification; "
        "with `folds` of 2 or more, `train` and `val` are lists keyed by fold."
    )
    slots: ClassVar[tuple[str | InputSlot, ...]] = ("data",)
    outputs: ClassVar[tuple[Port, ...]] = tuple(Port(part, DataType.DATASET) for part in _PARTS)

    @classmethod
    def chain(cls, config: SplitsConfig) -> PresetChain:
        """The whole set's labels, the split, and each part's steps."""
        limits = config.checks
        evaluators: list[Any] = [
            LabelHealthConfig(name="label-health", metadata=config.metadata),
        ]
        steps: list[dict[str, Any]] = [
            {"name": "label-health", "evaluator": "label-health", "input": "data"},
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
                "name": "class-stratification",
                "check": "class-stratification",
                "input": "label-health",
                "parts": [f"label-health-{part}" for part in parts],
                **shown,
                **limits.class_stratification.model_dump(),
            }
        )
        return PresetChain(steps=steps, evaluators=evaluators, outputs=handed)


def _val_frac(config: SplitsConfig) -> float:
    return _VAL_FRAC if config.val_frac is None else config.val_frac


def _split(config: SplitsConfig) -> dict[str, Any]:
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


def _filled(config: SplitsConfig) -> list[str]:
    """The parts these settings fill: `val` is empty with one fold and `val_frac: 0`, `test` with `test_frac: 0`."""
    empty = {"val"} if config.folds == 1 and _val_frac(config) == 0 else set()
    if config.test_frac == 0:
        empty.add("test")
    return [part for part in _PARTS if part not in empty]
