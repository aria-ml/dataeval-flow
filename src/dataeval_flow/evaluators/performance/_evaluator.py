"""The ``sufficiency`` evaluator: DataEval's Sufficiency over a training set and a test set, with a mounted trainer."""

__all__ = ["SufficiencyEvaluator"]

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from dataeval.performance import Sufficiency, SufficiencyOutput

from dataeval_flow._input_spec import InputKind
from dataeval_flow.evaluators._evaluator import Evaluator
from dataeval_flow.evaluators._fields import dataeval_arguments, require
from dataeval_flow.evaluators._inputs import EvaluatorInputs
from dataeval_flow.evaluators.performance._config import SufficiencyConfig

_logger: logging.Logger = logging.getLogger(__name__)

# DataEval's default schedule starts at 1% of the training set, so a smaller one has a first size of 0 items.
_DEFAULT_SCHEDULE_FLOOR = 100


def _log_progress(
    step: int,
    *,
    total: int | None = None,
    desc: str | None = None,  # noqa: ARG001
    extra_info: Mapping[str, Any] | None = None,
) -> None:
    """Log a line for each training size DataEval measures. A size can train for minutes, and the line is the only sign
    of life in a service run's logs, which run at INFO. Metrics that are not single numbers stay in the output."""
    info = extra_info or {}
    metrics = ", ".join(f"{name}={value:.4g}" for name, value in info.get("metrics", {}).items() if np.ndim(value) == 0)
    _logger.info(
        "Sufficiency %d/%s: run %d of %d, %d items%s",
        step,
        total,
        info["run"] + 1,
        info["runs"],
        info["size"],
        f", {metrics}" if metrics else "",
    )


@dataclass(repr=False)
class _TrainedSufficiencyOutput(SufficiencyOutput):
    """DataEval's output, with the trainer that made it, which ``data()`` leaves to the extras with the fit's cache."""

    trainer: dict[str, str] = field(default_factory=dict)

    def data(self) -> dict[str, Any]:
        """DataEval's measurements alone: not ``trainer``, an extra, nor the private cache of the curve fit."""
        return {key: value for key, value in super().data().items() if not key.startswith("_") and key != "trainer"}


class SufficiencyEvaluator(Evaluator[SufficiencyConfig, SufficiencyOutput]):
    """``sufficiency``: how a model's metrics grow with its training set, per DataEval's Sufficiency."""

    name: ClassVar[str] = "sufficiency"
    title: ClassVar[str] = "Sufficiency"
    description: ClassVar[str] = (
        "How a trained model's metrics grow with its training data: the first source trains, the second tests "
        "(DataEval Sufficiency)."
    )
    experimental: ClassVar[bool] = True  # a new contract: the trainer file and how it loads may change
    dataeval_class: ClassVar[type] = Sufficiency
    dataeval_methods: ClassVar[Mapping[InputKind, str]] = {InputKind.DATASET: "evaluate"}
    output_extras: ClassVar[tuple[str, ...]] = ("trainer", "params")
    reads_factors: ClassVar[bool] = False

    def run(self, config: SufficiencyConfig, inputs: Sequence[EvaluatorInputs]) -> SufficiencyOutput:
        """Check the schedule against the training set, load the trainer, and measure; the trainer is unloaded after.

        The power law is fitted once, when the output is serialized, so ``extras.params`` holds the parameters the
        ``sufficiency-target`` check projects from.
        """
        from dataeval_flow.config._loader import running_data_dir
        from dataeval_flow.evaluators.performance._trainer import load_trainer

        train, test = inputs
        train_data = require(train.dataset, "its dataset", train.source)
        test_data = require(test.dataset, "its dataset", test.source)
        _check_schedule(config.schedule, len(train_data), train.source)
        arguments = dataeval_arguments(config)
        schedule = arguments.pop("schedule", None)
        with load_trainer(config.trainer, running_data_dir()) as (kit, trainer):
            sufficiency = Sufficiency(
                kit.model,
                training_strategy=kit.training_strategy,
                evaluation_strategy=kit.evaluation_strategy,
                reset_strategy=kit.reset_strategy,
                **arguments,
            )
            measured = sufficiency.evaluate(train_data, test_data, schedule=schedule, progress_callback=_log_progress)
        output = _TrainedSufficiencyOutput(
            steps=measured.steps,
            measures=measured.measures,
            averaged_measures=measured.averaged_measures,
            unit_interval=measured.unit_interval,
            trainer=trainer,
        )
        object.__setattr__(output, "_meta", measured.meta())  # DataEval's record of the call, as `set_metadata` sets it
        return output


def _check_schedule(schedule: list[int] | None, size: int, source: str) -> None:
    """Refuse a size DataEval would train on without measuring it: one past the training set, or, with no schedule, a
    default schedule whose first size is 0 items. DataEval checks neither."""
    if schedule is None:
        if size < _DEFAULT_SCHEDULE_FLOOR:
            raise ValueError(
                f"The training set, source '{source}', holds {size} items: the default schedule needs at least "
                f"{_DEFAULT_SCHEDULE_FLOOR}, or its first size is 0 items. Set `schedule`."
            )
        return
    over = [step for step in schedule if step > size]
    if over:
        raise ValueError(
            f"`schedule` asks for {over}, past the {size} items of the training set, source '{source}'; DataEval would "
            "train on all of them and record the larger size."
        )
